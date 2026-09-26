"""Freeze a bounded MPI-PCB normal-bank pilot; no model work or test decoding."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes, audit_members
from scripts.preflight_mpi_pcb_normals import normal_members
from scripts.preflight_mpi_pcb_normals_v3 import validate_pool_report
from scripts.preflight_mpi_pcb_normals_v4 import EXPECTED_POOL_REPORT_SHA256
from scripts.preflight_mpi_pcb_models import (
    EXPECTED_ARCHIVE_SHA256, EXPECTED_V4_SHA256, sha256_file, validate_v4_report,
)

EVALUATION_SHA256 = "768714d947673b1b03f91782de802ce33f8796c21af582a5e5e078d172e2315e"
REGIONS = [f"grid{row}-{col}_s1024" for row in range(1, 5) for col in range(1, 4)]
EXPECTED_CONFIG = {
    "schema_version": 1,
    "purpose": "post_test_exploratory_target_normal_bank_preparation",
    "selection_salt": "mpi-pcb-normal-bank-pilot-v1",
    "training_original_photos": 64, "calibration_original_photos": 169,
    "regions_per_photo": 12,
    "invalid_photo_policy": "exclude_whole_photo_using_reviewed_pool_audit",
    "cross_split_duplicate_policy": "exclude_fit_photo_if_any_tile_matches_calibration_or_reviewed_test_rgb",
    "fit_duplicate_policy": "deduplicate_full_12_region_photos_report_partial_tile_duplicates",
    "memory_bank_scope": "one_shared_bank_for_12_regions",
    "backbone": "wide_resnet50_2", "layers": ["layer1", "layer2"],
    "pre_trained": True, "seed": 42, "image_size": 256,
    "preprocessing": "stretch_256_bilinear_antialias_imagenet", "augmentation": "none",
    "feature_dimension": 768, "feature_grid_side": 64, "spatial_patch_stride": 2,
    "coreset_sampling_ratio": 0.1, "maximum_candidate_embeddings": 900000,
    "num_neighbors": 9, "region_score": "mean_of_top_1pct_anomaly_map",
    "photo_score": "maximum_of_12_region_scores",
    "calibration_quantile": 0.95, "calibration_quantile_method": "higher",
    "test_previously_observed": True, "automatic_pass_allowed": False,
    "confirmatory_claim_allowed": False,
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if canonical(config) != canonical(EXPECTED_CONFIG):
        raise ValueError("Pilot configuration differs from the frozen v1 rules")
    return config


def patch_budget(config: dict) -> dict:
    images = config["training_original_photos"] * config["regions_per_photo"]
    per_image = len(range(0, config["feature_grid_side"], config["spatial_patch_stride"])) ** 2
    candidates = images * per_image
    if candidates > config["maximum_candidate_embeddings"]:
        raise ValueError("Candidate budget exceeded; no automatic reduction is permitted")
    bank = int(candidates * config["coreset_sampling_ratio"])
    return {
        "training_tiles": images, "native_patches_per_tile": config["feature_grid_side"] ** 2,
        "retained_patches_per_tile": per_image, "candidate_embeddings": candidates,
        "coreset_embeddings": bank, "feature_dimension": config["feature_dimension"],
        "candidate_float32_bytes": candidates * config["feature_dimension"] * 4,
        "bank_float32_bytes": bank * config["feature_dimension"] * 4,
        "peak_memory_measured": False,
        "note": "Tensor arithmetic only; excludes backbone, projection, distance buffers and copies. No peak-memory or runtime guarantee.",
    }


def read_test_fingerprints(handoff: Path) -> dict[str, list[str]]:
    """Read only existing audit JSON. Neither labels, masks nor scores select fit data."""
    if sha256_file(handoff) != EVALUATION_SHA256:
        raise ValueError("Reviewed external-evaluation-v2 ZIP checksum differs")
    with zipfile.ZipFile(handoff) as source:
        audit = json.loads(source.read("test-audit.json"))
    photos = audit["photos"]
    expected = {f"{kind}/{index:04d}" for kind in ("good", "defect") for index in range(55)}
    if len(photos) != 110 or {photo["key"] for photo in photos} != expected:
        raise ValueError("Reviewed test fingerprint photo inventory differs")
    fingerprints = defaultdict(list)
    for photo in photos:
        rows = photo["regions"]
        if len(rows) != len(REGIONS) or {row["region"] for row in rows} != set(REGIONS):
            raise ValueError("Reviewed test fingerprint region inventory differs")
        for row in rows:
            rgb = row["decoded_rgb_sha256"]
            if not isinstance(rgb, str) or len(rgb) != 64 or any(c not in "0123456789abcdef" for c in rgb):
                raise ValueError("Invalid reviewed test RGB fingerprint")
            fingerprints[rgb].append(f'{photo["key"]}/{row["region"]}')
    return dict(fingerprints)


def scan_normals(source: zipfile.ZipFile, normal: dict, complete_ids: list[str]) -> list[dict]:
    """Normal payload access is centralized here, and guarded before every read."""
    rows = []
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    for region in sorted(normal):
        for index, photo_id in enumerate(sorted(complete_ids), start=1):
            info = normal[region][photo_id]
            expected = f"extracted/{region}/train/good/{photo_id}.png"
            if info.filename != expected or info.file_size > 8 * 1024 * 1024:
                raise ValueError(f"Unexpected normal member: {info.filename}")
            raw = source.read(info)  # ZIP CRC check; never a test or mask member.
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
                        raise ValueError("Expected RGB PNG 256x256")
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    rgb = hashlib.sha256(image.tobytes()).hexdigest()
            except (OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f"Normal integrity failure; no repair/exclusion performed: {expected}: {exc}") from exc
            rows.append({
                "region": region, "photo_id": photo_id, "member": expected,
                "member_sha256": hashlib.sha256(raw).hexdigest(), "decoded_rgb_sha256": rgb,
                "width": 256, "height": 256,
            })
            if index % 250 == 0 or index == len(complete_ids):
                print(f"{region} normal-only fingerprint audit {index}/{len(complete_ids)}", flush=True)
    return rows


def build_split(rows: list[dict], calibration: dict, test_rgb: dict, config: dict) -> dict:
    """Group by original photo; keep all 12 regions on the same side of the split."""
    regions = REGIONS
    by_photo = defaultdict(dict)
    for row in rows:
        photo_id, region = row["photo_id"], row["region"]
        if region not in regions or region in by_photo[photo_id]:
            raise ValueError("Unknown or duplicate normal region")
        by_photo[photo_id][region] = row
    if any(set(group) != set(regions) for group in by_photo.values()):
        raise ValueError("Incomplete normal photo group")
    calibration_ids = calibration["calibration_photo_ids"]
    if len(set(calibration_ids)) != config["calibration_original_photos"]:
        raise ValueError("Calibration photo count differs")
    calibration_rows = [by_photo[p][r] for p in calibration_ids for r in regions]
    expected_rows = {(r["photo_id"], r["region"]): r for r in calibration["calibration_rows"]}
    if len(expected_rows) != len(calibration_rows):
        raise ValueError("Calibration row count differs")
    if any(row != expected_rows[(row["photo_id"], row["region"])] for row in calibration_rows):
        raise ValueError("Calibration bytes/pixels differ from reviewed v4; cohort unchanged")
    calibration_rgb = defaultdict(list)
    for row in calibration_rows:
        calibration_rgb[row["decoded_rgb_sha256"]].append(f'{row["photo_id"]}/{row["region"]}')
    order = sorted(by_photo, key=lambda p: (hashlib.sha256(
        f'{config["selection_salt"]}\0{p}'.encode()).hexdigest(), p))
    excluded, eligible, seen_composites = [], [], {}
    for photo_id in order:
        if photo_id in calibration_ids:
            continue
        group = by_photo[photo_id]
        fingerprints = tuple(group[r]["decoded_rgb_sha256"] for r in regions)
        matches = []
        for region, rgb in zip(regions, fingerprints, strict=True):
            for source, mapping in (("calibration", calibration_rgb), ("reviewed_test", test_rgb)):
                if rgb in mapping:
                    matches.append({"region": region, "source": source, "matches": mapping[rgb]})
        if matches:
            excluded.append({"photo_id": photo_id, "reason": "exact_cross_split_rgb", "tiles": matches})
        elif fingerprints in seen_composites:
            excluded.append({"photo_id": photo_id, "reason": "identical_full_photo",
                             "retained_photo_id": seen_composites[fingerprints]})
        else:
            seen_composites[fingerprints] = photo_id
            eligible.append(photo_id)
    ready = len(eligible) >= config["training_original_photos"]
    selected = eligible[:config["training_original_photos"]] if ready else []
    train_rows = [by_photo[p][r] for p in selected for r in regions]
    tile_groups = defaultdict(list)
    for row in train_rows:
        tile_groups[row["decoded_rgb_sha256"]].append(f'{row["photo_id"]}/{row["region"]}')
    return {
        "status": "normal_bank_preparation_ready_for_review" if ready else "normal_bank_preparation_needs_review_insufficient_unique_photos",
        "training_photo_ids": selected, "training_photo_ids_sha256": digest(selected),
        "training_rows": train_rows, "calibration_photo_ids": calibration_ids,
        "calibration_rows": calibration_rows,
        "eligible_original_photos": len(eligible), "excluded_fit_candidates": excluded,
        "partial_tile_duplicate_groups_in_fit": [v for v in tile_groups.values() if len(v) > 1],
        "calibration_test_exact_tile_overlap": [
            {"calibration_tiles": references, "test_tiles": test_rgb[rgb]}
            for rgb, references in sorted(calibration_rgb.items()) if rgb in test_rgb
        ],
        "duplicate_scope_note": "Exact decoded RGB only; not proof of independent boards, sessions or near-duplicate absence. Existing calibration/test overlap is reported, not altered.",
    }


def implementation_identity(project: Path) -> dict:
    files = [Path(__file__), *[project / "scripts" / name for name in (
        "audit_mpi_pcb_archive.py", "audit_mpi_pcb_normal_pool.py",
        "preflight_mpi_pcb_normals.py", "preflight_mpi_pcb_normals_v3.py",
        "preflight_mpi_pcb_normals_v4.py", "preflight_mpi_pcb_models.py",
    )]]
    result = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(project), "status", "--porcelain"], capture_output=True, text=True)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "source_files_sha256": {p.name: sha256_file(p) for p in files}}


def read_completed(output: Path, identity: dict) -> dict:
    with zipfile.ZipFile(output) as source:
        manifest = json.loads(source.read("handoff-manifest.json"))
        expected = {"preparation.json", "normal-fingerprints.json", "config.json"}
        if set(manifest) != expected or len(source.namelist()) != 4 or set(source.namelist()) != expected | {"handoff-manifest.json"}:
            raise ValueError("Existing preparation ZIP member set differs; preserved")
        for name, expected_hash in manifest.items():
            if hashlib.sha256(source.read(name)).hexdigest() != expected_hash:
                raise ValueError("Existing preparation checksum differs; preserved")
        report = json.loads(source.read("preparation.json"))
    if report.get("input_identity") != identity:
        raise ValueError("Existing preparation uses different inputs/code; preserved. Share the existing ZIP for review.")
    return report


def write_handoff(output: Path, report: dict, rows: list[dict], config: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mpi-normal-preparation-", suffix=".zip", dir=output.parent)
    os.close(fd)
    staging = Path(name)
    try:
        data = {"preparation.json": canonical(report), "normal-fingerprints.json": canonical(rows), "config.json": canonical(config)}
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as destination:
            for member, content in data.items():
                destination.writestr(member, content)
            destination.writestr("handoff-manifest.json", canonical({k: hashlib.sha256(v).hexdigest() for k, v in data.items()}))
        read_completed(staging, report["input_identity"])
        os.link(staging, output)  # Atomic, no overwrite on macOS and Windows/NTFS.
    finally:
        staging.unlink(missing_ok=True)  # Only our uniquely-created temporary archive.


def run(project: Path, archive: Path, pool_audit: Path, preflight: Path,
        evaluation: Path, config_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    budget = patch_budget(config)
    md5, archive_sha256 = archive_hashes(archive)
    if (md5, archive_sha256) != (SOURCE_MD5, EXPECTED_ARCHIVE_SHA256):
        raise ValueError("Official MPI source archive checksum differs")
    if sha256_file(pool_audit) != EXPECTED_POOL_REPORT_SHA256:
        raise ValueError("Reviewed normal pool audit checksum differs")
    pool = json.loads(pool_audit.read_text(encoding="utf-8"))
    calibration = validate_v4_report(preflight)
    test_rgb = read_test_fingerprints(evaluation)
    implementation = implementation_identity(project)
    identity = {
        "archive_sha256": archive_sha256, "pool_audit_sha256": EXPECTED_POOL_REPORT_SHA256,
        "calibration_v4_sha256": EXPECTED_V4_SHA256, "external_evaluation_v2_sha256": EVALUATION_SHA256,
        "config_sha256": digest(config), "source_files_sha256": implementation["source_files_sha256"],
        "pillow_version": Image.__version__,
    }
    if output.exists():
        return read_completed(output, identity)
    with zipfile.ZipFile(archive) as source:
        if audit_members(source.infolist())["errors"]:
            raise ValueError("Archive metadata audit failed")
        normal = normal_members(source.infolist())
        if sorted(normal) != REGIONS:
            raise ValueError("Region set/order differs")
        all_ids = set(normal[REGIONS[0]])
        _, invalid_ids = validate_pool_report(pool, all_ids, pool["registration_json_sha256"])
        rows = scan_normals(source, normal, sorted(all_ids - invalid_ids))
    report = build_split(rows, calibration, test_rgb, config)
    report.update({
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "input_identity": identity, "implementation": implementation, "planned_patch_budget": budget,
        "invalid_original_photo_ids_excluded": sorted(invalid_ids),
        "normal_tiles_decoded_this_run": len(rows),
        "test_images_decoded_this_run": 0, "test_masks_decoded_this_run": 0,
        "test_previously_observed": True, "training_performed": False, "model_scores_computed": 0,
        "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
        "note": "Preparation only, not a trained model or evaluation. Do not reuse transferred-model thresholds for the future target-specific bank.",
    })
    write_handoff(output, report, rows, config)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pool-audit", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    output = args.output or project / "artifacts/handoff/mpi-pcb-normal-bank-preparation-v1.zip"
    try:
        report = run(project, args.archive,
                     args.pool_audit or project / "artifacts/data-audit/mpi-pcb-normal-pool-integrity-v1.json",
                     args.preflight or project / "artifacts/data-audit/mpi-pcb-normal-preflight-v4.json",
                     args.evaluation or project / "artifacts/handoff/mpi-pcb-external-evaluation-v2.zip",
                     args.config or project / "configs/mpi-pcb-normal-bank-pilot-v1.toml", output)
        print(json.dumps({
            "status": report["status"], "training_original_photos": len(report["training_photo_ids"]),
            "training_tiles": len(report["training_rows"]),
            "calibration_original_photos": len(report["calibration_photo_ids"]),
            "eligible_original_photos": report["eligible_original_photos"],
            "cross_split_or_full_duplicate_exclusions": len(report["excluded_fit_candidates"]),
            "partial_tile_duplicate_groups_in_fit": len(report["partial_tile_duplicate_groups_in_fit"]),
            "planned_patch_budget": report["planned_patch_budget"], "training_performed": False,
            "output": str(output), "sha256": sha256_file(output),
        }, indent=2, ensure_ascii=False))
        return 0 if report["training_photo_ids"] else 2
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
