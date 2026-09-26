"""One-shot MPI-PCB comparison: replay normals, audit all tests, then score."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectops.padim_experiment import git_state
from defectops.realiad_followup_development import _seed_everything, candidate_image_score
from defectops.realiad_followup_final import _gpu_modules, _model, _predict
from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes
from scripts.calibrate_mpi_pcb_normals import (
    CANDIDATE, MODEL_ORDER, REGIONS, model_contexts, score_normal_cohort, validate_model_preflight,
)
from scripts.mpi_pcb_external_protocol import (
    audit_test, compare_replay, decode_image, load_evaluation_config,
    model_metrics, paired_comparison, test_inventory, validate_calibration,
)
from scripts.preflight_mpi_pcb_models import (
    EXPECTED_ARCHIVE_SHA256, load_protocol, photo_score, sha256_file, validate_v4_report,
)


NAME = "mpi-pcb-external-evaluation-v1"
COMPLETE = "mpi_pcb_external_evaluated_after_normal_adaptation"


def write_json(path: Path, report: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def package_reports(output: Path, handoff: Path) -> None:
    """Only explicit JSON reports; never ship images, masks, maps or checkpoints."""
    names = ("input-lock.json", "calibration-replay.json", "test-opening.json", "test-audit.json",
             "realiad_pcb-scores.json", "visa_pcb3_v2-scores.json", "evaluation.json", "failure.json")
    files = [output / name for name in names if (output / name).is_file()]
    manifest = {"schema_version": 1, "content": "JSON reports only; no images/masks/maps/checkpoints",
                "members": [{"name": path.name, "sha256": sha256_file(path)} for path in files]}
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff, "x", compression=zipfile.ZIP_DEFLATED) as writer:
        for path in files:
            writer.write(path, path.name)
        writer.writestr("handoff-manifest.json", json.dumps(manifest, indent=2, allow_nan=False))


def score_test_photos(source, photos, model, recipe, gpu, threshold, map_dir, *, predict=_predict):
    """Use the audited bytes, preserve raw maps locally, and aggregate per photo."""
    map_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, photo in enumerate(photos, 1):
        if [row["region"] for row in photo["regions"]] != REGIONS:
            raise ValueError("Audited test region order changed")
        region_scores, maps = [], []
        for row in photo["regions"]:
            raw = source.read(row["member"])
            if hashlib.sha256(raw).hexdigest() != row["member_sha256"]:
                raise ValueError(f'Test bytes changed after audit: {row["member"]}')
            with decode_image(raw) as image:
                if hashlib.sha256(image.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
                    raise ValueError("Test decoded pixels changed after audit")
                raw_score, anomaly_map = predict(model, image, recipe, gpu)
            anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
            if anomaly_map.shape != (256, 256) or not np.isfinite(anomaly_map).all():
                raise ValueError("Expected a finite native 256x256 anomaly map")
            score = candidate_image_score(raw_score, anomaly_map, CANDIDATE)
            region_scores.append({"region": row["region"], "member": row["member"], "score": score})
            maps.append(anomaly_map)
        score = photo_score([row["score"] for row in region_scores])
        map_path = map_dir / f"photo-{index:03d}.npz"
        with map_path.open("xb") as stream:
            np.savez_compressed(stream, anomaly_maps=np.stack(maps), regions=np.array(REGIONS))
        results.append({"key": photo["key"], "photo_id": photo["photo_id"], "label": photo["label"],
                        "score": score, "threshold": threshold,
                        "decision": "REVIEW_REQUIRED" if score >= threshold else "BELOW_THRESHOLD_NOT_AUTO_PASS",
                        "regions": region_scores, "automatic_pass_allowed": False,
                        "map_file": map_path.relative_to(map_dir.parent.parent).as_posix(),
                        "map_sha256": sha256_file(map_path)})
        if index % 10 == 0 or index == len(photos):
            print(f"test scoring {index}/{len(photos)} original photos", flush=True)
    return results


def execute(source, output, config, v4, calibration, gate, recipes, gpu):
    """Ordering is deliberate: neither audit nor test scoring precedes replay."""
    inventory = test_inventory(source.infolist())  # Names only, no test pixels.
    torch = gpu[0]
    replay = {"models": {}, "thresholds_unchanged": config["thresholds"],
              "test_images_decoded": 0, "test_masks_decoded": 0, "test_scores_computed": 0}
    print("1/3 Replaying all 169 normal photos per model; test remains sealed...", flush=True)
    for key in MODEL_ORDER:
        _seed_everything(42, torch)
        recipe = {"base_model": recipes[key]["base_model"]}
        model = _model(recipe, Path(gate["models"][key]["checkpoint_path"]), recipes[key]["checkpoint_context"], gpu)
        try:
            current = score_normal_cohort(source, v4["calibration_rows"], v4["calibration_photo_ids"], model, recipe, gpu)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        comparison = compare_replay(calibration["models"][key]["photo_scores"], current)
        comparison["passed"] = max(comparison["maximum_absolute_region_score_difference"],
                                   comparison["maximum_absolute_photo_score_difference"]) <= config["calibration_replay_absolute_tolerance"]
        replay["models"][key] = {**comparison, "photo_scores": current}
    replay["passed"] = all(row["passed"] for row in replay["models"].values())
    write_json(output / "calibration-replay.json", replay)
    if not replay["passed"]:
        return {"status": "normal_replay_blocked_test_sealed", "test_images_decoded": 0,
                "test_masks_decoded": 0, "test_scores_computed": 0, "automatic_pass_allowed": False}

    # Create marker BEFORE first attempt to decode test bytes. Retained on crash.
    write_json(output / "test-opening.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                                              "stage": "starting_test_integrity_audit"})
    print("2/3 Auditing all 1320 test images and 660 masks; no test scores yet...", flush=True)
    audit = audit_test(source, inventory, v4["calibration_rows"])
    write_json(output / "test-audit.json", audit)
    if audit["issues"]:
        return {"status": "test_audit_blocked_no_scores", "issue_count": len(audit["issues"]),
                "test_images_decoded": audit["test_images_decoded"], "test_masks_decoded": audit["test_masks_decoded"],
                "test_scores_computed": 0, "automatic_pass_allowed": False}
    if not (len(audit["photos"]) == 110 and audit["test_images_decoded"] == 1320 and audit["test_masks_decoded"] == 660
            and sum(p["label"] == 0 for p in audit["photos"]) == 55
            and sum(p["label"] == 1 for p in audit["photos"]) == 55):
        raise ValueError("Audit denominator differs from 55 normal / 55 altered photos")

    print("3/3 Comparing two frozen models; thresholds will NOT change...", flush=True)
    results = {}
    for key in MODEL_ORDER:
        print(f"Frozen model: {key}", flush=True)
        _seed_everything(42, torch)
        recipe = {"base_model": recipes[key]["base_model"]}
        model = _model(recipe, Path(gate["models"][key]["checkpoint_path"]), recipes[key]["checkpoint_context"], gpu)
        try:
            photos = score_test_photos(source, audit["photos"], model, recipe, gpu,
                                       config["thresholds"][key], output / "maps" / key)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if len(photos) != 110 or any(len(p["regions"]) != 12 for p in photos):
            raise ValueError("Test scoring did not complete all 110 photos and 1320 regions")
        results[key] = {"checkpoint_sha256": gate["models"][key]["checkpoint_sha256"],
                        "threshold": config["thresholds"][key], "photos": photos,
                        "metrics": model_metrics(photos, config["thresholds"][key])}
        write_json(output / f"{key}-scores.json", results[key])
    return {"status": COMPLETE, "original_photos_per_model": 110, "normal_original_photos": 55,
            "manually_altered_original_photos": 55, "test_images_decoded_at_audit": 1320,
            "test_masks_decoded_at_audit": 660, "test_region_scores_per_model": 1320,
            "test_region_scores_total": 2640, "test_photo_scores_total": 220,
            "models": {key: {field: data[field] for field in ("checkpoint_sha256", "threshold", "metrics")}
                       for key, data in results.items()}, "paired_comparison": paired_comparison(results, config),
            "pixel_metrics_computed": False, "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
            "limitation": "One physical board, repeated views and manually added alterations; not production defect validation."}


def run(project: Path, archive_path: Path) -> dict:
    started = time.perf_counter()
    output = project / "artifacts" / "evaluations" / NAME
    handoff = project / "artifacts" / "handoff" / f"{NAME}.zip"
    for path in (output, handoff):
        if path.exists():
            raise FileExistsError(f"Existing output is preserved; do not delete or retry blindly: {path}")
    git = git_state(project)
    protocol = load_protocol(project / "configs" / "mpi-pcb-model-protocol-v1.toml")
    config = load_evaluation_config(project / "configs" / f"{NAME}.toml")
    v4 = validate_v4_report(project / "artifacts" / "data-audit" / "mpi-pcb-normal-preflight-v4.json")
    gate = validate_model_preflight(project / "artifacts" / "data-audit" / "mpi-pcb-model-preflight-v1.json", project, protocol)
    calibration = validate_calibration(project / "artifacts" / "calibrations" / "mpi-pcb-normal-calibration-v1.json", v4, protocol, config)
    md5, digest = archive_hashes(archive_path)
    if md5 != SOURCE_MD5 or digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MPI-PCB source archive checksum changed")
    recipes = model_contexts(project, protocol)
    gpu = _gpu_modules()
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input-lock.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                                           "git": git, "evaluation_config": config, "normal_protocol": protocol,
                                           "models": gate["models"], "archive_sha256": digest,
                                           "python": sys.version,
                                           "versions": {name: importlib.metadata.version(name) for name in
                                                        ("torch", "torchvision", "anomalib", "numpy", "Pillow")},
                                           "gpu": gpu[0].cuda.get_device_name(0)})
    try:
        with zipfile.ZipFile(archive_path) as source:
            report = execute(source, output, config, v4, calibration, gate, recipes, gpu)
        report.update({"git": git, "evaluation_config_sha256": config["config_sha256"],
                       "calibration_report_sha256": config["calibration_report_sha256"],
                       "archive_sha256": digest, "elapsed_seconds": time.perf_counter() - started})
        write_json(output / "evaluation.json", report)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "incomplete_no_valid_comparison", "git": git,
                                             "error": f"{type(exc).__name__}: {exc}",
                                             "test_opening_started": (output / "test-opening.json").exists(),
                                             "test_scores_computed": None,
                                             "note": "Unknown partial progress; no success metrics claimed. Do not exclude samples or rerun without review.",
                                             "automatic_pass_allowed": False})
        package_reports(output, handoff)
        raise
    package_reports(output, handoff)
    return {**report, "output": str(output), "handoff": str(handoff), "handoff_sha256": sha256_file(handoff)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
