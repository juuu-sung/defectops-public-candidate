"""Verify the integrity-filtered MPI-PCB normal cohort while test stays sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes, audit_members
from scripts.audit_mpi_pcb_normal_pool import _ids_sha256, propose_calibration_ids
from scripts.audit_mpi_pcb_selected_normals import inspect_png
from scripts.preflight_mpi_pcb_normals import (
    CALIBRATION_PHOTOS,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_NORMAL_TRAIN_COUNT,
    EXPECTED_REGION_COUNT,
    decode_normal,
    normal_members,
    select_calibration_ids,
    summarize_decoded_duplicates,
)


EXPECTED_POOL_INVALID_TILES = 16
EXPECTED_POOL_COMPLETE_PHOTOS = 1671
EXPECTED_INVALID_PHOTO_IDS = {
    "0158", "0159", "0160", "0643", "0695", "0696", "0815", "0889",
    "1040", "1225", "1303", "1404", "1674", "1675", "1678", "1679",
}
EXPECTED_CANDIDATE_SHA256 = "af412abd25eac87c80ed97d83c7d9548ceb0fae1c7d513588377cf1590d5928c"


def validate_pool_report(
    pool: dict[str, object], all_ids: set[str], registration_sha256: str,
) -> tuple[list[str], set[str]]:
    """Recompute every decision-relevant cohort field from the train/good inventory."""
    if len(all_ids) != EXPECTED_NORMAL_TRAIN_COUNT:
        raise ValueError("Training normal photo inventory changed")
    if not isinstance(pool.get("invalid_photo_ids"), list) or not isinstance(pool.get("issues"), list):
        raise ValueError("Pool report is missing issue lists")
    invalid_ids = set(pool["invalid_photo_ids"])
    issues = pool["issues"]
    candidate = propose_calibration_ids(all_ids, invalid_ids, count=CALIBRATION_PHOTOS)
    original = select_calibration_ids(all_ids)
    complete_ids = sorted(all_ids - invalid_ids)
    issue_regions = Counter(issue.get("region") for issue in issues)
    issue_ids = [issue.get("photo_id") for issue in issues]
    issue_paths_valid = all(
        isinstance(issue.get("region"), str)
        and isinstance(issue.get("photo_id"), str)
        and issue.get("member") == (
            f'extracted/{issue["region"]}/train/good/{issue["photo_id"]}.png'
        )
        and ("zip_read_error" in issue or "verify_error" in issue or "load_error" in issue)
        for issue in issues
    )
    if not (
        pool.get("status") == "normal_pool_integrity_audited_test_sealed"
        and pool.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and pool.get("registration_json_sha256") == registration_sha256
        and pool.get("training_photo_ids") == EXPECTED_NORMAL_TRAIN_COUNT
        and pool.get("regions") == EXPECTED_REGION_COUNT
        and pool.get("attempted_normal_tiles") == EXPECTED_NORMAL_TRAIN_COUNT * EXPECTED_REGION_COUNT
        and pool.get("invalid_normal_tiles") == EXPECTED_POOL_INVALID_TILES
        and pool.get("valid_normal_tiles") == (
            EXPECTED_NORMAL_TRAIN_COUNT * EXPECTED_REGION_COUNT - EXPECTED_POOL_INVALID_TILES
        )
        and pool.get("valid_dimensions") == {"256x256": pool.get("valid_normal_tiles")}
        and len(issues) == EXPECTED_POOL_INVALID_TILES
        and issue_paths_valid
        and len(set(issue_ids)) == len(issue_ids) == EXPECTED_POOL_INVALID_TILES
        and set(issue_ids) == invalid_ids == EXPECTED_INVALID_PHOTO_IDS
        and pool.get("issues_by_region") == dict(sorted(issue_regions.items()))
        and pool.get("complete_photo_ids") == complete_ids
        and pool.get("complete_photo_ids_sha256") == _ids_sha256(complete_ids)
        and pool.get("complete_photo_count") == EXPECTED_POOL_COMPLETE_PHOTOS
        and pool.get("original_calibration_photo_ids_sha256") == _ids_sha256(original)
        and pool.get("candidate_protocol_version") == 3
        and pool.get("candidate_selection_status") == "candidate_ready_for_review"
        and pool.get("candidate_calibration_photo_ids") == candidate
        and pool.get("candidate_calibration_photo_ids_sha256") == _ids_sha256(candidate)
        and pool.get("candidate_calibration_photo_ids_sha256") == EXPECTED_CANDIDATE_SHA256
        and pool.get("candidate_calibration_photo_count") == CALIBRATION_PHOTOS
        and pool.get("pillow_version") == Image.__version__
        and pool.get("test_images_decoded") == 0
        and pool.get("test_masks_decoded") == 0
        and pool.get("model_scores_computed") == 0
        and pool.get("automatic_pass_allowed") is False
    ):
        raise ValueError("Pool report differs from the reviewed, test-sealed v3 candidate")
    return candidate, invalid_ids


def scan_candidate(
    source: zipfile.ZipFile,
    normal: dict[str, dict[str, zipfile.ZipInfo]],
    candidate: list[str],
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows: list[dict[str, object]] = []
    dimensions: Counter[str] = Counter()
    for region in sorted(normal):
        for index, photo_id in enumerate(candidate, start=1):
            info = normal[region][photo_id]
            if not info.filename.startswith(f"extracted/{region}/train/good/"):
                raise ValueError(f"Non-training candidate member: {info.filename}")
            raw = source.read(info)  # Validates the outer ZIP CRC.
            details = inspect_png(raw)
            if "verify_error" in details or "load_error" in details:
                raise ValueError(f"Candidate normal failed integrity check: {info.filename}: {details}")
            size, rgb_sha256 = decode_normal(raw, info.filename)
            dimensions[f"{size[0]}x{size[1]}"] += 1
            rows.append({
                "region": region,
                "photo_id": photo_id,
                "member": info.filename,
                "width": size[0],
                "height": size[1],
                "member_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_rgb_sha256": rgb_sha256,
            })
            if index % 50 == 0 or index == len(candidate):
                print(f"{region} v3 normal preflight {index}/{len(candidate)}", flush=True)
    return rows, dimensions


def run(archive: Path, registration: Path, pool_audit: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    if not archive.is_file() or not registration.is_file() or not pool_audit.is_file():
        raise FileNotFoundError("Archive, registration, or pool-audit JSON is missing")
    registration_bytes = registration.read_bytes()
    record = json.loads(registration_bytes)
    if not (
        record.get("status") == "registered_metadata_only"
        and record.get("region_count") == EXPECTED_REGION_COUNT
        and record.get("archive_md5") == SOURCE_MD5
        and record.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and record.get("archive_size_bytes") == archive.stat().st_size
        and record.get("test_images_decoded") == 0
        and record.get("test_masks_decoded") == 0
        and record.get("model_scores_computed") == 0
        and record.get("errors") == []
    ):
        raise ValueError("Registration report differs from the locked metadata-only record")
    md5, sha256 = archive_hashes(archive)
    if md5 != SOURCE_MD5 or sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the locked official download")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    pool_bytes = pool_audit.read_bytes()
    pool = json.loads(pool_bytes)
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        metadata = audit_members(infos)
        if metadata["status"] != "registered_metadata_only":
            raise ValueError(f"Archive metadata audit failed: {metadata['errors']}")
        normal = normal_members(infos)
        candidate, invalid_ids = validate_pool_report(
            pool, set(normal[sorted(normal)[0]]), hashlib.sha256(registration_bytes).hexdigest(),
        )
        rows, dimensions = scan_candidate(source, normal, candidate)
    if len(rows) != CALIBRATION_PHOTOS * EXPECTED_REGION_COUNT:
        raise ValueError("Incomplete v3 normal calibration cohort")
    tile_duplicates, composite_duplicates = summarize_decoded_duplicates(
        rows, candidate, sorted(normal),
    )
    old_ids = set(select_calibration_ids(set(normal[sorted(normal)[0]])))
    report: dict[str, object] = {
        "status": (
            "normal_preflight_v3_needs_review_test_sealed"
            if composite_duplicates else "normal_preflight_v3_passed_test_sealed"
        ),
        "protocol_version": 3,
        "archive_sha256": sha256,
        "registration_json_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "pool_audit_json_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "excluded_incomplete_photo_ids": sorted(invalid_ids),
        "original_cohort_removed_photo_ids": sorted(old_ids - set(candidate)),
        "replacement_photo_ids": sorted(set(candidate) - old_ids),
        "selection_rule": pool["candidate_selection_rule"],
        "calibration_photo_ids": candidate,
        "calibration_photo_ids_sha256": _ids_sha256(candidate),
        "calibration_original_photos": len(candidate),
        "calibration_region_images_decoded": len(rows),
        "regions": len(normal),
        "dimensions": dict(sorted(dimensions.items())),
        "calibration_rows": rows,
        "tile_duplicate_summary": tile_duplicates,
        "identical_12_region_photo_id_groups": composite_duplicates,
        "pillow_version": Image.__version__,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
        "note": (
            "This preflight verifies normal-data integrity and exact 12-region duplicates. "
            "It does not train or evaluate a model and does not permit opening test data "
            "until the checkpoint and evaluation protocol are separately locked."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--pool-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.archive, args.registration, args.pool_audit, args.output)
    print(json.dumps({
        "status": report["status"],
        "calibration_original_photos": report["calibration_original_photos"],
        "calibration_region_images_decoded": report["calibration_region_images_decoded"],
        "calibration_photo_ids_sha256": report["calibration_photo_ids_sha256"],
        "original_cohort_removed_photo_ids": report["original_cohort_removed_photo_ids"],
        "replacement_photo_ids": report["replacement_photo_ids"],
        "identical_12_region_photo_id_groups": report["identical_12_region_photo_id_groups"],
        "test_images_decoded": report["test_images_decoded"],
        "model_scores_computed": report["model_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "normal_preflight_v3_needs_review_test_sealed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
