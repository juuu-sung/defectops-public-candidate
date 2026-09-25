"""Verify a deduplicated MPI-PCB normal cohort without opening test data."""

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
from scripts.audit_mpi_pcb_normal_pool import _ids_sha256
from scripts.audit_mpi_pcb_selected_normals import inspect_png
from scripts.preflight_mpi_pcb_normals import (
    CALIBRATION_PHOTOS,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_REGION_COUNT,
    SELECTION_SALT,
    decode_normal,
    normal_members,
    summarize_decoded_duplicates,
)
from scripts.preflight_mpi_pcb_normals_v3 import (
    EXPECTED_CANDIDATE_SHA256,
    validate_pool_report,
)


EXPECTED_V3_REPORT_SHA256 = "da5794ff706be9e91b87903626b0c82fedf63b5a41c7f229f9b9149dcaa98e3e"
EXPECTED_POOL_REPORT_SHA256 = "33b58281d5840550712f253a793913754cdfd8a9a8bcecc91cd18d9720d41703"
EXPECTED_V3_DUPLICATE_GROUP = [["0879", "0722"]]
EXPECTED_DEDUPLICATED_ID = "0722"
EXPECTED_REPLACEMENT_ID = "1481"
EXPECTED_V4_CANDIDATE_SHA256 = "75768131852bfe1234f754c3b77054cce64e2a7c890a94db9724943b21faf11e"


def validate_v3_report(
    report: dict[str, object], candidate: list[str], regions: list[str],
    registration_sha256: str, pool_sha256: str,
) -> None:
    """Verify the reviewed v3 duplicate from its complete normal-only rows."""
    rows = report.get("calibration_rows")
    if not isinstance(rows, list):
        raise ValueError("v3 preflight has no calibration rows")
    expected_pairs = {(region, photo_id) for region in regions for photo_id in candidate}
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("v3 preflight has a malformed calibration row")
    pairs = [(row.get("region"), row.get("photo_id")) for row in rows]
    if len(pairs) != len(expected_pairs) or len(set(pairs)) != len(pairs) or set(pairs) != expected_pairs:
        raise ValueError("v3 preflight rows are incomplete or repeated")
    if not all(
        row.get("member") == f'extracted/{row["region"]}/train/good/{row["photo_id"]}.png'
        and row.get("width") == row.get("height") == 256
        and all(
            isinstance(row.get(field), str)
            and len(row[field]) == 64
            and all(character in "0123456789abcdef" for character in row[field])
            for field in ("member_sha256", "decoded_rgb_sha256")
        )
        for row in rows
    ):
        raise ValueError("v3 preflight has an unexpected normal tile")
    tile_summary, composite_groups = summarize_decoded_duplicates(rows, candidate, regions)
    duplicate_rows = {
        photo_id: {row["region"]: row for row in rows if row["photo_id"] == photo_id}
        for photo_id in EXPECTED_V3_DUPLICATE_GROUP[0]
    }
    raw_matches = all(
        duplicate_rows["0879"][region]["member_sha256"]
        == duplicate_rows["0722"][region]["member_sha256"]
        for region in regions
    )
    if not (
        report.get("status") == "normal_preflight_v3_needs_review_test_sealed"
        and report.get("protocol_version") == 3
        and report.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and report.get("registration_json_sha256") == registration_sha256
        and report.get("pool_audit_json_sha256") == pool_sha256
        and report.get("calibration_photo_ids") == candidate
        and report.get("calibration_photo_ids_sha256") == _ids_sha256(candidate)
        and report.get("calibration_photo_ids_sha256") == EXPECTED_CANDIDATE_SHA256
        and report.get("calibration_original_photos") == CALIBRATION_PHOTOS
        and report.get("calibration_region_images_decoded") == CALIBRATION_PHOTOS * EXPECTED_REGION_COUNT
        and report.get("regions") == EXPECTED_REGION_COUNT
        and report.get("dimensions") == {"256x256": len(rows)}
        and report.get("tile_duplicate_summary") == tile_summary
        and report.get("identical_12_region_photo_id_groups") == composite_groups
        and composite_groups == EXPECTED_V3_DUPLICATE_GROUP
        and raw_matches
        and report.get("pillow_version") == Image.__version__
        and report.get("test_images_decoded") == 0
        and report.get("test_masks_decoded") == 0
        and report.get("model_scores_computed") == 0
        and report.get("automatic_pass_allowed") is False
    ):
        raise ValueError("v3 preflight differs from the reviewed, test-sealed duplicate report")


def select_v4_ids(
    all_ids: set[str], invalid_ids: set[str], v3_candidate: list[str],
) -> list[str]:
    """Keep the earlier hash-ranked duplicate and fill from the next complete ID."""
    ranked_complete = sorted(
        all_ids - invalid_ids,
        key=lambda photo_id: (
            hashlib.sha256(f"{SELECTION_SALT}\0{photo_id}".encode()).digest(), photo_id,
        ),
    )
    if ranked_complete[:CALIBRATION_PHOTOS] != v3_candidate:
        raise ValueError("v3 cohort is not the original hash-ranked complete cohort")
    if not (
        v3_candidate.index("0879") < v3_candidate.index(EXPECTED_DEDUPLICATED_ID)
        and ranked_complete[CALIBRATION_PHOTOS] == EXPECTED_REPLACEMENT_ID
    ):
        raise ValueError("Reviewed duplicate ranking or next complete ID changed")
    selected = [photo_id for photo_id in v3_candidate if photo_id != EXPECTED_DEDUPLICATED_ID]
    selected.append(EXPECTED_REPLACEMENT_ID)
    if not (
        len(selected) == len(set(selected)) == CALIBRATION_PHOTOS
        and _ids_sha256(selected) == EXPECTED_V4_CANDIDATE_SHA256
    ):
        raise ValueError("v4 calibration cohort differs from the locked proposal")
    return selected


def scan_v4_candidate(
    source: zipfile.ZipFile,
    normal: dict[str, dict[str, zipfile.ZipInfo]],
    candidate: list[str],
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows: list[dict[str, object]] = []
    dimensions: Counter[str] = Counter()
    for region in sorted(normal):
        for index, photo_id in enumerate(candidate, start=1):
            info = normal[region][photo_id]
            if info.filename != f"extracted/{region}/train/good/{photo_id}.png":
                raise ValueError(f"Non-training candidate member: {info.filename}")
            raw = source.read(info)  # Checks the outer ZIP CRC.
            details = inspect_png(raw)
            if "verify_error" in details or "load_error" in details:
                raise ValueError(f"v4 candidate failed PNG integrity: {info.filename}: {details}")
            size, rgb_sha256 = decode_normal(raw, info.filename)
            if size != (256, 256):
                raise ValueError(f"Unexpected v4 candidate dimensions: {info.filename}: {size}")
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
                print(f"{region} v4 normal preflight {index}/{len(candidate)}", flush=True)
    return rows, dimensions


def run(
    archive: Path, registration: Path, pool_audit: Path, v3_preflight: Path, output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    if not all(path.is_file() for path in (archive, registration, pool_audit, v3_preflight)):
        raise FileNotFoundError("Archive, registration, pool audit, or v3 preflight JSON is missing")
    registration_bytes = registration.read_bytes()
    pool_bytes = pool_audit.read_bytes()
    v3_bytes = v3_preflight.read_bytes()
    registration_sha256 = hashlib.sha256(registration_bytes).hexdigest()
    pool_sha256 = hashlib.sha256(pool_bytes).hexdigest()
    v3_sha256 = hashlib.sha256(v3_bytes).hexdigest()
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
        and pool_sha256 == EXPECTED_POOL_REPORT_SHA256
        and v3_sha256 == EXPECTED_V3_REPORT_SHA256
    ):
        raise ValueError("Input record or v3/pool report checksum differs from the reviewed evidence")
    md5, archive_sha256 = archive_hashes(archive)
    if md5 != SOURCE_MD5 or archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the locked official download")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    pool = json.loads(pool_bytes)
    v3 = json.loads(v3_bytes)
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        metadata = audit_members(infos)
        if metadata["status"] != "registered_metadata_only":
            raise ValueError(f"Archive metadata audit failed: {metadata['errors']}")
        normal = normal_members(infos)
        regions = sorted(normal)
        all_ids = set(normal[regions[0]])
        v3_candidate, invalid_ids = validate_pool_report(pool, all_ids, registration_sha256)
        validate_v3_report(v3, v3_candidate, regions, registration_sha256, pool_sha256)
        candidate = select_v4_ids(all_ids, invalid_ids, v3_candidate)
        rows, dimensions = scan_v4_candidate(source, normal, candidate)
    if len(rows) != CALIBRATION_PHOTOS * EXPECTED_REGION_COUNT:
        raise ValueError("Incomplete v4 normal calibration cohort")
    tile_duplicates, composite_duplicates = summarize_decoded_duplicates(rows, candidate, regions)
    report: dict[str, object] = {
        "status": (
            "normal_preflight_v4_needs_review_test_sealed"
            if composite_duplicates else "normal_preflight_v4_passed_test_sealed"
        ),
        "protocol_version": 4,
        "archive_sha256": archive_sha256,
        "registration_json_sha256": registration_sha256,
        "pool_audit_json_sha256": pool_sha256,
        "v3_preflight_json_sha256": v3_sha256,
        "excluded_incomplete_photo_ids": sorted(invalid_ids),
        "removed_duplicate_photo_id": EXPECTED_DEDUPLICATED_ID,
        "retained_duplicate_photo_id": "0879",
        "replacement_photo_id": EXPECTED_REPLACEMENT_ID,
        "selection_rule": (
            "Start from the locked v3 169 complete-normal IDs in original hash order; "
            "for the sole 12-region-identical pair (0879, 0722), retain the earlier "
            "hash-ranked 0879, remove 0722, and append the next complete-normal ID 1481."
        ),
        "calibration_photo_ids": candidate,
        "calibration_photo_ids_sha256": _ids_sha256(candidate),
        "calibration_original_photos": len(candidate),
        "calibration_region_images_decoded": len(rows),
        "regions": len(regions),
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
            "Normal-only deduplication preflight. A passed result verifies this 169-ID "
            "calibration cohort, not either model or the unopened test data. "
            "Checkpoint and evaluation rules must be locked separately before test."
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
    parser.add_argument("--v3-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.archive, args.registration, args.pool_audit, args.v3_preflight, args.output)
    print(json.dumps({
        "status": report["status"],
        "calibration_original_photos": report["calibration_original_photos"],
        "calibration_region_images_decoded": report["calibration_region_images_decoded"],
        "calibration_photo_ids_sha256": report["calibration_photo_ids_sha256"],
        "removed_duplicate_photo_id": report["removed_duplicate_photo_id"],
        "replacement_photo_id": report["replacement_photo_id"],
        "identical_12_region_photo_id_groups": report["identical_12_region_photo_id_groups"],
        "test_images_decoded": report["test_images_decoded"],
        "model_scores_computed": report["model_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "normal_preflight_v4_needs_review_test_sealed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
