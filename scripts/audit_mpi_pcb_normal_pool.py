"""Audit every MPI-PCB training normal and propose a test-sealed calibration cohort."""

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
from scripts.audit_mpi_pcb_selected_normals import inspect_png
from scripts.preflight_mpi_pcb_normals import (
    CALIBRATION_PHOTOS,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_NORMAL_TRAIN_COUNT,
    EXPECTED_REGION_COUNT,
    SELECTION_SALT,
    normal_members,
    select_calibration_ids,
)


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def propose_calibration_ids(
    all_ids: set[str], invalid_ids: set[str], count: int = CALIBRATION_PHOTOS,
) -> list[str]:
    """Apply the original hash order after excluding integrity-failed photo IDs."""
    if len(all_ids) != EXPECTED_NORMAL_TRAIN_COUNT or not invalid_ids <= all_ids:
        raise ValueError("Invalid complete-training-photo ID inventory")
    valid_ids = all_ids - invalid_ids
    ordered = sorted(valid_ids, key=lambda value: (
        hashlib.sha256(f"{SELECTION_SALT}\0{value}".encode()).digest(), value,
    ))
    return ordered[:count] if len(ordered) >= count else []


def scan_normal_pool(
    source: zipfile.ZipFile, normal: dict[str, dict[str, zipfile.ZipInfo]],
) -> tuple[int, Counter[str], Counter[str], list[dict[str, object]], set[str]]:
    photo_ids = sorted(normal[sorted(normal)[0]])
    attempted = 0
    dimensions: Counter[str] = Counter()
    issues_by_region: Counter[str] = Counter()
    issues: list[dict[str, object]] = []
    invalid_ids: set[str] = set()
    for region in sorted(normal):
        for index, photo_id in enumerate(photo_ids, start=1):
            info = normal[region][photo_id]
            if not info.filename.startswith(f"extracted/{region}/train/good/"):
                raise ValueError(f"Non-training member in normal pool: {info.filename}")
            attempted += 1
            issue: dict[str, object] = {
                "region": region, "photo_id": photo_id, "member": info.filename,
            }
            try:
                raw = source.read(info)  # Checks the outer ZIP CRC for this member.
            except Exception as exc:
                issue["zip_read_error"] = f"{type(exc).__name__}: {exc}"
            else:
                details = inspect_png(raw)
                if "verify_error" in details or "load_error" in details:
                    issue.update(details)
                else:
                    dimensions[f'{details["width"]}x{details["height"]}'] += 1
                    issue = {}
            if issue:
                issues.append(issue)
                issues_by_region[region] += 1
                invalid_ids.add(photo_id)
            if index % 100 == 0 or index == len(photo_ids):
                print(f"{region} train/good integrity {index}/{len(photo_ids)}", flush=True)
    return attempted, dimensions, issues_by_region, issues, invalid_ids


def run(archive: Path, registration: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    if not archive.is_file() or not registration.is_file():
        raise FileNotFoundError("Archive or registration JSON is missing")
    registration_bytes = registration.read_bytes()
    record = json.loads(registration_bytes)
    if (
        record.get("status") != "registered_metadata_only"
        or record.get("region_count") != EXPECTED_REGION_COUNT
        or record.get("archive_md5") != SOURCE_MD5
        or record.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256
        or record.get("archive_size_bytes") != archive.stat().st_size
        or record.get("test_images_decoded") != 0
        or record.get("test_masks_decoded") != 0
        or record.get("model_scores_computed") != 0
        or record.get("errors") != []
    ):
        raise ValueError("Registration report differs from the locked metadata-only record")
    md5, sha256 = archive_hashes(archive)
    if md5 != SOURCE_MD5 or sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the locked official download")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        metadata = audit_members(infos)
        if metadata["status"] != "registered_metadata_only":
            raise ValueError(f"Archive metadata audit failed: {metadata['errors']}")
        normal = normal_members(infos)
        all_ids = set(normal[sorted(normal)[0]])
        attempted, dimensions, issues_by_region, issues, invalid_ids = scan_normal_pool(
            source, normal,
        )
    if attempted != EXPECTED_NORMAL_TRAIN_COUNT * EXPECTED_REGION_COUNT:
        raise ValueError("Normal pool integrity audit did not attempt every training tile")
    complete_ids = sorted(all_ids - invalid_ids)
    proposed = propose_calibration_ids(all_ids, invalid_ids, count=CALIBRATION_PHOTOS)
    original = select_calibration_ids(all_ids)
    report: dict[str, object] = {
        "status": "normal_pool_integrity_audited_test_sealed",
        "archive_sha256": sha256,
        "registration_json_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "training_photo_ids": len(all_ids),
        "regions": EXPECTED_REGION_COUNT,
        "attempted_normal_tiles": attempted,
        "valid_normal_tiles": attempted - len(issues),
        "invalid_normal_tiles": len(issues),
        "valid_dimensions": dict(sorted(dimensions.items())),
        "issues_by_region": dict(sorted(issues_by_region.items())),
        "issues": issues,
        "invalid_photo_ids": sorted(invalid_ids),
        "complete_photo_ids": complete_ids,
        "complete_photo_ids_sha256": _ids_sha256(complete_ids),
        "complete_photo_count": len(complete_ids),
        "original_calibration_photo_ids_sha256": _ids_sha256(original),
        "candidate_protocol_version": 3,
        "candidate_selection_rule": (
            f"From all {EXPECTED_NORMAL_TRAIN_COUNT} training photo IDs, exclude any ID "
            "with an invalid tile in any of 12 regions; order the remaining IDs "
            f"by SHA256({SELECTION_SALT}\\0photo_id), ascending; take the first "
            f"{CALIBRATION_PHOTOS}."
        ),
        "candidate_selection_status": (
            "candidate_ready_for_review" if proposed else "insufficient_complete_normals"
        ),
        "candidate_calibration_photo_ids": proposed,
        "candidate_calibration_photo_ids_sha256": _ids_sha256(proposed) if proposed else None,
        "candidate_calibration_photo_count": len(proposed),
        "pillow_version": Image.__version__,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
        "note": (
            "Read-only integrity audit and candidate selection, not an approved preflight. "
            "No image was repaired, excluded from the archive, or replaced. "
            "The candidate requires report review and a separate versioned preflight "
            "before any test image or mask is opened."
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.archive, args.registration, args.output)
    print(json.dumps({
        "status": report["status"],
        "attempted_normal_tiles": report["attempted_normal_tiles"],
        "invalid_normal_tiles": report["invalid_normal_tiles"],
        "invalid_photo_ids": report["invalid_photo_ids"],
        "complete_photo_count": report["complete_photo_count"],
        "candidate_selection_status": report["candidate_selection_status"],
        "candidate_calibration_photo_count": report["candidate_calibration_photo_count"],
        "candidate_calibration_photo_ids_sha256": report[
            "candidate_calibration_photo_ids_sha256"
        ],
        "test_images_decoded": report["test_images_decoded"],
        "model_scores_computed": report["model_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
