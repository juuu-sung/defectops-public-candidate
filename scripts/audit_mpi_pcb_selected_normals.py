"""Read-only integrity audit of the locked MPI-PCB normal calibration selection."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes, audit_members
from scripts.preflight_mpi_pcb_normals import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_REGION_COUNT,
    normal_members,
    select_calibration_ids,
)


def inspect_png(raw: bytes) -> dict[str, object]:
    """Require both the PNG chunk CRC check and a full pixel decode."""
    result: dict[str, object] = {
        "member_bytes": len(raw),
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "signature_hex": raw[:8].hex(),
    }
    try:
        with Image.open(io.BytesIO(raw)) as image:
            result.update({
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            })
            if image.format != "PNG":
                raise ValueError(f"Expected PNG encoding, got {image.format}")
            image.verify()
    except Exception as exc:
        result["verify_error"] = f"{type(exc).__name__}: {exc}"
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG encoding, got {image.format}")
            image.load()
    except Exception as exc:
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def scan_selected(
    source: zipfile.ZipFile,
    normal: dict[str, dict[str, zipfile.ZipInfo]],
    selected: list[str],
) -> tuple[int, Counter[str], list[dict[str, object]]]:
    attempted = 0
    dimensions: Counter[str] = Counter()
    issues: list[dict[str, object]] = []
    for region in sorted(normal):
        for index, photo_id in enumerate(selected, start=1):
            info = normal[region][photo_id]
            if not info.filename.startswith(f"extracted/{region}/train/good/"):
                raise ValueError(f"Non-training member in normal selection: {info.filename}")
            attempted += 1
            row: dict[str, object] = {
                "region": region, "photo_id": photo_id, "member": info.filename,
            }
            try:
                raw = source.read(info)  # zipfile validates this member's ZIP CRC.
            except Exception as exc:
                row["zip_read_error"] = f"{type(exc).__name__}: {exc}"
                issues.append(row)
            else:
                details = inspect_png(raw)
                if "verify_error" in details or "load_error" in details:
                    row.update(details)
                    issues.append(row)
                else:
                    dimensions[f'{details["width"]}x{details["height"]}'] += 1
            if index % 50 == 0 or index == len(selected):
                print(f"{region} normal decode audit {index}/{len(selected)}", flush=True)
    return attempted, dimensions, issues


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
        selected = select_calibration_ids(set(normal[sorted(normal)[0]]))
        attempted, dimensions, issues = scan_selected(source, normal, selected)
    if attempted != len(selected) * EXPECTED_REGION_COUNT:
        raise ValueError("Normal decode audit did not attempt the locked selection")
    selected_digest = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report: dict[str, object] = {
        "status": (
            "normal_decode_audit_completed_with_issues_test_sealed"
            if issues else "normal_decode_audit_completed_no_issues_test_sealed"
        ),
        "archive_sha256": sha256,
        "registration_json_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "selection_rule": "SHA256(mpi-pcb-calibration-v1\\0photo_id), ascending",
        "selected_photo_ids": selected,
        "selected_photo_ids_sha256": selected_digest,
        "selected_original_photos": len(selected),
        "regions": EXPECTED_REGION_COUNT,
        "attempted_normal_tiles": attempted,
        "valid_normal_tiles": attempted - len(issues),
        "invalid_normal_tiles": len(issues),
        "valid_dimensions": dict(sorted(dimensions.items())),
        "issues": issues,
        "pillow_version": Image.__version__,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
        "note": (
            "Diagnostic only. No image is repaired, excluded, or replaced. "
            "This report is not a passing preflight or permission to open test data."
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
        "selected_original_photos": report["selected_original_photos"],
        "attempted_normal_tiles": report["attempted_normal_tiles"],
        "invalid_normal_tiles": report["invalid_normal_tiles"],
        "issues": report["issues"],
        "test_images_decoded": report["test_images_decoded"],
        "model_scores_computed": report["model_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
