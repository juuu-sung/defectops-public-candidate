"""Verify MPI-PCB calibration normals without opening any test image or mask."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes, audit_members


EXPECTED_ARCHIVE_SHA256 = "0fa9b20fae64142a418487a1d898ee2ee824eee0bd564896a245f4d991112607"
EXPECTED_REGION_COUNT = 12
EXPECTED_NORMAL_TRAIN_COUNT = 1687
CALIBRATION_PHOTOS = 169
SELECTION_SALT = "mpi-pcb-calibration-v1"


def select_calibration_ids(ids: set[str], count: int = CALIBRATION_PHOTOS) -> list[str]:
    if len(ids) != EXPECTED_NORMAL_TRAIN_COUNT:
        raise ValueError(f"Expected {EXPECTED_NORMAL_TRAIN_COUNT} training photo IDs, got {len(ids)}")
    ordered = sorted(ids, key=lambda value: (
        hashlib.sha256(f"{SELECTION_SALT}\0{value}".encode()).digest(), value,
    ))
    return ordered[:count]


def normal_members(infos: list[zipfile.ZipInfo]) -> dict[str, dict[str, zipfile.ZipInfo]]:
    by_region: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        if len(parts) != 5 or parts[0] != "extracted" or parts[2:4] != ("train", "good"):
            continue
        region = parts[1]
        stem = PurePosixPath(info.filename).stem
        if stem in by_region[region]:
            raise ValueError(f"Duplicate normal ID: {region}/{stem}")
        by_region[region][stem] = info
    if len(by_region) != EXPECTED_REGION_COUNT:
        raise ValueError(f"Expected {EXPECTED_REGION_COUNT} normal regions, got {len(by_region)}")
    reference: set[str] | None = None
    for region, rows in sorted(by_region.items()):
        if len(rows) != EXPECTED_NORMAL_TRAIN_COUNT:
            raise ValueError(f"{region}: expected 1687 normal training photos, got {len(rows)}")
        if reference is not None and set(rows) != reference:
            raise ValueError(f"{region}: normal photo IDs differ between regions")
        reference = set(rows)
    return dict(by_region)


def decode_normal(raw: bytes, name: str) -> tuple[tuple[int, int], str]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            rgb = image.convert("RGB")
            if rgb.width <= 0 or rgb.height <= 0:
                raise ValueError("empty image dimensions")
            return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()
    except Exception as exc:
        raise ValueError(f"Unreadable calibration normal: {name}") from exc


def summarize_decoded_duplicates(
    rows: list[dict[str, object]], selected: list[str], regions: list[str],
) -> tuple[dict[str, dict[str, object]], list[list[str]]]:
    """Distinguish identical tiles from identical 12-region photo representations."""
    by_photo: dict[str, dict[str, str]] = defaultdict(dict)
    by_region: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        region = str(row["region"])
        photo_id = str(row["photo_id"])
        if region not in regions or photo_id not in selected or region in by_photo[photo_id]:
            raise ValueError(f"Unexpected or repeated calibration row: {region}/{photo_id}")
        fingerprint = f'{row["width"]}x{row["height"]}:{row["decoded_rgb_sha256"]}'
        by_photo[photo_id][region] = fingerprint
        by_region[region][fingerprint].append(photo_id)
    if len(rows) != len(selected) * len(regions) or any(
        set(by_photo[photo_id]) != set(regions) for photo_id in selected
    ):
        raise ValueError("Incomplete 12-region calibration representation")
    tile_summary = {}
    for region in regions:
        groups = [ids for ids in by_region[region].values() if len(ids) > 1]
        tile_summary[region] = {
            "repeated_tile_groups": len(groups),
            "photos_in_repeated_tile_groups": sum(map(len, groups)),
            "example_photo_id_groups": groups[:5],
        }
    composites: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for photo_id in selected:
        composites[tuple(by_photo[photo_id][region] for region in regions)].append(photo_id)
    repeated_composites = [ids for ids in composites.values() if len(ids) > 1]
    return tile_summary, repeated_composites


def run(archive: Path, registration: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    if not archive.is_file() or not registration.is_file():
        raise FileNotFoundError("Archive or registration JSON is missing")
    record = json.loads(registration.read_text(encoding="utf-8"))
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
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        audit = audit_members(infos)
        if audit["status"] != "registered_metadata_only":
            raise ValueError(f"Archive metadata audit failed: {audit['errors']}")
        normal = normal_members(infos)
        first_region = sorted(normal)[0]
        selected = select_calibration_ids(set(normal[first_region]))
        rows: list[dict[str, object]] = []
        dimensions: dict[str, int] = defaultdict(int)
        for region in sorted(normal):
            for index, photo_id in enumerate(selected, start=1):
                info = normal[region][photo_id]
                raw = source.read(info)
                size, pixel_sha256 = decode_normal(raw, info.filename)
                dimensions[f"{size[0]}x{size[1]}"] += 1
                rows.append({
                    "region": region, "photo_id": photo_id, "member": info.filename,
                    "width": size[0], "height": size[1],
                    "compressed_image_sha256": hashlib.sha256(raw).hexdigest(),
                    "decoded_rgb_sha256": pixel_sha256,
                })
                if index % 50 == 0 or index == len(selected):
                    print(f"{region} normal preflight {index}/{len(selected)}", flush=True)
    tile_duplicates, composite_duplicates = summarize_decoded_duplicates(
        rows, selected, sorted(normal),
    )
    selected_digest = hashlib.sha256(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "status": (
            "normal_preflight_needs_review_test_sealed"
            if composite_duplicates else "normal_preflight_passed_test_sealed"
        ),
        "protocol_version": 2,
        "archive_sha256": sha256,
        "registration_json_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
        "selection_rule": f"SHA256({SELECTION_SALT}\\0photo_id), ascending",
        "selected_photo_ids": selected,
        "selected_photo_ids_sha256": selected_digest,
        "calibration_original_photos": len(selected),
        "calibration_region_images_decoded": len(rows),
        "regions": len(normal),
        "dimensions": dict(sorted(dimensions.items())),
        "calibration_rows": rows,
        "tile_duplicate_summary": tile_duplicates,
        "identical_12_region_photo_id_groups": composite_duplicates,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
        "note": (
            "Identical individual tiles are recorded but do not fail the preflight. "
            "Identical complete 12-region representations require review. "
            "Same physical board; normal calibration is an adaptation, not zero-shot evidence."
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
        "calibration_original_photos": report["calibration_original_photos"],
        "calibration_region_images_decoded": report["calibration_region_images_decoded"],
        "selected_photo_ids_sha256": report["selected_photo_ids_sha256"],
        "identical_12_region_photo_id_groups": report["identical_12_region_photo_id_groups"],
        "test_images_decoded": report["test_images_decoded"],
        "model_scores_computed": report["model_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "normal_preflight_needs_review_test_sealed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
