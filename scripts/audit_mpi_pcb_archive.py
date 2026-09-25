"""Metadata-only gate for the official MPI-PCB extracted.zip archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


SOURCE_MD5 = "501d11d78a58ed0f26bc4d0baa34170e"
REGION_COUNT = 12
TEST_IMAGES_PER_CLASS = 55
PREFIX = "extracted/"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def archive_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - match Zenodo's published identity digest
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def audit_members(infos: list[zipfile.ZipInfo]) -> dict[str, object]:
    seen: set[str] = set()
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    errors: list[str] = []
    unsafe: list[str] = []
    for info in infos:
        name = info.filename
        parts = PurePosixPath(name).parts
        if (
            not name.startswith(PREFIX)
            or name.startswith("/")
            or "\\" in name
            or ".." in parts
            or info.flag_bits & 0x1
            or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
        ):
            unsafe.append(name)
            continue
        if name in seen:
            errors.append(f"duplicate ZIP member: {name}")
            continue
        seen.add(name)
        if info.is_dir():
            continue
        if len(parts) < 5:
            continue
        region = parts[1]
        group = "/".join(parts[2:-1])
        if group in {"train/good", "test/good", "test/defect", "ground_truth/defect"}:
            if len(parts) != 5 or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                errors.append(f"unexpected test member: {name}")
                continue
            stem = Path(name).stem
            if stem in groups[region][group]:
                errors.append(f"duplicate image ID: {region}/{group}/{stem}")
            groups[region][group].add(stem)
    if unsafe:
        errors.append(f"unsafe or encrypted ZIP members: {len(unsafe)}")
    if len(groups) != REGION_COUNT:
        errors.append(f"expected {REGION_COUNT} regions, found {len(groups)}")
    region_rows = []
    reference: dict[str, set[str]] = {}
    for region, types in sorted(groups.items()):
        counts = {kind: len(types.get(kind, set())) for kind in
                  ("train/good", "test/good", "test/defect", "ground_truth/defect")}
        for kind, count in counts.items():
            if kind == "train/good" and count == 0:
                errors.append(f"{region}/{kind}: no normal calibration candidates")
            elif kind != "train/good" and count != TEST_IMAGES_PER_CLASS:
                errors.append(f"{region}/{kind}: expected 55, found {count}")
        if types.get("test/defect", set()) != types.get("ground_truth/defect", set()):
            errors.append(f"{region}: defect image/mask stems differ")
        for kind in ("train/good", "test/good", "test/defect"):
            if kind in reference and types.get(kind, set()) != reference[kind]:
                errors.append(f"{region}: {kind} IDs differ between regions")
            reference.setdefault(kind, types.get(kind, set()))
        region_rows.append({"region": region, "counts": counts})
    return {
        "status": "registered_metadata_only" if not errors else "failed_metadata_audit",
        "region_count": len(groups),
        "regions": region_rows,
        "errors": errors,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive does not exist: {args.archive}")
    if args.output.exists():
        parser.error(f"output already exists and will not be overwritten: {args.output}")
    md5, sha256 = archive_hashes(args.archive)
    if md5 != SOURCE_MD5:
        parser.error(f"archive MD5 differs from Zenodo record: {md5}")
    with zipfile.ZipFile(args.archive) as source:
        report = audit_members(source.infolist())
    report.update({
        "source": "https://zenodo.org/records/8213098",
        "archive_name": args.archive.name,
        "archive_size_bytes": args.archive.stat().st_size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "rights_note": "Dataset license not explicitly specified; do not redistribute images or masks.",
    })
    if report["status"] != "registered_metadata_only":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "region_count": report["region_count"],
                      "archive_sha256": sha256, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
