#!/usr/bin/env python3
"""Safely register the frozen Real-IAD C1 subsets and build portable manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.manifest import checksum, summary, verify_manifest
from defectops.realiad_protocol import (
    ANNOTATION_VARIANT,
    REAL_IAD_CATEGORIES,
    REAL_IAD_SOURCE_SHA256,
    build_realiad_manifest,
)


MANIFEST_NAMES = {
    "realiad_transistor1": "realiad-transistor1-v1.json",
    "realiad_plastic_nut": "realiad-plastic-nut-v1.json",
}


def _validate_archive(path: Path, expected_name: str) -> Path:
    path = path.resolve(strict=True)
    actual = checksum(path)
    expected = REAL_IAD_SOURCE_SHA256[expected_name]
    if actual != expected:
        raise ValueError(
            f"Unexpected SHA-256 for {expected_name}: expected {expected}, found {actual}"
        )
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt ZIP member in {expected_name}: {bad}")
    return path


def _safe_member(value: object, prefix: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Real-IAD JSON contains an invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe Real-IAD ZIP member: {value}")
    member = PurePosixPath(prefix, path).as_posix()
    if not member.startswith(f"{prefix}/"):
        raise ValueError(f"Real-IAD ZIP member escaped its category: {member}")
    return member


def _read_annotation(archive: zipfile.ZipFile, source_object: str) -> tuple[str, bytes, dict]:
    member = f"{ANNOTATION_VARIANT}/{source_object}.json"
    try:
        raw_bytes = archive.read(member)
    except KeyError as exc:
        raise ValueError(f"Missing annotation member: {member}") from exc
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), list) or not isinstance(
        raw.get("test"), list
    ):
        raise ValueError(f"Malformed Real-IAD annotation: {member}")
    return member, raw_bytes, raw


def _referenced_members(source_object: str, annotation: dict) -> set[str]:
    members: set[str] = set()
    for split in ("train", "test"):
        for item in annotation[split]:
            members.add(_safe_member(item.get("image_path"), source_object))
            if item.get("mask_path") is not None:
                members.add(_safe_member(item["mask_path"], source_object))
    return members


def _extract_selected(archive_path: Path, members: set[str], destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        by_name = {item.filename: item for item in archive.infolist()}
        missing = sorted(members - set(by_name))
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} referenced members: {missing[0]}")
        for member in sorted(members):
            info = by_name[member]
            if info.is_dir() or info.file_size <= 0:
                raise ValueError(f"Referenced member is empty or not a file: {member}")
            target = destination / PurePosixPath(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def register(args: argparse.Namespace) -> list[dict]:
    root = args.root.resolve(strict=True)
    jsons_zip = _validate_archive(args.jsons_zip, "realiad_jsons.zip")
    image_zips = {
        "transistor1": _validate_archive(args.transistor_zip, "transistor1.zip"),
        "plastic_nut": _validate_archive(args.plastic_nut_zip, "plastic_nut.zip"),
    }
    target = root / "real_iad"
    manifests_directory = root / "manifests"
    manifest_paths = {
        category: manifests_directory / name for category, name in MANIFEST_NAMES.items()
    }
    if target.exists():
        raise FileExistsError(f"Real-IAD registration target already exists: {target}")
    existing = [str(path) for path in manifest_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Real-IAD manifest already exists: {existing[0]}")

    manifests: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=".real-iad-staging-", dir=root) as temporary:
        stage_root = Path(temporary)
        annotations_root = stage_root / "real_iad" / "annotations" / ANNOTATION_VARIANT
        images_root = stage_root / "real_iad" / "images"
        with zipfile.ZipFile(jsons_zip) as annotation_archive:
            for category, source_object in REAL_IAD_CATEGORIES.items():
                _, raw_bytes, annotation = _read_annotation(annotation_archive, source_object)
                annotations_root.mkdir(parents=True, exist_ok=True)
                (annotations_root / f"{source_object}.json").write_bytes(raw_bytes)
                _extract_selected(
                    image_zips[source_object],
                    _referenced_members(source_object, annotation),
                    images_root,
                )
                manifests[category] = build_realiad_manifest(
                    stage_root, category, seed=args.seed, calibration_fraction=args.calibration_fraction
                )
        shutil.move(str(stage_root / "real_iad"), target)

    manifests_directory.mkdir(parents=True, exist_ok=True)
    try:
        for category, manifest in manifests.items():
            with manifest_paths[category].open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
    except Exception:
        raise RuntimeError(
            "Images were registered but manifest writing failed; preserve data/real_iad for diagnosis"
        )
    return [summary(manifests[category]) for category in sorted(manifests)]


def verify(args: argparse.Namespace) -> list[dict]:
    root = args.root.resolve(strict=True)
    results = []
    for category, name in MANIFEST_NAMES.items():
        manifest = json.loads((root / "manifests" / name).read_text(encoding="utf-8"))
        if manifest.get("categories") != [category]:
            raise ValueError(f"Unexpected category in {name}")
        verify_manifest(root, manifest)
        results.append(summary(manifest))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("register")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--jsons-zip", type=Path, required=True)
    create.add_argument("--transistor-zip", type=Path, required=True)
    create.add_argument("--plastic-nut-zip", type=Path, required=True)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--calibration-fraction", type=float, default=0.2)
    check = subparsers.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = register(args) if args.command == "register" else verify(args)
        print(json.dumps({"status": f"real_iad_{args.command}_passed", "manifests": results}, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
