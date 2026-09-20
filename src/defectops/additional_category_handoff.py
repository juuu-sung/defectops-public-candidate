"""Validate and import the cable/metal_nut data-registration handoff."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from defectops.manifest import canonical


EXPECTED_COUNTS = {
    "cable": {
        "train_good": 179,
        "calibration_good": 45,
        "test_good": 58,
        "test_anomaly": 92,
    },
    "metal_nut": {
        "train_good": 176,
        "calibration_good": 44,
        "test_good": 22,
        "test_anomaly": 93,
    },
}
EXPECTED_ENTRIES = Counter(
    {"cable-v1.json": 1, "metal-nut-v1.json": 1, "summary.md": 2}
)
EXPECTED_RECORD_KEYS = {
    "image_path",
    "image_sha256",
    "category",
    "official_split",
    "split",
    "label",
    "defect_type",
    "mask_path",
    "mask_sha256",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(data: bytes, category: str) -> dict[str, Any]:
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {category} manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object for {category}")
    stored_hash = manifest.get("manifest_sha256")
    unhashed = deepcopy(manifest)
    unhashed.pop("manifest_sha256", None)
    calculated_hash = _sha256(canonical(unhashed))
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "dataset": manifest.get("dataset") == "mvtec_ad",
        "protocol": manifest.get("protocol") == "A-normal-only-v1",
        "categories": manifest.get("categories") == [category],
        "split_seed": manifest.get("split_seed") == 42,
        "calibration_fraction": manifest.get("calibration_fraction") == 0.2,
        "split_algorithm": manifest.get("split_algorithm")
        == "sha256(seed:relative_path), ceil(n*fraction)",
        "manifest_sha256": stored_hash == calculated_hash,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{category} manifest contract mismatch: {', '.join(failed)}")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{category} manifest records must be a non-empty list")
    if records != sorted(records, key=lambda row: row.get("image_path", "")):
        raise ValueError(f"{category} manifest records are not sorted by image_path")

    counts = Counter()
    image_paths = set()
    image_hashes = set()
    mask_paths = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != EXPECTED_RECORD_KEYS:
            raise ValueError(f"{category} manifest contains an unexpected record shape")
        image_path = record["image_path"]
        image_hash = record["image_sha256"]
        if (
            not isinstance(image_path, str)
            or PurePosixPath(image_path).is_absolute()
            or ".." in PurePosixPath(image_path).parts
            or not image_path.startswith(f"{category}/")
            or not image_path.endswith(".png")
            or image_path in image_paths
        ):
            raise ValueError(f"Unsafe or duplicate {category} image path: {image_path!r}")
        if not isinstance(image_hash, str) or not HEX_SHA256.fullmatch(image_hash):
            raise ValueError(f"Invalid image SHA-256 for {image_path}")
        if image_hash in image_hashes:
            raise ValueError(f"Duplicate image content hash in {category}")
        image_paths.add(image_path)
        image_hashes.add(image_hash)
        if record["category"] != category or type(record["label"]) is not int:
            raise ValueError(f"Invalid category or label for {image_path}")

        split = record["split"]
        official_split = record["official_split"]
        label = record["label"]
        defect_type = record["defect_type"]
        mask_path = record["mask_path"]
        mask_hash = record["mask_sha256"]
        if split in {"train", "calibration"}:
            if (
                official_split != "train"
                or label != 0
                or defect_type != "good"
                or mask_path is not None
                or mask_hash is not None
            ):
                raise ValueError(f"Invalid normal development record: {image_path}")
            counts[f"{split}_good"] += 1
        elif split == "test" and official_split == "test":
            if label == 0:
                if defect_type != "good" or mask_path is not None or mask_hash is not None:
                    raise ValueError(f"Invalid normal test record: {image_path}")
                counts["test_good"] += 1
            elif label == 1:
                expected_mask = (
                    f"{category}/ground_truth/{defect_type}/"
                    f"{PurePosixPath(image_path).stem}_mask.png"
                )
                if (
                    defect_type == "good"
                    or mask_path != expected_mask
                    or not isinstance(mask_hash, str)
                    or not HEX_SHA256.fullmatch(mask_hash)
                    or mask_path in mask_paths
                ):
                    raise ValueError(f"Invalid anomalous test record: {image_path}")
                mask_paths.add(mask_path)
                counts["test_anomaly"] += 1
            else:
                raise ValueError(f"Invalid test label for {image_path}")
        else:
            raise ValueError(f"Invalid split for {image_path}")
    if dict(counts) != EXPECTED_COUNTS[category]:
        raise ValueError(
            f"{category} manifest counts differ: {dict(counts)} != {EXPECTED_COUNTS[category]}"
        )
    return manifest


def validate_handoff(zip_path: Path, expected_zip_sha256: str | None = None) -> dict[str, Any]:
    """Validate archive identity, safe entries, manifests, and audit summaries."""
    zip_bytes = zip_path.read_bytes()
    zip_sha256 = _sha256(zip_bytes)
    if expected_zip_sha256 is not None and zip_sha256 != expected_zip_sha256.lower():
        raise ValueError("Handoff ZIP SHA-256 differs from the Windows output")
    if len(zip_bytes) > 10 * 1024 * 1024:
        raise ValueError("Handoff ZIP is unexpectedly large")

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid handoff ZIP") from exc
    with archive:
        entries = archive.infolist()
        if Counter(entry.filename for entry in entries) != EXPECTED_ENTRIES:
            raise ValueError("Handoff ZIP entries differ from the expected contract")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
                raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
            if entry.is_dir() or entry.file_size > 2 * 1024 * 1024:
                raise ValueError(f"Unexpected ZIP entry size or type: {entry.filename}")

        manifest_bytes = {
            "cable": archive.read(next(e for e in entries if e.filename == "cable-v1.json")),
            "metal_nut": archive.read(
                next(e for e in entries if e.filename == "metal-nut-v1.json")
            ),
        }
        manifests = {
            category: _read_manifest(data, category)
            for category, data in manifest_bytes.items()
        }
        all_hashes = [
            row["image_sha256"]
            for manifest in manifests.values()
            for row in manifest["records"]
        ]
        if len(all_hashes) != len(set(all_hashes)):
            raise ValueError("Image content hash is duplicated across categories")

        summary_entries = [entry for entry in entries if entry.filename == "summary.md"]
        summaries: dict[str, str] = {}
        for entry in summary_entries:
            try:
                text = archive.read(entry).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Audit summary is not UTF-8") from exc
            matches = [category for category in EXPECTED_COUNTS if f"| {category}/" in text]
            if len(matches) != 1 or matches[0] in summaries:
                raise ValueError("Could not uniquely associate an audit summary with a category")
            category = matches[0]
            expected = EXPECTED_COUNTS[category]
            required = [
                "점검 상태: **passed**",
                "- 오류: 0건",
                "- 디코딩 픽셀 중복: 0쌍",
                f"| {category}/train/good | {expected['train_good']} |",
                f"| {category}/calibration/good | {expected['calibration_good']} |",
                f"| {category}/test/good | {expected['test_good']} |",
                f"| {category}/test/anomaly | {expected['test_anomaly']} |",
            ]
            if any(value not in text for value in required):
                raise ValueError(f"{category} audit summary differs from the manifest contract")
            summaries[category] = text

    return {
        "status": "validated_additional_category_handoff",
        "zip_sha256": zip_sha256,
        "manifests": manifests,
        "manifest_bytes": manifest_bytes,
        "summaries": summaries,
        "counts": EXPECTED_COUNTS,
    }


def import_handoff(
    result: dict[str, Any], manifest_directory: Path, report_directory: Path
) -> dict[str, str]:
    """Import validated manifests and rename the duplicate audit summary entries safely."""
    manifest_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    targets = {
        "cable_manifest": manifest_directory / "cable-v1.json",
        "metal_nut_manifest": manifest_directory / "metal-nut-v1.json",
        "cable_report": report_directory / "cable-data-audit-v1.md",
        "metal_nut_report": report_directory / "metal-nut-data-audit-v1.md",
    }
    if any(path.exists() for path in targets.values()):
        existing = [str(path) for path in targets.values() if path.exists()]
        raise FileExistsError("Import target already exists: " + ", ".join(existing))
    for category, key in (("cable", "cable_manifest"), ("metal_nut", "metal_nut_manifest")):
        targets[key].write_bytes(result["manifest_bytes"][category])
    for category, key, title in (
        ("cable", "cable_report", "Cable 데이터 준비 결과"),
        ("metal_nut", "metal_nut_report", "Metal Nut 데이터 준비 결과"),
    ):
        summary = result["summaries"][category]
        if not summary.startswith("# 데이터 준비 결과"):
            raise ValueError(f"Unexpected {category} audit heading")
        targets[key].write_text(
            summary.replace("# 데이터 준비 결과", f"# {title}", 1), encoding="utf-8"
        )
    return {name: str(path) for name, path in targets.items()}
