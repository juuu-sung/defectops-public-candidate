"""Frozen Real-IAD single-view inventory for the two selected external categories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from defectops.manifest import canonical, checksum


REAL_IAD_CATEGORIES = {
    "realiad_transistor1": "transistor1",
    "realiad_plastic_nut": "plastic_nut",
}
REAL_IAD_SOURCE_SHA256 = {
    "realiad_jsons.zip": "7299bfee9d8c90e4caf72f49dd1340ce425cdcd2a5f32c70339ddc2123a2c27f",
    "plastic_nut.zip": "c34b4d7eb5f6498b7ea1a847cef0e9fc1bc292e315d470b81e5ab25966b539cf",
    "transistor1.zip": "e36484433b37e713ea70142e0ae973bce475c8450f5c1f459d1d9e0bfb39d9af",
}
ANNOTATION_VARIANT = "realiad_jsons_sv"
_CAMERA_PATTERN = re.compile(r"_C([1-5])_")


def _source_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Real-IAD annotation contains an invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe Real-IAD annotation path: {value}")
    return path.as_posix()


def _resolved_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return path


def _load_annotation(root: Path, source_object: str) -> tuple[Path, dict[str, Any]]:
    path = root / "real_iad" / "annotations" / ANNOTATION_VARIANT / f"{source_object}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or set(raw) != {"meta", "train", "test"}
        or raw["meta"].get("prefix") != f"{source_object}/"
        or raw["meta"].get("normal_class") != "OK"
        or not isinstance(raw["train"], list)
        or not isinstance(raw["test"], list)
    ):
        raise ValueError(f"Unexpected Real-IAD annotation contract: {path}")
    return path, raw


def build_realiad_manifest(
    root: Path,
    category: str,
    seed: int = 42,
    calibration_fraction: float = 0.2,
) -> dict[str, Any]:
    """Inventory one registered Real-IAD single-view category without decoding images."""
    root = root.resolve(strict=True)
    if category not in REAL_IAD_CATEGORIES:
        raise ValueError("Choose realiad_plastic_nut or realiad_transistor1")
    if type(seed) is not int or not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("seed must be an integer; calibration_fraction must be between 0 and 1")

    source_object = REAL_IAD_CATEGORIES[category]
    annotation_path, annotation = _load_annotation(root, source_object)
    entries: list[tuple[str, dict[str, Any]]] = []
    for official_split in ("train", "test"):
        for item in annotation[official_split]:
            if not isinstance(item, dict) or item.get("category") != source_object:
                raise ValueError(f"Malformed Real-IAD {official_split} entry")
            entries.append((official_split, item))
    if len(entries) < 2:
        raise ValueError(f"{category}: insufficient annotated images")

    train_entries = [item for split, item in entries if split == "train"]
    test_entries = [item for split, item in entries if split == "test"]
    if not train_entries or not test_entries:
        raise ValueError(f"{category}: missing official train or test entries")
    if any(item.get("anomaly_class") != "OK" for item in train_entries):
        raise ValueError(f"{category}: official training entries must be view-level normal")
    if not any(item.get("anomaly_class") == "OK" for item in test_entries) or not any(
        item.get("anomaly_class") != "OK" for item in test_entries
    ):
        raise ValueError(f"{category}: test must contain normal and anomalous images")

    image_paths = [_source_relative(item.get("image_path")) for _, item in entries]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError(f"{category}: duplicate image paths in the official split")
    cameras = [(_CAMERA_PATTERN.search(path)) for path in image_paths]
    if any(match is None or match.group(1) != "1" for match in cameras):
        raise ValueError(f"{category}: single-view protocol must contain only camera C1")

    n_calibration = math.ceil(len(train_entries) * calibration_fraction)
    if n_calibration >= len(train_entries):
        raise ValueError(f"{category}: split leaves no training images")
    ranked = sorted(
        (_source_relative(item["image_path"]) for item in train_entries),
        key=lambda path: (hashlib.sha256(f"{seed}:{path}".encode()).hexdigest(), path),
    )
    calibration_paths = set(ranked[:n_calibration])

    records = []
    seen_digests: dict[str, str] = {}
    for official_split, item in entries:
        source_image = _source_relative(item["image_path"])
        image_relative = f"real_iad/images/{source_object}/{source_image}"
        image_path = _resolved_child(root, image_relative)
        if image_path.suffix.lower() not in {".jpg", ".jpeg"} or image_path.stat().st_size == 0:
            raise ValueError(f"Missing, empty, or unexpected Real-IAD image: {image_relative}")
        image_digest = checksum(image_path)
        if image_digest in seen_digests:
            raise ValueError(
                f"Duplicate Real-IAD image content: {seen_digests[image_digest]} and {image_relative}"
            )
        seen_digests[image_digest] = image_relative

        anomaly_class = item.get("anomaly_class")
        if not isinstance(anomaly_class, str) or not anomaly_class:
            raise ValueError(f"Missing anomaly class for {image_relative}")
        label = int(anomaly_class != "OK")
        source_mask = item.get("mask_path")
        mask_relative = None
        mask_digest = None
        if label:
            source_mask = _source_relative(source_mask)
            mask_relative = f"real_iad/images/{source_object}/{source_mask}"
            mask_path = _resolved_child(root, mask_relative)
            if mask_path.suffix.lower() != ".png" or mask_path.stat().st_size == 0:
                raise ValueError(f"Missing or invalid Real-IAD mask: {mask_relative}")
            mask_digest = checksum(mask_path)
        elif source_mask is not None:
            raise ValueError(f"View-level normal entry unexpectedly has a mask: {image_relative}")

        split = "test" if official_split == "test" else (
            "calibration" if source_image in calibration_paths else "train"
        )
        sample_directory = PurePosixPath(source_image).parent.as_posix()
        records.append(
            {
                "image_path": image_relative,
                "image_sha256": image_digest,
                "category": category,
                "source_object": source_object,
                "official_split": official_split,
                "split": split,
                "label": label,
                "defect_type": anomaly_class,
                "view": "C1",
                "sample_group": f"{source_object}/{sample_directory}",
                "mask_path": mask_relative,
                "mask_sha256": mask_digest,
            }
        )

    records.sort(key=lambda record: record["image_path"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "real_iad",
        "protocol": "A-normal-only-v1",
        "registration_protocol": "real-iad-single-view-c1-v1",
        "annotation_variant": ANNOTATION_VARIANT,
        "categories": [category],
        "source_object": source_object,
        "split_seed": seed,
        "calibration_fraction": calibration_fraction,
        "split_algorithm": "sha256(seed:source_image_path), ceil(n*fraction)",
        "annotation_path": annotation_path.relative_to(root).as_posix(),
        "annotation_sha256": checksum(annotation_path),
        "source_archive_sha256": {
            "realiad_jsons.zip": REAL_IAD_SOURCE_SHA256["realiad_jsons.zip"],
            f"{source_object}.zip": REAL_IAD_SOURCE_SHA256[f"{source_object}.zip"],
        },
        "records": records,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload
