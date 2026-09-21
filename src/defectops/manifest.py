"""Portable MVTec AD inventory. This module never decodes or trains on images."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

MVTEC_CATEGORIES = {"transistor", "cable", "metal_nut"}
CATEGORIES = MVTEC_CATEGORIES | {"realiad_transistor1", "realiad_plastic_nut"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_manifest(root: Path, categories: list[str], seed: int = 42,
                   calibration_fraction: float = 0.2) -> dict:
    root = root.resolve(strict=True)
    if not categories or len(set(categories)) != len(categories) or set(categories) - MVTEC_CATEGORIES:
        raise ValueError("Choose unique categories from cable, metal_nut, transistor")
    if type(seed) is not int or not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("seed must be an integer; calibration_fraction must be between 0 and 1")
    records = []
    seen_images: dict[str, str] = {}

    def relative(path: Path) -> str:
        # Reject symlink escapes so a manifest cannot inventory arbitrary outside files.
        path.resolve(strict=True).relative_to(root)
        return path.relative_to(root).as_posix()

    for category in sorted(categories):
        base = root / category
        train = sorted((base / "train").glob("*/*.png"))
        test = sorted((base / "test").glob("*/*.png"))
        if len(train) < 2 or not test:
            raise ValueError(f"{category}: missing train/test PNG files or insufficient normal training data")
        if any(path.parent.name != "good" for path in train):
            raise ValueError(f"{category}: official train must contain only good images")
        if not any(p.parent.name == "good" for p in test) or not any(p.parent.name != "good" for p in test):
            raise ValueError(f"{category}: test must include good and anomalous images")
        n_calibration = math.ceil(len(train) * calibration_fraction)
        if n_calibration >= len(train):
            raise ValueError(f"{category}: split leaves no training images")
        ranked = sorted(train, key=lambda p: (hashlib.sha256(
            f"{seed}:{relative(p)}".encode()).hexdigest(), relative(p)))
        calibration_paths = set(ranked[:n_calibration])
        used_masks = set()
        for path in train + test:
            rel = relative(path)
            if path.stat().st_size == 0:
                raise ValueError(f"Empty image: {rel}")
            digest = checksum(path)
            if digest in seen_images:
                raise ValueError(f"Duplicate image content: {seen_images[digest]} and {rel}")
            seen_images[digest] = rel
            official_split = path.parent.parent.name
            defect = path.parent.name
            mask = None
            mask_digest = None
            if official_split == "test" and defect != "good":
                mask_path = base / "ground_truth" / defect / f"{path.stem}_mask.png"
                if not mask_path.is_file() or mask_path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty mask for {rel}")
                mask = relative(mask_path)
                mask_digest = checksum(mask_path)
                used_masks.add(mask_path)
            split = "test" if official_split == "test" else (
                "calibration" if path in calibration_paths else "train")
            records.append({"image_path": rel, "image_sha256": digest, "category": category,
                            "official_split": official_split, "split": split,
                            "label": int(defect != "good"), "defect_type": defect,
                            "mask_path": mask, "mask_sha256": mask_digest})
        actual_masks = set((base / "ground_truth").glob("*/*.png"))
        if actual_masks != used_masks:
            raise ValueError(f"{category}: orphaned or unexpected ground-truth masks")
    records.sort(key=lambda record: record["image_path"])
    payload = {"schema_version": 1, "dataset": "mvtec_ad", "protocol": "A-normal-only-v1",
               "categories": sorted(categories), "split_seed": seed,
               "calibration_fraction": calibration_fraction,
               "split_algorithm": "sha256(seed:relative_path), ceil(n*fraction)",
               "records": records}
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def verify_manifest(root: Path, manifest: dict) -> None:
    try:
        if manifest.get("dataset") == "real_iad":
            from defectops.realiad_protocol import build_realiad_manifest

            categories = manifest["categories"]
            if not isinstance(categories, list) or len(categories) != 1:
                raise ValueError("Real-IAD manifests contain exactly one category")
            expected = build_realiad_manifest(
                root,
                categories[0],
                manifest["split_seed"],
                manifest["calibration_fraction"],
            )
        else:
            expected = build_manifest(root, manifest["categories"], manifest["split_seed"],
                                      manifest["calibration_fraction"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed manifest") from exc
    if canonical(manifest) != canonical(expected):
        raise ValueError("Manifest mismatch: files, checksums, protocol or split assignments changed")


def summary(manifest: dict) -> dict:
    counts = Counter(f"{r['category']}/{r['split']}/{'anomaly' if r['label'] else 'good'}"
                     for r in manifest["records"])
    return {"manifest_sha256": manifest["manifest_sha256"], "counts": dict(sorted(counts.items()))}
