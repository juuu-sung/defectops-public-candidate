"""Metadata-only preregistration for a new, cross-dataset PCB transfer test."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


CONFIG_NAME = "visa-pcb-external-v1.toml"
PLAN_NAME = "visa-pcb-external-plan-v1.json"
EXPECTED_COUNTS = {
    "pcb1": {"train_normal": 904, "test_normal": 100, "test_anomaly": 100},
    "pcb2": {"train_normal": 901, "test_normal": 100, "test_anomaly": 100},
    "pcb3": {"train_normal": 905, "test_normal": 101, "test_anomaly": 100},
    "pcb4": {"train_normal": 904, "test_normal": 101, "test_anomaly": 100},
}


def canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "protocol": "A-normal-only-v2",
        "status": "prospective_external_transfer_metadata_lock_before_visa_images_observed",
        "categories": list(EXPECTED_COUNTS),
        "calibration_normal_count_per_category": 56,
        "source_split_csv_sha256": "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995",
        "model_training_on_visa_allowed": False,
        "automatic_pass_allowed": False,
        "candidates": {
            "baseline": "mean_of_top_1pct_map",
            "challenger": "mean_of_top_0_25pct_map",
            "other_candidates_allowed": False,
        },
    }
    failed = [key for key, value in expected.items() if config.get(key) != value]
    if failed:
        raise ValueError("External transfer protocol differs: " + ", ".join(failed))
    config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    return config


def _safe_member(value: str, category: str, suffix: str) -> str:
    path = PurePosixPath(value)
    if (
        not value or "\\" in value or path.is_absolute() or ".." in path.parts
        or path.parts[0] != category or path.suffix.lower() != suffix
        or len(path.parts) < 4
    ):
        raise ValueError(f"Unsafe or unexpected VisA member: {value}")
    return path.as_posix()


def build_plan(csv_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    if csv_sha != config["source_split_csv_sha256"]:
        raise ValueError("Official VisA one-class split CSV SHA-256 differs")
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
        raise ValueError("Unexpected VisA one-class CSV columns")
    source_rows: dict[str, list[dict[str, str]]] = {category: [] for category in config["categories"]}
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    for row in reader:
        category = row["object"]
        if category not in source_rows:
            continue
        split, label = row["split"], row["label"]
        if (split, label) not in {("train", "normal"), ("test", "normal"), ("test", "anomaly")}:
            raise ValueError(f"Unexpected official split/label for {category}: {split}/{label}")
        image = _safe_member(row["image"], category, ".jpg")
        if image in seen_images:
            raise ValueError(f"Duplicate VisA image path: {image}")
        seen_images.add(image)
        if label == "anomaly":
            mask = _safe_member(row["mask"], category, ".png")
            if mask in seen_masks:
                raise ValueError(f"Duplicate VisA anomaly mask path: {mask}")
            seen_masks.add(mask)
        elif row["mask"]:
            raise ValueError("Normal VisA row unexpectedly has a mask")
        else:
            mask = ""
        source_rows[category].append({
            "category": category, "official_split": split, "official_label": label,
            "source_image_member": image, "source_mask_member": mask,
        })
    records = []
    counts: dict[str, dict[str, int]] = {}
    for category, rows in source_rows.items():
        counter = Counter(f"{row['official_split']}_{row['official_label']}" for row in rows)
        if dict(counter) != EXPECTED_COUNTS[category]:
            raise ValueError(f"Official VisA count changed for {category}: {dict(counter)}")
        normals = [row for row in rows if row["official_split"] == "train"]
        normals.sort(key=lambda row: (
            hashlib.sha256(
                f"visa-pcb-external-v1:{category}:{row['source_image_member']}".encode("utf-8")
            ).hexdigest(), row["source_image_member"]
        ))
        selected = normals[:config["calibration_normal_count_per_category"]]
        test = sorted(
            (row for row in rows if row["official_split"] == "test"),
            key=lambda row: row["source_image_member"],
        )
        records.extend({**row, "protocol_split": "calibration", "label": 0} for row in selected)
        records.extend({
            **row, "protocol_split": "external_final", "label": int(row["official_label"] == "anomaly")
        } for row in test)
        counts[category] = {
            "calibration_normal": len(selected),
            "external_final_normal": counter["test_normal"],
            "external_final_anomaly": counter["test_anomaly"],
        }
    plan = {
        "schema_version": 1,
        "status": "metadata_locked_external_final_unopened",
        "dataset": config["source_dataset"],
        "source_url": config["source_url"],
        "source_archive_bytes": config["source_archive_bytes"],
        "source_repo_commit": config["source_repo_commit"],
        "source_split_csv_sha256": csv_sha,
        "config_sha256": config["config_sha256"],
        "categories": config["categories"],
        "counts": counts,
        "records": records,
        "external_final_images_opened": False,
        "model_training_on_visa_allowed": False,
        "automatic_pass_allowed": False,
        "limitations": [
            "VisA pcb1-4 are different boards from Real-IAD pcb, so this tests cross-dataset transfer, not same-product improvement.",
            "The official metadata does not establish physical-part identity; duplicate and near-duplicate checks are still required.",
            "Archive SHA-256 and member bytes are not known until separate registration; no test image was decoded to build this plan.",
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
    return plan


def verify_plan(plan: dict[str, Any], csv_bytes: bytes, config: dict[str, Any]) -> None:
    if plan != build_plan(csv_bytes, config):
        raise ValueError("External PCB plan differs from the pinned CSV and protocol")
