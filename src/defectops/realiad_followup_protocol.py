"""Preregister and deterministically seal the Real-IAD follow-up split."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from defectops.manifest import canonical, checksum
from defectops.realiad_protocol import REAL_IAD_SOURCE_SHA256


FOLLOWUP_CATEGORIES = {
    "realiad_pcb": "pcb",
    "realiad_terminalblock": "terminalblock",
}
EXPECTED_REGISTRATION_STATUS = (
    "annotation_metadata_observed_images_and_scores_unobserved"
)
EXPECTED_EVIDENCE_STATUS = (
    "realiad_transistor1_and_plastic_nut_c1_test_observed_new_categories_unobserved"
)
ANNOTATION_VARIANT = "realiad_jsons_sv"
_CAMERA_PATTERN = re.compile(r"_C([1-5])_")


def _expected_candidates() -> list[dict[str, Any]]:
    return [
        {
            "id": "baseline_raw",
            "layers": ["layer2", "layer3"],
            "image_score": "anomalib_patchcore_raw",
            "top_fraction": 0.0,
        },
        {
            "id": "map_top_1pct",
            "layers": ["layer2", "layer3"],
            "image_score": "mean_of_top_map_fraction",
            "top_fraction": 0.01,
        },
        {
            "id": "map_top_5pct",
            "layers": ["layer2", "layer3"],
            "image_score": "mean_of_top_map_fraction",
            "top_fraction": 0.05,
        },
        {
            "id": "fine_features_top_1pct",
            "layers": ["layer1", "layer2"],
            "image_score": "mean_of_top_map_fraction",
            "top_fraction": 0.01,
        },
    ]


def load_realiad_followup_protocol(path: Path) -> dict[str, Any]:
    """Load the follow-up contract and reject silent scope or selection changes."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema_version": config.get("schema_version") == 1,
        "protocol": config.get("protocol") == "A-normal-only-v2",
        "registration_status": config.get("registration_status")
        == EXPECTED_REGISTRATION_STATUS,
        "evidence_status": config.get("evidence_status") == EXPECTED_EVIDENCE_STATUS,
        "source_dataset": config.get("source_dataset") == "Real-IAD-256",
        "annotation_variant": config.get("annotation_variant") == ANNOTATION_VARIANT,
        "categories": config.get("categories")
        == ["realiad_pcb", "realiad_terminalblock"],
        "views": config.get("views") == ["C1"],
        "split_seed": config.get("split_seed") == 20260922,
        "calibration_group_fraction": config.get("calibration_group_fraction") == 0.20,
        "development_group_fraction": config.get("development_group_fraction") == 0.50,
        "group_key": config.get("group_key")
        == "source_object/relative_parent_directory",
        "holdout_stratification": config.get("holdout_stratification") == "defect_type",
        "leakage_control": config.get("leakage_control")
        == {
            "group_disjoint": True,
            "exclude_previously_observed_categories": [
                "realiad_transistor1",
                "realiad_plastic_nut",
            ],
            "exclude_other_views_of_observed_samples": True,
            "final_images_may_be_opened_after_candidate_lock": True,
            "final_reuse_after_change_is_exploratory": True,
        },
        "base_model": config.get("base_model")
        == {
            "model": "patchcore",
            "backbone": "wide_resnet50_2",
            "coreset_sampling_ratio": 0.10,
            "num_neighbors": 9,
            "image_size": 256,
            "development_seed": 42,
            "final_seeds": [42, 43, 44],
        },
        "calibration": config.get("calibration")
        == {
            "image_quantile": 0.95,
            "pixel_quantile": 0.995,
            "quantile_method": "higher",
            "decision_rule": "score >= threshold",
            "normal_only": True,
        },
        "candidates": config.get("candidates") == _expected_candidates(),
        "selection": config.get("selection")
        == {
            "candidate_count": 4,
            "development_only": True,
            "primary_metric": "macro_escape_rate",
            "maximum_category_overkill_rate": 0.10,
            "tie_breakers": [
                "macro_image_auroc",
                "macro_overkill_rate",
                "warm_p95_latency_ms",
            ],
            "choose_exactly_one_candidate": True,
            "automatic_pass_allowed": False,
        },
        "final_gate": config.get("final_gate")
        == {
            "required_before_open": [
                "selected_candidate_id",
                "selection_report_sha256",
                "git_commit",
                "final_manifest_sha256",
            ],
            "run_once": True,
            "post_final_tuning_allowed": False,
            "report_category_seed_values_and_macro": True,
            "report_image_level_95pct_confidence_intervals": True,
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Real-IAD follow-up protocol mismatch: " + ", ".join(failed))
    return config


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Real-IAD annotation contains an invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe Real-IAD annotation path: {value}")
    return path.as_posix()


def _rank(seed: int, category: str, purpose: str, group: str) -> tuple[str, str]:
    value = f"{seed}:{category}:{purpose}:{group}"
    return hashlib.sha256(value.encode()).hexdigest(), group


def _split_groups(
    groups: set[str],
    fraction: float,
    seed: int,
    category: str,
    purpose: str,
) -> set[str]:
    if len(groups) < 2:
        raise ValueError(f"{category}/{purpose}: at least two sample groups are required")
    count = math.ceil(len(groups) * fraction)
    if count <= 0 or count >= len(groups):
        raise ValueError(f"{category}/{purpose}: split leaves an empty partition")
    return set(sorted(groups, key=lambda group: _rank(seed, category, purpose, group))[:count])


def _entry(
    category: str,
    source_object: str,
    official_split: str,
    planned_split: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    if item.get("category") != source_object:
        raise ValueError(f"{category}: annotation category does not match its source object")
    image_path = _safe_relative(item.get("image_path"))
    camera = _CAMERA_PATTERN.search(image_path)
    if camera is None or camera.group(1) != "1":
        raise ValueError(f"{category}: follow-up protocol accepts only camera C1")
    anomaly_class = item.get("anomaly_class")
    if not isinstance(anomaly_class, str) or not anomaly_class:
        raise ValueError(f"{category}: missing anomaly class")
    label = int(anomaly_class != "OK")
    mask_path = item.get("mask_path")
    if label:
        mask_path = _safe_relative(mask_path)
    elif mask_path is not None:
        raise ValueError(f"{category}: normal entry unexpectedly has a mask")
    parent = PurePosixPath(image_path).parent.as_posix()
    return {
        "category": category,
        "source_object": source_object,
        "official_split": official_split,
        "planned_split": planned_split,
        "label": label,
        "defect_type": anomaly_class,
        "view": "C1",
        "sample_group": f"{source_object}/{parent}",
        "source_image_path": image_path,
        "source_mask_path": mask_path,
    }


def build_realiad_followup_plan(
    jsons_zip: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a path-only plan without extracting, decoding, previewing, or scoring images."""
    config = load_realiad_followup_protocol(config_path)
    jsons_zip = jsons_zip.resolve(strict=True)
    if checksum(jsons_zip) != REAL_IAD_SOURCE_SHA256["realiad_jsons.zip"]:
        raise ValueError("Unexpected SHA-256 for realiad_jsons.zip")
    records: list[dict[str, Any]] = []
    annotation_sha256: dict[str, str] = {}
    with zipfile.ZipFile(jsons_zip) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt annotation ZIP member: {bad}")
        for category in config["categories"]:
            source_object = FOLLOWUP_CATEGORIES[category]
            member = f"{ANNOTATION_VARIANT}/{source_object}.json"
            try:
                raw_bytes = archive.read(member)
            except KeyError as exc:
                raise ValueError(f"Missing annotation member: {member}") from exc
            annotation_sha256[member] = hashlib.sha256(raw_bytes).hexdigest()
            annotation = json.loads(raw_bytes)
            if (
                not isinstance(annotation, dict)
                or set(annotation) != {"meta", "train", "test"}
                or annotation["meta"].get("prefix") != f"{source_object}/"
                or annotation["meta"].get("normal_class") != "OK"
                or not isinstance(annotation["train"], list)
                or not isinstance(annotation["test"], list)
            ):
                raise ValueError(f"Unexpected Real-IAD annotation contract: {member}")
            train_items = annotation["train"]
            test_items = annotation["test"]
            if not train_items or not test_items:
                raise ValueError(f"{category}: official train and test are required")
            if any(item.get("anomaly_class") != "OK" for item in train_items):
                raise ValueError(f"{category}: official training entries must be normal")

            train_groups = {
                f"{source_object}/{PurePosixPath(_safe_relative(item['image_path'])).parent.as_posix()}"
                for item in train_items
            }
            calibration_groups = _split_groups(
                train_groups,
                config["calibration_group_fraction"],
                config["split_seed"],
                category,
                "calibration",
            )
            for item in train_items:
                group = (
                    f"{source_object}/"
                    f"{PurePosixPath(_safe_relative(item['image_path'])).parent.as_posix()}"
                )
                split = "calibration" if group in calibration_groups else "train"
                records.append(_entry(category, source_object, "train", split, item))

            test_by_defect: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in test_items:
                anomaly_class = item.get("anomaly_class")
                if not isinstance(anomaly_class, str) or not anomaly_class:
                    raise ValueError(f"{category}: malformed test anomaly class")
                test_by_defect[anomaly_class].append(item)
            if "OK" not in test_by_defect or len(test_by_defect) < 2:
                raise ValueError(f"{category}: test must contain normal and anomalous images")
            for defect_type, items in sorted(test_by_defect.items()):
                groups = {
                    f"{source_object}/"
                    f"{PurePosixPath(_safe_relative(item['image_path'])).parent.as_posix()}"
                    for item in items
                }
                development_groups = _split_groups(
                    groups,
                    config["development_group_fraction"],
                    config["split_seed"],
                    category,
                    f"development:{defect_type}",
                )
                for item in items:
                    group = (
                        f"{source_object}/"
                        f"{PurePosixPath(_safe_relative(item['image_path'])).parent.as_posix()}"
                    )
                    split = "development" if group in development_groups else "final_holdout"
                    records.append(_entry(category, source_object, "test", split, item))

    image_paths = [f"{row['source_object']}/{row['source_image_path']}" for row in records]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Follow-up annotations contain duplicate image paths")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        group_splits[f"{row['category']}:{row['sample_group']}"].add(row["planned_split"])
    overlap = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if overlap:
        raise ValueError(f"Sample groups span planned splits: {overlap[0]}")

    records.sort(key=lambda row: (row["category"], row["source_image_path"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "real_iad",
        "protocol": config["protocol"],
        "registration_status": config["registration_status"],
        "evidence_status": config["evidence_status"],
        "annotation_variant": ANNOTATION_VARIANT,
        "categories": config["categories"],
        "split_seed": config["split_seed"],
        "calibration_group_fraction": config["calibration_group_fraction"],
        "development_group_fraction": config["development_group_fraction"],
        "split_algorithm": (
            "sha256(seed:category:purpose:sample_group), ceil(group_count*fraction), "
            "test stratified by defect_type"
        ),
        "annotation_zip_sha256": REAL_IAD_SOURCE_SHA256["realiad_jsons.zip"],
        "annotation_sha256": dict(sorted(annotation_sha256.items())),
        "config_sha256": checksum(config_path),
        "records": records,
    }
    payload["plan_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def summarize_realiad_followup_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return auditable image and sample-group counts without exposing image content."""
    images = Counter(
        f"{row['category']}/{row['planned_split']}/"
        f"{'anomaly' if row['label'] else 'normal'}"
        for row in plan["records"]
    )
    groups = Counter(
        (row["category"], row["planned_split"], row["sample_group"])
        for row in plan["records"]
    )
    group_counts = Counter(f"{category}/{split}" for category, split, _ in groups)
    return {
        "plan_sha256": plan["plan_sha256"],
        "image_counts": dict(sorted(images.items())),
        "sample_group_counts": dict(sorted(group_counts.items())),
    }
