"""Fixed MPI-PCB external-test contract, integrity audit and photo-level metrics."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tomllib
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from defectops.evaluation_metrics import binary_auroc, binary_average_precision, confusion_counts
from defectops.padim_evaluation import _fixed_image_metrics
from defectops.visa_pcb_external_protocol import canonical
from scripts.audit_mpi_pcb_archive import audit_members
from scripts.audit_mpi_pcb_selected_normals import inspect_png
from scripts.calibrate_mpi_pcb_normals import MODEL_ORDER, REGIONS, EXPECTED_MODEL_PREFLIGHT_SHA256
from scripts.preflight_mpi_pcb_models import (
    EXPECTED_ARCHIVE_SHA256, EXPECTED_IDS_SHA256, EXPECTED_V4_SHA256,
    normal_threshold, photo_score, sha256_file,
)


EXPECTED_CALIBRATION_SHA256 = "6fb949ae182453fc703770dba30342a644af4641eeb69ec57c89d10250019cbd"
EXPECTED_CONFIG = {
    "schema_version": 1,
    "status": "prospective_external_stress_test_after_normal_adaptation",
    "calibration_report_sha256": EXPECTED_CALIBRATION_SHA256,
    "normal_protocol_config_sha256": "0fe96e7985ab39ec7833da9c4624fbc720faae464bf9b780567d25c8b005255c",
    "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
    "test_original_photos_per_class": 55, "regions_per_photo": 12, "image_size": 256,
    "calibration_replay_absolute_tolerance": 0.0001,
    "decision_rule": "score >= frozen_model_threshold",
    "photo_score": "maximum_of_12_region_top_1pct_map_means",
    "duplicate_blocking_unit": "all_12_regions_identical_decoded_RGB",
    "partial_tile_duplicates_report_only": True,
    "mask_rule": "native_256_grayscale_nonzero_no_resizing",
    "empty_region_mask_allowed": True, "empty_whole_defect_mask_allowed": False,
    "audit_failure_action": "stop_all_test_scoring_no_exclusions",
    "primary_metrics": ["escape_rate", "overkill_rate"],
    "secondary_metrics": ["image_auroc", "image_average_precision", "precision", "recall", "f1", "review_rate"],
    "pixel_metrics_enabled": False, "confidence_level": 0.95,
    "paired_bootstrap_replicates": 5000, "paired_bootstrap_seed": 20260926,
    "post_test_tuning_allowed": False, "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
    "thresholds": {"realiad_pcb": 67.42196179599297, "visa_pcb3_v2": 66.18967593588481},
}


def load_evaluation_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise ValueError("MPI-PCB external-test contract differs from locked values")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def validate_calibration(path: Path, v4: dict, protocol: dict, config: dict) -> dict:
    if sha256_file(path) != config["calibration_report_sha256"]:
        raise ValueError("Reviewed normal calibration JSON checksum differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("status") == "mpi_pcb_normal_calibrated_test_sealed"
        and report.get("git", {}).get("commit") == "285f9aecabc17049c9aaa00cddcb2e1696a01759"
        and report.get("protocol_config_sha256") == protocol["config_sha256"] == config["normal_protocol_config_sha256"]
        and report.get("model_preflight_sha256") == EXPECTED_MODEL_PREFLIGHT_SHA256
        and report.get("normal_preflight_v4_sha256") == EXPECTED_V4_SHA256
        and report.get("normal_photo_ids_sha256") == EXPECTED_IDS_SHA256
        and report.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and report.get("normal_original_photos_scored_per_model") == 169
        and report.get("normal_region_images_scored_per_model") == 2028
        and report.get("photo_aggregation") == "maximum_of_12_region_scores"
        and report.get("region_score") == "mean_of_top_1pct_anomaly_map"
        and all(report.get(field) == 0 for field in ("test_images_decoded", "test_masks_decoded", "test_scores_computed"))
        and report.get("automatic_pass_allowed") is False
        and report.get("confirmatory_claim_allowed") is False
        and set(report.get("models", {})) == set(MODEL_ORDER)
    ):
        raise ValueError("Normal calibration provenance differs from the reviewed sealed result")
    for key in MODEL_ORDER:
        model = report["models"][key]
        photos = model["photo_scores"]
        if [photo["photo_id"] for photo in photos] != v4["calibration_photo_ids"]:
            raise ValueError("Normal calibration photo IDs/order changed")
        for photo in photos:
            if [row["region"] for row in photo["regions"]] != REGIONS:
                raise ValueError("Normal calibration regions changed")
            for row in photo["regions"]:
                if row["member"] != f'extracted/{row["region"]}/train/good/{photo["photo_id"]}.png':
                    raise ValueError("Calibration member is not a selected training normal")
            if photo_score([row["score"] for row in photo["regions"]]) != photo["score"]:
                raise ValueError("Normal photo score is not the 12-region maximum")
        threshold = normal_threshold([photo["score"] for photo in photos])
        if not (
            threshold == model["threshold"] == config["thresholds"][key]
            and model["checkpoint_sha256"] == protocol["models"][key]["checkpoint_sha256"]
            and model["normal_original_photos_scored"] == 169
            and model["normal_region_images_scored"] == 2028
            and model["normal_photos_at_or_above_threshold"] == sum(photo["score"] >= threshold for photo in photos)
        ):
            raise ValueError("Frozen calibration threshold/count/checkpoint mismatch")
    return report


def test_inventory(infos: list[zipfile.ZipInfo]) -> list[dict]:
    """Central-directory reads only. No test bytes before gates/replay pass."""
    audit = audit_members(infos)
    if audit["status"] != "registered_metadata_only" or [r["region"] for r in audit["regions"]] != REGIONS:
        raise ValueError("MPI-PCB test inventory changed")
    indexed = {}
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        if len(parts) == 5 and parts[2:4] in (("test", "good"), ("test", "defect"), ("ground_truth", "defect")):
            indexed[(parts[1], "/".join(parts[2:4]), PurePosixPath(info.filename).stem)] = info.filename
    photos = []
    for group, label in (("good", 0), ("defect", 1)):
        ids = sorted(key[2] for key in indexed if key[:2] == (REGIONS[0], f"test/{group}"))
        if len(ids) != 55:
            raise ValueError("Expected 55 original photos per class")
        for photo_id in ids:
            photos.append({"key": f"{group}/{photo_id}", "photo_id": photo_id, "label": label, "regions": [{
                "region": region, "member": indexed[(region, f"test/{group}", photo_id)],
                "mask_member": indexed[(region, "ground_truth/defect", photo_id)] if label else None,
            } for region in REGIONS]})
    return photos


def decode_image(raw: bytes) -> Image.Image:
    details = inspect_png(raw)
    if "verify_error" in details or "load_error" in details:
        raise ValueError(f"PNG integrity failed: {details}")
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(f"Expected RGB 256x256 test PNG, got {image.mode} {image.size}")
        image.load()
        return image.copy()


def mask_details(raw: bytes) -> dict:
    """Integrity only: native nonzero grayscale; an individual tile may be empty."""
    details = inspect_png(raw)
    if "verify_error" in details or "load_error" in details:
        raise ValueError(f"Mask PNG integrity failed: {details}")
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (256, 256) or image.mode not in ("1", "L", "P", "RGB", "RGBA"):
            raise ValueError(f"Unsupported mask mode/size: {image.mode} {image.size}")
        rgba = np.asarray(image.convert("RGBA"))
        if not np.all(rgba[:, :, 3] == 255) or not np.array_equal(rgba[:, :, 0], rgba[:, :, 1]) or not np.array_equal(rgba[:, :, 0], rgba[:, :, 2]):
            raise ValueError("Mask must be opaque grayscale; colored/transparent semantics need review")
        values, counts = np.unique(rgba[:, :, 0], return_counts=True)
        return {"mode": image.mode, "nonzero_pixels": int(np.count_nonzero(rgba[:, :, 0])),
                "value_counts": {str(int(v)): int(c) for v, c in zip(values, counts)}}


def duplicate_summary(photos: list[dict], calibration_rows: list[dict]) -> dict:
    calibration = defaultdict(dict)
    cal_tiles = defaultdict(set)
    for row in calibration_rows:
        calibration[row["photo_id"]][row["region"]] = row["decoded_rgb_sha256"]
        cal_tiles[row["region"]].add(row["decoded_rgb_sha256"])
    cal_signatures = {tuple(rows[r] for r in REGIONS): photo_id for photo_id, rows in calibration.items()}
    seen = defaultdict(list)
    overlap = []
    tile_seen = defaultdict(list)
    partial_cal_overlap = 0
    for photo in photos:
        good_rows = {row["region"]: row for row in photo["regions"] if "decoded_rgb_sha256" in row}
        for region, row in good_rows.items():
            digest = row["decoded_rgb_sha256"]
            tile_seen[(region, digest)].append(photo["key"])
            partial_cal_overlap += int(digest in cal_tiles[region])
        if set(good_rows) == set(REGIONS):
            signature = tuple(good_rows[r]["decoded_rgb_sha256"] for r in REGIONS)
            seen[signature].append(photo["key"])
            if signature in cal_signatures:
                overlap.append({"test_photo": photo["key"], "calibration_photo_id": cal_signatures[signature]})
    return {
        "identical_test_photo_groups": [group for group in seen.values() if len(group) > 1],
        "identical_calibration_test_photos": overlap,
        "repeated_test_tile_groups_report_only": sum(len(group) > 1 for group in tile_seen.values()),
        "calibration_test_tile_matches_report_only": partial_cal_overlap,
        "near_duplicates_checked": False,
    }


def audit_test(source: zipfile.ZipFile, inventory: list[dict], calibration_rows: list[dict]) -> dict:
    issues = []
    photos = []
    counts = {"test_image_read_attempts": 0, "test_images_decoded": 0,
              "test_mask_read_attempts": 0, "test_masks_decoded": 0}
    for index, photo in enumerate(inventory, 1):
        recorded = {key: photo[key] for key in ("key", "photo_id", "label")}
        recorded["regions"] = []
        for entry in photo["regions"]:
            row = dict(entry)
            counts["test_image_read_attempts"] += 1
            try:
                raw = source.read(entry["member"])
                row["member_sha256"] = hashlib.sha256(raw).hexdigest()
                with decode_image(raw) as image:
                    row["decoded_rgb_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                counts["test_images_decoded"] += 1
            except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, KeyError) as exc:
                issues.append({"member": entry["member"], "error": f"{type(exc).__name__}: {exc}"})
            if entry["mask_member"]:
                counts["test_mask_read_attempts"] += 1
                try:
                    raw = source.read(entry["mask_member"])
                    row["mask_sha256"] = hashlib.sha256(raw).hexdigest()
                    row["mask"] = mask_details(raw)
                    counts["test_masks_decoded"] += 1
                except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, KeyError) as exc:
                    issues.append({"member": entry["mask_member"], "error": f"{type(exc).__name__}: {exc}"})
            recorded["regions"].append(row)
        if photo["label"] and all("mask" in row for row in recorded["regions"]):
            if sum(row["mask"]["nonzero_pixels"] for row in recorded["regions"]) == 0:
                issues.append({"photo": photo["key"], "error": "All 12 defect masks are empty"})
        photos.append(recorded)
        if index % 10 == 0 or index == len(inventory):
            print(f"test integrity audit {index}/{len(inventory)} original photos", flush=True)
    duplicates = duplicate_summary(photos, calibration_rows)
    if duplicates["identical_test_photo_groups"] or duplicates["identical_calibration_test_photos"]:
        issues.append({"error": "Exact 12-region photo duplicate: test/test or calibration/test"})
    return {"status": "test_audit_blocked_no_scores" if issues else "test_audit_passed_no_scores",
            **counts, "intended_normal_original_photos": 55, "intended_defect_original_photos": 55,
            "issues": issues, "duplicates": duplicates, "photos": photos,
            "test_scores_computed": 0, "automatic_pass_allowed": False}


def compare_replay(previous: list[dict], current: list[dict]) -> dict:
    if [p["photo_id"] for p in previous] != [p["photo_id"] for p in current]:
        raise ValueError("Normal replay photo IDs changed")
    region_deltas, photo_deltas = [], []
    for old, new in zip(previous, current, strict=True):
        if [(r["region"], r["member"]) for r in old["regions"]] != [(r["region"], r["member"]) for r in new["regions"]]:
            raise ValueError("Normal replay region identities changed")
        photo_deltas.append(abs(old["score"] - new["score"]))
        region_deltas.extend(abs(a["score"] - b["score"]) for a, b in zip(old["regions"], new["regions"], strict=True))
    if not region_deltas or not all(math.isfinite(v) for v in region_deltas + photo_deltas):
        raise ValueError("Normal replay scores are missing or nonfinite")
    return {"maximum_absolute_region_score_difference": max(region_deltas),
            "maximum_absolute_photo_score_difference": max(photo_deltas),
            "normal_original_photos": len(current), "normal_region_scores": len(region_deltas)}


def model_metrics(photos: list[dict], threshold: float) -> dict:
    labels = np.array([photo["label"] for photo in photos], dtype=bool)
    scores = np.array([photo["score"] for photo in photos], dtype=np.float64)
    counts = confusion_counts(labels, scores, threshold)
    return {"image_auroc": binary_auroc(labels, scores),
            "image_average_precision": binary_average_precision(labels, scores),
            "fixed_threshold": {"threshold": threshold, **_fixed_image_metrics(counts, 0.95)}}


def paired_comparison(results: dict, config: dict) -> dict:
    old, new = [results[key]["photos"] for key in MODEL_ORDER]
    if [(p["key"], p["label"]) for p in old] != [(p["key"], p["label"]) for p in new]:
        raise ValueError("Paired test photos differ between models")
    comparison = {}
    rng = np.random.default_rng(config["paired_bootstrap_seed"])
    for label, title in ((0, "normal"), (1, "defect")):
        indices = [i for i, p in enumerate(old) if p["label"] == label]
        a = np.array([old[i]["score"] >= config["thresholds"][MODEL_ORDER[0]] for i in indices])
        b = np.array([new[i]["score"] >= config["thresholds"][MODEL_ORDER[1]] for i in indices])
        errors_a = ~a if label else a
        errors_b = ~b if label else b
        differences = errors_b.astype(float) - errors_a.astype(float)
        samples = rng.integers(0, len(indices), size=(config["paired_bootstrap_replicates"], len(indices)))
        bootstrapped = differences[samples].mean(axis=1)
        comparison[title] = {
            "count": len(indices), "both_review": int(np.sum(a & b)),
            "realiad_only_review": int(np.sum(a & ~b)), "visa_only_review": int(np.sum(~a & b)),
            "neither_review": int(np.sum(~a & ~b)),
            "error_rate_delta_visa_minus_realiad": float(differences.mean()),
            "paired_photo_bootstrap_95pct_interval": np.quantile(bootstrapped, [0.025, 0.975], method="linear").tolist(),
        }
    comparison["limitation"] = "Descriptive photo-level intervals only; repeated views of one board are not independent boards or production defects."
    return comparison
