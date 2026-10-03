"""Frozen normal-only EfficientAD fit contract; CPU-only input and score checks."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import zipfile

import numpy as np
from scripts import fit_mpi_pcb_normal_bank as fit

NAME = "mpi-efficientad-fit-v1"
PROBE_SHA = "12ec720e2e12efa1bb5e17edf802f6b68651454f5e2c061535de477d952bd98a"
DEVELOPMENT_SHA = "5784bf5d6cd24754788088d8e2f573a6804808934a3d7f6117b5ad75f4275795"
POLICY = {
    "seed": 42, "model": "EfficientAD-S", "batch_size": 1, "image_size": 256,
    "teacher_channels": 384, "padding": False, "pad_maps": True,
    "updates": 70000, "maximum_attempted_updates": 72000,
    "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-5,
    "lr_drop_after_completed_updates": 66500, "final_lr": 1e-5,
    "checkpoint_interval": 1000, "selected_checkpoint": "final_70000_only",
    "training_photos": 64, "development_photos": 128, "calibration_photos": 169,
    "auxiliary": "all_9469_imagenette2_train_jpegs_no_val",
    "shuffle": "independent_SHA256_sorted_per_epoch_for_train_and_aux",
    "normal_monitor": "first_training_photo_all_12_regions_each_checkpoint_raw_maps",
    "map_quantiles": [.9, .995], "map_quantile_method": "linear_all_development_pixels",
    "map_formula": "0.5*(0.1*(ST-q90_ST)/width_ST+0.1*(AE-q90_AE)/width_AE)",
    "negative_normalized_map_policy": "preserve_signed_no_clipping",
    "tile_score": "float64_mean_of_highest_656_of_65536_pixels",
    "region_scale": "development_median_and_Q90_higher_minus_median",
    "photo_score": "max_of_12_region_standardized_scores",
    "threshold": "calibration_Q95_higher", "alert_operator": ">=",
    "minimum_width_exclusive": 1e-6, "reload_tolerance": 1e-4,
    "resource_plan": {"comparison_seconds": 14400, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "free_disk_GiB": 30, "free_gpu_GiB": 6,
    "test_images_decoded": 0, "test_masks_decoded": 0,
    "automatic_pass_allowed": False, "service_model_changed": False,
    "future_evaluation": {
        "first_gate": "same_frozen_12_prior_TP_all_retained_before_full_comparison",
        "regression_keys": ["defect/0006", "defect/0025", "defect/0048", "defect/0000",
            "defect/0054", "defect/0051", "defect/0013", "defect/0033", "defect/0038",
            "defect/0001", "defect/0034", "defect/0042"],
        "minimum_tp55": 53, "lost_baseline_detections_allowed": 0,
        "maximum_fp54": 2, "maximum_fp256": 12,
        "miss_improvement_claim_requires_old_FN_recovered": True,
        "warm_photo_p95_seconds": 2,
        "later_evaluation_compute_cap_seconds": 14400,
        "stability": {"scopes": ["development_only", "calibration_only", "both"],
            "repetitions_per_scope": 200, "PCG64_seeds": [20261004, 20261005, 20261006],
            "paired_photo_indices_with_baseline": True,
            "refit": "ST/AE map quantiles AND region Q90 AND photo threshold; whole-photo resampling",
            "minimum_tp_q025": 50, "maximum_tp_interval_width_over_baseline": .5,
            "maximum_fp54_q975": 5, "maximum_fp256_q975": 25,
            "invalid_repetitions_allowed": 0, "interval_quantile_method": "linear"},
        "executed_by_this_runner": False, "independent_final_available": False},
}


def read_probe(path):
    if fit.sha256_file(path) != PROBE_SHA:
        raise ValueError("Expected reviewed successful EfficientAD probe ZIP")
    with zipfile.ZipFile(path) as z:
        report = json.loads(z.read("report.json"))
        if report["status"] != "normal_only_resource_probe_completed":
            raise ValueError("Probe did not complete")
        return report["environment"]


def validate_cohorts(groups):
    ids, pixels = set(), set()
    for label, count in (("training", 64), ("development", 128), ("calibration", 169)):
        cohort = groups[label]
        names, rows = cohort["photo_ids"], cohort["rows"]
        if len(names) != count or len(set(names)) != count or ids.intersection(names):
            raise ValueError("Normal photo counts or disjointness differ")
        ids.update(names)
        if [(r["photo_id"], r["region"]) for r in rows] != [(p, r) for p in names for r in fit.REGIONS]:
            raise ValueError("Normal row order differs")
        for row in rows:
            pid, region = row["photo_id"], row["region"]
            if (not pid.isascii() or len(pid) != 4 or not pid.isdigit()
                    or row["member"] != f"extracted/{region}/train/good/{pid}.png"
                    or row["decoded_rgb_sha256"] in pixels):
                raise ValueError("Non-normal path or duplicate normal pixels")
            pixels.add(row["decoded_rgb_sha256"])


def prepare(project, handoffs):
    cfg = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    original = fit.load_preparation(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", cfg)
    source = handoffs / "mpi-pcb-score-probe-v1.zip"
    if fit.sha256_file(source) != DEVELOPMENT_SHA:
        raise ValueError("Normal development source ZIP differs")
    with zipfile.ZipFile(source) as z:
        # Read only the existing normal manifest, no cards, test scores or labels.
        dev = json.loads(z.read("plan.json"))["development"]
    groups = {g: {"photo_ids": original[g + "_photo_ids"], "rows": original[g + "_rows"]}
              for g in ("training", "calibration")}
    groups["development"] = {k: dev[k] for k in ("photo_ids", "rows")}
    validate_cohorts(groups)
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    return {"name": NAME, "policy": POLICY, "cohorts": groups,
            "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
            "preparation_sha256": fit.PREPARATION_SHA256, "development_source_sha256": DEVELOPMENT_SHA,
            "probe_sha256": PROBE_SHA, "environment": read_probe(handoffs / "mpi-efficientad-probe-v1.zip"),
            "sources_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}}


@lru_cache(maxsize=8)
def epoch_order(size, epoch, role):
    return tuple(sorted(range(size), key=lambda i: hashlib.sha256(
        f"{NAME}\0{POLICY['seed']}\0{role}\0{epoch}\0{i}".encode()).digest()))


def sample_index(step, size, role):
    return epoch_order(size, step // size, role)[step % size]


def learning_rate(index):
    return POLICY["lr"] if index < POLICY["lr_drop_after_completed_updates"] else POLICY["final_lr"]


def validate_maps(values):
    a = np.asarray(values)
    if a.shape[-3:] != (2, 256, 256) or a.dtype != np.float32 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected finite nonnegative float32 raw ST/AE maps")
    return a


def map_quantiles(development, check=lambda: None):
    # Exact all-pixel CPU quantiles, one path at a time; no random pixel reduction.
    result = []
    for channel in range(2):
        check()
        arrays = [validate_maps(a)[:, channel].reshape(-1) for a in development]
        flat = np.concatenate(arrays)
        qa, qb = np.quantile(flat, [.9, .995], method="linear")
        if not np.isfinite([qa, qb]).all() or qb - qa <= POLICY["minimum_width_exclusive"]:
            raise ValueError("Degenerate EfficientAD map quantiles; no fallback")
        result.append({"q90": float(qa), "q995": float(qb)})
        del flat, arrays
        check()
    return result


def tile_scores(raw, quantiles):
    raw = validate_maps(raw).astype(np.float64)
    normalized = []
    for channel, q in enumerate(quantiles):
        width = q["q995"] - q["q90"]
        if not np.isfinite(width) or width <= POLICY["minimum_width_exclusive"]:
            raise ValueError("Invalid map normalization width")
        normalized.append(.1 * (raw[:, channel] - q["q90"]) / width)
    combined = .5 * (normalized[0] + normalized[1])
    flat = combined.reshape(len(raw), -1)
    scores = np.partition(flat, flat.shape[1] - 656, axis=1)[:, -656:].mean(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite normalized score")
    return scores


def calibrate(development, calibration, quantiles):
    dev, cal = np.asarray(development), np.asarray(calibration)
    if dev.shape != (128, 12) or cal.shape != (169, 12) or not np.isfinite(dev).all() or not np.isfinite(cal).all():
        raise ValueError("Incomplete normal calibration matrix")
    center = np.median(dev, axis=0)
    scale = np.quantile(dev, .9, axis=0, method="higher") - center
    if (scale <= POLICY["minimum_width_exclusive"]).any():
        raise ValueError("Degenerate region Q90 width; no fallback")
    scores = ((cal - center) / scale).max(axis=1)
    return {"map_quantiles": quantiles, "centers": center.tolist(), "scales": scale.tolist(),
            "threshold": float(np.quantile(scores, .95, method="higher")),
            "region_order": fit.REGIONS, "operator": ">=", "normalization_clipped": False,
            "development_used_for_map_and_region_calibration": True, "test_scores_used": 0}


def photo_scores(tiles, contract):
    return ((np.asarray(tiles) - contract["centers"]) / contract["scales"]).max(axis=-1)
