"""Phase 2: frozen 300-epoch normal-only fit and native-logit calibration."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import zipfile

import numpy as np
from scripts import mpi_supersimplenet_protocol as probe
from scripts.mpi_efficientad_fit_protocol import DEVELOPMENT_SHA, validate_cohorts
from scripts import fit_mpi_pcb_normal_bank as fit

NAME = "mpi-supersimplenet-fit-v1"
PROBE_SHA = "432b709ec363d86ef21dbd9ff978f713f796f2eae6963c6bafc2e04d0fb3d707"
POLICY = deepcopy(probe.POLICY)
POLICY.update({
    "purpose": "formal_normal_only_research_fit_not_service", "model": "SuperSimpleNet",
    "normal_photos": 64, "tiles": 768, "epochs": 300, "updates_per_epoch": 96,
    "optimizer_updates": 28800, "updates": 28800, "checkpoint_interval": 960,
    "maximum_attempted_updates": 29760, "selected_checkpoint": "epoch300_step28800_only",
    "lr_schedule": "after_completed_epochs240_and270_multiply_both_lrs_by0.4",
    "shuffle": "SHA256(name,NUL,seed,NUL,epoch,NUL,row_index)_ascending_each_epoch",
    "development_photos": 128, "calibration_photos": 169,
    "region_scale": "development_median_and_Q90_higher_minus_median",
    "photo_score": "max_of_12_region_standardized_native_logits",
    "threshold": "calibration_Q95_higher", "alert_operator": ">=",
    "minimum_width_exclusive": 1e-6, "reload_tolerance": 1e-4,
    "map_normalization": "none_raw_signed_map_preserved_not_used_for_image_decision",
    "resource_plan": {"comparison_seconds": 14400, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "free_disk_GiB": 20, "full_training_authorized": True,
    "test_images_decoded": 0, "test_masks_decoded": 0, "service_model_changed": False,
    "future_evaluation": {"first_gate": "fixed_prior12_all_retained", "executed_by_this_runner": False,
        "minimum_tp55": 53, "lost_baseline_detections_allowed": 0,
        "maximum_fp54": 2, "maximum_fp256": 12, "warm_photo_p95_seconds": 2,
        "independent_final_available": False},
})


def read_probe(path):
    if fit.sha256_file(path) != PROBE_SHA:
        raise ValueError("Expected reviewed successful SuperSimpleNet probe ZIP")
    with zipfile.ZipFile(path) as z:
        report = json.loads(z.read("report.json"))
        if report["status"] != "normal_only_resource_probe_completed" or report["completed_updates"] != 100:
            raise ValueError("Reviewed probe did not complete")
        return report


def prepare(project, handoffs):
    cfg = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    original = fit.load_preparation(handoffs / "mpi-pcb-normal-bank-preparation-v1.zip", cfg)
    source = handoffs / "mpi-pcb-score-probe-v1.zip"
    if fit.sha256_file(source) != DEVELOPMENT_SHA:
        raise ValueError("Normal development source ZIP differs")
    with zipfile.ZipFile(source) as z:
        dev = json.loads(z.read("plan.json"))["development"]  # only normal manifest, no test scores
    groups = {g: {"photo_ids": original[g+"_photo_ids"], "rows": original[g+"_rows"]}
              for g in ("training", "calibration")}
    groups["development"] = {k: dev[k] for k in ("photo_ids", "rows")}
    validate_cohorts(groups)
    report = read_probe(handoffs / "mpi-supersimplenet-probe-v1.zip")
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    return {"name": NAME, "policy": POLICY, "cohorts": groups,
        "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256, "preparation_sha256": fit.PREPARATION_SHA256,
        "development_source_sha256": DEVELOPMENT_SHA, "probe_sha256": PROBE_SHA,
        "environment": report["environment"], "weights": probe.WEIGHTS,
        "expected_fresh_fingerprints": report["checks"]["initial_fingerprints"],
        "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in files}}


@lru_cache(maxsize=4)
def epoch_order(epoch):
    return tuple(sorted(range(768), key=lambda i: hashlib.sha256(
        f"{NAME}\0{42}\0{epoch}\0{i}".encode()).digest()))


def batch_indices(index):
    if not 0 <= index < 28800:
        raise ValueError("Formal fit index outside 300 epochs")
    epoch, batch = divmod(index, 96)
    return epoch_order(epoch)[batch*8:(batch+1)*8]


def learning_rates(index):
    if not 0 <= index < 28800:
        raise ValueError("Learning rate index outside formal fit")
    drops = int(index >= 240*96) + int(index >= 270*96)
    return [rate * .4**drops for rate in (.0001, .0002)]


def validate_predictions(values):
    a = np.asarray(values)
    if a.ndim != 2 or a.shape[1] != 65537 or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError("Expected float32 [tiles, native_logit+65536_raw_map_pixels]")
    return a


def tile_scores(values):
    return validate_predictions(values)[:, 0].astype(np.float64)


def calibrate(development, calibration):
    dev, cal = np.asarray(development, dtype=np.float64), np.asarray(calibration, dtype=np.float64)
    if dev.shape != (128, 12) or cal.shape != (169, 12) or not np.isfinite(dev).all() or not np.isfinite(cal).all():
        raise ValueError("Incomplete/nonfinite normal calibration matrix")
    center = np.median(dev, axis=0)
    scale = np.quantile(dev, .9, axis=0, method="higher") - center
    if not np.isfinite(scale).all() or (scale <= 1e-6).any():
        raise ValueError("Degenerate region Q90 width; no fallback")
    scores = ((cal-center)/scale).max(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite photo scores")
    return {"centers": center.tolist(), "scales": scale.tolist(),
        "threshold": float(np.quantile(scores, .95, method="higher")), "region_order": fit.REGIONS,
        "operator": ">=", "tile_score": "native_classification_logit",
        "normalization_clipped": False, "test_scores_used": 0, "map_normalization": "none"}


def photo_scores(tiles, contract):
    values = np.asarray(tiles, dtype=np.float64)
    centers, scales = np.asarray(contract["centers"]), np.asarray(contract["scales"])
    if (values.shape[-1:] != (12,) or centers.shape != (12,) or scales.shape != (12,)
            or not np.isfinite(values).all() or not np.isfinite(centers).all()
            or not np.isfinite(scales).all() or (scales <= 1e-6).any()):
        raise ValueError("Invalid scoring contract or tile scores")
    result = ((values-centers)/scales).max(axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite standardized photo score")
    return result
