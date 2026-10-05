"""Fixed normal-only functional check, not a new fit or performance evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import mpi_supersimplenet_evaluation_protocol as e
from defectops.evaluation_metrics import binary_auroc

fit, p = e.fit, e.p
NAME = "mpi-supersimplenet-synthetic-check-v1"
POLICY = {
    "photos_per_group": 8, "groups": ["training", "development"], "batch_size": 8,
    "seeds": [42, 43], "selection": "first8_photos_in_frozen_fit_manifest; all12_regions; contiguous_batch8",
    "modes": ["train_mixed", "eval_mixed", "eval_clean"],
    "synthetic": "unchanged_official_generator_v2.2.0; noise_std0.015; perlin0.6; no_reroll",
    "same_mixed_features_across_modes": True, "native_map_size": 64,
    "normal_replay_tolerance": 1e-4, "synthetic_zero_logit_is_not_production_threshold": True,
    "resource_plan": {"comparison_seconds": 900, "rss_GiB": 20, "selection_and_inference_reserved_GiB": 6},
    "free_disk_GiB": 2, "maximum_attempted_trials": 50,
    "optimizer_updates": 0, "test_images_decoded": 0, "test_masks_decoded": 0,
    "threshold_changed": False, "service_model_changed": False, "automatic_pass_allowed": False,
    "interpretation": "synthetic functional diagnosis only; no automatic selection or accuracy claim",
}
FILES = ["arrays.npz", "result.json"]


def jobs(plan):
    result = []
    for group in POLICY["groups"]:
        cohort = plan["cohorts"][group]
        ids = cohort["photo_ids"][:8]
        rows = cohort["rows"][:96]
        if len(set(ids)) != 8 or len(rows) != 96:
            raise ValueError("Expected eight fixed normal photos in each group")
        if [(r["photo_id"], r["region"]) for r in rows] != [(pid, region) for pid in ids for region in fit.REGIONS]:
            raise ValueError("Normal row order differs")
        for batch in range(12):
            for seed in POLICY["seeds"]:
                result.append({"name": f"{group}-batch{batch:02d}-seed{seed}", "group": group,
                    "row_start": batch*8, "seed": seed, "rows": rows[batch*8:batch*8+8]})
    return result


def prepare(project, handoffs, fit_root):
    records = e.read_fit(project, handoffs, fit_root)
    plan = records["plan.json"]
    paths = ["scripts/mpi_supersimplenet_synthetic_protocol.py", "scripts/mpi_supersimplenet_synthetic_runtime.py",
             "scripts/run_mpi_supersimplenet_synthetic_check.py", "scripts/run_mpi_supersimplenet_synthetic_check.ps1",
             "docs/mpi-supersimplenet-synthetic-check-v1.md"]
    identity = {"name": NAME, "policy": POLICY, "fit_zip_sha256": e.FIT_SHA,
        "fit_signature": fit.digest(plan), "checkpoint_sha256": records["checkpoint-receipts.json"][-1]["files"]["model.pt"],
        "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256, "environment": plan["environment"],
        "contract": records["normal-results.json"]["contract"], "jobs": jobs(plan),
        "sources_lf_sha256": {n: hashlib.sha256((project/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in paths}}
    return records, identity


def summarize_arrays(arrays):
    expected = {"masks", "labels"} | {m+suffix for m in POLICY["modes"] for suffix in ("_maps", "_scores")}
    if set(arrays) != expected:
        raise ValueError("Synthetic array inventory differs")
    masks, labels = arrays["masks"], arrays["labels"]
    if (masks.shape != (16,64,64) or masks.dtype != np.bool_ or labels.shape != (16,)
            or labels.dtype != np.bool_ or not np.array_equal(labels, masks.reshape(16,-1).any(1))):
        raise ValueError("Synthetic masks/labels differ")
    result = {"positive_samples": int(labels.sum()), "normal_samples": int((~labels).sum()),
              "anomalous_pixel_fraction": float(masks.mean()), "modes": {}}
    for mode in POLICY["modes"]:
        maps, scores = arrays[mode+"_maps"], arrays[mode+"_scores"]
        if (maps.shape != masks.shape or scores.shape != labels.shape or maps.dtype != np.float32
                or scores.dtype != np.float32 or not np.isfinite(maps).all() or not np.isfinite(scores).all()):
            raise ValueError("Expected finite float32 native maps and scores")
        metrics = {"scores": scores.tolist(), "minimum_map": float(maps.min()), "maximum_map": float(maps.max())}
        if mode != "eval_clean":
            metrics.update({"classification_auroc": binary_auroc(labels, scores),
                "pixel_auroc": binary_auroc(masks, maps),
                "synthetic_score_positive_fraction": float(np.mean(scores[labels] > 0)) if labels.any() else None,
                "normal_score_positive_fraction": float(np.mean(scores[~labels] > 0)) if (~labels).any() else None,
                "gt_map_mean": float(maps[masks].mean()) if masks.any() else None,
                "background_map_mean": float(maps[~masks].mean()) if (~masks).any() else None})
        result["modes"][mode] = metrics
    delta = arrays["eval_mixed_scores"] - arrays["eval_clean_scores"]
    result["eval_paired_score_delta"] = delta.tolist()
    result["eval_synthetic_delta_mean"] = float(delta[labels].mean()) if labels.any() else None
    result["mode_score_max_abs_difference"] = float(np.max(np.abs(arrays["train_mixed_scores"]-arrays["eval_mixed_scores"])))
    return result


def read_trial(root, job, signature):
    receipt = fit.read_phase(root, job["name"], signature, FILES)
    if receipt["job_sha256"] != fit.digest(job):
        raise ValueError("Synthetic trial input changed")
    with np.load(root/job["name"]/"arrays.npz", allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    result = json.loads((root/job["name"]/"result.json").read_bytes())
    if result["metrics"] != summarize_arrays(arrays) or not result["checks"]["state_unchanged"]:
        raise ValueError("Synthetic result cannot be reproduced or state was changed")
    return result


def aggregate(results, jobs_):
    report = {}
    for group in POLICY["groups"]:
        selected = [r for r,j in zip(results,jobs_,strict=True) if j["group"] == group]
        report[group] = {"trials": len(selected), "positive_presentations": sum(r["metrics"]["positive_samples"] for r in selected),
                         "modes": {}}
        for mode in ("train_mixed", "eval_mixed"):
            report[group]["modes"][mode] = {}
            for metric in ("classification_auroc", "pixel_auroc", "synthetic_score_positive_fraction", "normal_score_positive_fraction"):
                values = [r["metrics"]["modes"][mode][metric] for r in selected if r["metrics"]["modes"][mode][metric] is not None]
                report[group]["modes"][mode][metric] = {"mean_over_defined_trials": float(np.mean(values)) if values else None,
                                                        "defined_trials": len(values)}
        delta = [r["metrics"]["eval_synthetic_delta_mean"] for r in selected if r["metrics"]["eval_synthetic_delta_mean"] is not None]
        report[group]["mean_trial_synthetic_score_delta"] = float(np.mean(delta)) if delta else None
    return report
