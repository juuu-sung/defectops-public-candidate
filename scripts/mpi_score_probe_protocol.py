"""Frozen, exploratory scoring ablations; no training or service decisions."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

from scripts import evaluate_mpi_pcb_normal_bank as ev

fit = ev.fit
NAME = "mpi-pcb-score-probe-v1"
EVALUATION_SHA256 = "070a15bfed987802201868dee759fc5c24445bf12c6c6d3643feb3b25798e857"
SCORERS = ("baseline_top1", "a_region_mad_top1", "b_native_raw")
POLICY = {
    "schema_version": 1, "normal_development_photos": 128,
    "selection_salt": NAME, "normalization": "(tile_top1 - region_median) / region_MAD",
    "mad_definition": "median(abs(x-median(x))), unscaled",
    "minimum_mad_exclusive": 1e-6, "degenerate_scale_policy": "stop_no_fallback",
    "photo_score": "max_of_all_12_regions", "calibration_quantile": .95,
    "quantile_method": "higher", "ab_combination": False,
    "baseline_threshold": ev.THRESHOLD, "replay_tolerance": 1e-4,
    "normal_cards": "top_10_plus_middle_3_baseline_calibration_scores",
    "test_cards": "lowest_middle_highest_3_baseline_FN_all_TP_FP_middle_3_TN",
    "test_previously_observed": True, "automatic_pass_allowed": False,
    "confirmatory_claim_allowed": False, "training_performed": False,
    "service_model_changed": False,
}


def select_development(rows, preparation, audit_photos, count=128):
    """Selection uses identities/pixels only, never test scores or masks."""
    groups = defaultdict(dict)
    for row in rows:
        p, r = row["photo_id"], row["region"]
        if r not in fit.REGIONS or r in groups[p]:
            raise ValueError("Duplicate/unknown fingerprint region")
        if row["member"] != f"extracted/{r}/train/good/{p}.png":
            raise ValueError("Non-normal development fingerprint")
        groups[p][r] = row
    if any(set(g) != set(fit.REGIONS) for g in groups.values()):
        raise ValueError("Incomplete development candidate photo")
    excluded_ids = set(preparation["training_photo_ids"] + preparation["calibration_photo_ids"])
    blocked_rgb = {r["decoded_rgb_sha256"] for r in preparation["training_rows"] + preparation["calibration_rows"]}
    blocked_rgb.update(r["decoded_rgb_sha256"] for p in audit_photos for r in p["regions"])
    order = sorted(groups, key=lambda p: (hashlib.sha256(f"{NAME}\0{p}".encode()).hexdigest(), p))
    selected, rejected, seen_rgb = [], [], set()
    for p in order:
        pixels = {groups[p][r]["decoded_rgb_sha256"] for r in fit.REGIONS}
        if p in excluded_ids:
            reason = "bank_fit_or_calibration_photo"
        elif pixels & blocked_rgb:
            reason = "exact_tile_overlap_with_fit_calibration_or_reviewed_test"
        elif pixels & seen_rgb:
            reason = "exact_tile_overlap_with_selected_development"
        else:
            selected.append(p)
            seen_rgb.update(pixels)
            if len(selected) == count:
                break
            continue
        rejected.append({"photo_id": p, "reason": reason})
    if len(selected) != count:
        raise ValueError("Insufficient disjoint normal development photos; no automatic reduction")
    chosen = [groups[p][r] for p in selected for r in fit.REGIONS]
    return {"photo_ids": selected, "photo_ids_sha256": fit.digest(selected), "rows": chosen,
            "rejected_before_selection_complete": rejected,
            "scope": "Exact tile RGB separation, not independence of boards/sessions or near-duplicates"}


def validate_sample(sample, photo):
    if any(sample[k] != photo[k] for k in ("key", "photo_id", "label")):
        raise ValueError("Sample identity differs")
    if [(r["region"], r["member"]) for r in sample["regions"]] != [(r["region"], r["member"]) for r in photo["regions"]]:
        raise ValueError("Sample region inventory differs")
    for row in sample["regions"]:
        if any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) or row[k] < 0
               for k in ("top1", "native_raw")):
            raise ValueError("Invalid score")


def replay(sample, expected):
    rows = sample["regions"]
    if [(r["region"], r["member"]) for r in rows] != [(r["region"], r["member"]) for r in expected["regions"]]:
        raise ValueError("Replay region order differs")
    delta = max(abs(r["top1"] - e["score"]) for r, e in zip(rows, expected["regions"], strict=True))
    delta = max(delta, abs(max(r["top1"] for r in rows) - expected["score"]))
    if delta > POLICY["replay_tolerance"]:
        raise ValueError(f"Previously recorded top1 score drift: {delta}")
    if "threshold" in expected and (max(r["top1"] for r in rows) >= expected["threshold"]) != (expected["score"] >= expected["threshold"]):
        raise ValueError("Baseline decision changed despite small replay drift")
    return delta


def fit_normalization(samples):
    if not samples or any(p["label"] != 0 or not p["key"].startswith("development/") for p in samples):
        raise ValueError("Normalization accepts only separate normal development samples")
    result = []
    for index, region in enumerate(fit.REGIONS):
        if any(p["regions"][index]["region"] != region for p in samples):
            raise ValueError("Normalization region order differs")
        values = np.array([p["regions"][index]["top1"] for p in samples], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if not np.isfinite(values).all() or mad <= POLICY["minimum_mad_exclusive"]:
            raise ValueError(f"Degenerate normal development MAD: {region}; no fallback or tuning")
        result.append({"region": region, "median": median, "mad": mad, "normal_photos": len(samples)})
    return result


def score(sample, scorer, normalization):
    if scorer not in SCORERS:
        raise ValueError("Unknown scorer")
    values = []
    for row, stats in zip(sample["regions"], normalization, strict=True):
        if row["region"] != stats["region"]:
            raise ValueError("Region normalization order differs")
        value = ((row["top1"] - stats["median"]) / stats["mad"] if scorer == SCORERS[1]
                 else row["native_raw"] if scorer == SCORERS[2] else row["top1"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite transformed score")
        values.append(float(value))
    return max(values)


def calibration_contract(development, calibration):
    if not calibration or any(p["label"] != 0 or not p["key"].startswith("calibration/") for p in calibration):
        raise ValueError("Threshold calculation accepts only normal calibration photos")
    if {p["photo_id"] for p in development} & {p["photo_id"] for p in calibration}:
        raise ValueError("Development/calibration photo overlap")
    normalization = fit_normalization(development)
    thresholds = {key: float(np.quantile([score(p, key, normalization) for p in calibration], .95, method="higher"))
                  for key in SCORERS}
    if abs(thresholds[SCORERS[0]] - ev.THRESHOLD) > POLICY["replay_tolerance"]:
        raise ValueError("Replayed baseline calibration threshold differs")
    thresholds[SCORERS[0]] = ev.THRESHOLD  # Never replace the published baseline threshold.
    return {"normalization": normalization, "thresholds": thresholds,
            "development_scores_sha256": fit.digest(development), "calibration_scores_sha256": fit.digest(calibration),
            "test_scores_used": 0, "automatic_pass_allowed": False}


def comparison(samples, contract, baseline):
    if [(s["key"], s["label"]) for s in samples] != [(p["key"], p["label"]) for p in baseline]:
        raise ValueError("Comparison test cohort differs")
    photos = {}
    for key in SCORERS:
        threshold = contract["thresholds"][key]
        photos[key] = [{k: s[k] for k in ("key", "photo_id", "label")} | {
            "score": score(s, key, contract["normalization"]), "threshold": threshold,
            "automatic_pass_allowed": False,
            "decision": "REVIEW_REQUIRED" if score(s, key, contract["normalization"]) >= threshold else "BELOW_THRESHOLD_NOT_AUTO_PASS",
        } for s in samples]
    main = {key: ev.model_metrics(ps, contract["thresholds"][key]) for key, ps in photos.items()}
    sensitivity = {key: ev.model_metrics([p for p in ps if p["key"] != "defect/0054"], contract["thresholds"][key])
                   for key, ps in photos.items()}
    return {"models": main, "sensitivity_without_reconstructed_photo": sensitivity,
            "paired_changes_from_published_baseline": {key: ev.paired_changes(baseline, ps) for key, ps in photos.items()},
            "original_photos": len(samples), "defect_photos": sum(p["label"] for p in samples),
            "normal_photos": sum(p["label"] == 0 for p in samples),
            "interpretation": "Exploratory ablations on previously observed tests. No automatic model selection or production claim."}, photos


def middle_three(rows):
    start = max(0, len(rows) // 2 - 1)
    return rows[start:start+3]


def card_selection(calibration, baseline):
    """Freeze representative selection using already reviewed baseline only."""
    cal = sorted(calibration, key=lambda p: (p["score"], p["photo_id"]))
    result = {f"calibration/{p['photo_id']}": "high_normal_calibration" for p in cal[-10:]}
    for p in middle_three(cal):
        result.setdefault(f"calibration/{p['photo_id']}", "middle_normal_control")
    groups = defaultdict(list)
    for p in sorted(baseline, key=lambda p: (p["score"], p["key"])):
        flagged = p["score"] >= p["threshold"]
        groups[("tp" if flagged else "fn") if p["label"] else ("fp" if flagged else "tn")].append(p)
    for group, rows in groups.items():
        chosen = rows if group in ("tp", "fp") else middle_three(rows)
        if group == "fn":
            chosen = rows[:3] + chosen + rows[-3:]
        for p in chosen:
            result[p["key"]] = f"baseline_{group}"
    return result
