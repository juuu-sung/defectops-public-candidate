"""Frozen exploratory scorers and a previously unused normal-only audit cohort."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

from scripts import mpi_score_probe_protocol as old

fit, ev = old.fit, old.ev
NAME = "mpi-score-validation-v1"
PROBE_SHA256 = "5784bf5d6cd24754788088d8e2f573a6804808934a3d7f6117b5ad75f4275795"
RAW_SHA256 = "6471e598103e63c17259110907647cc9f976d8488b2c2c9493322baffa93c9cd"
SCORERS = {"control_top1_mad": ("top1", "mad"),
           "candidate_top1_q90": ("top1", "q90"),
           "candidate_native_mad": ("native_raw", "mad")}
POLICY = {"name": NAME, "normal_validation_count": 256, "selection_salt": NAME,
          "development_count": 128, "calibration_count": 169, "observed_test_count": 109,
          "region_order": list(fit.REGIONS), "photo_aggregation": "max_of_12_regions",
          "normalization_center": "median", "mad": "unscaled_median_absolute_deviation",
          "q90_scale": "quantile_0.90_higher_minus_median", "minimum_scale_exclusive": 1e-6,
          "threshold_quantile": .95, "quantile_method": "higher", "decision_operator": ">=",
          "raw_replay_absolute_tolerance": 1e-4, "replay_decision_changes_allowed": 0,
          "candidate_selection_used_observed_test": True, "training_performed": False,
          "threshold_refit_on_validation": False, "automatic_pass_allowed": False,
          "service_model_changed": False, "confirmatory_claim_allowed": False,
          "normal_validation_scope": "same dataset; exact RGB disjoint, not independent boards/sessions",
          "normal_review_budget": .05, "budget_use": "descriptive triage only; no automatic promotion"}


def validate_samples(samples, prefix=None):
    if not samples or len({s["key"] for s in samples}) != len(samples):
        raise ValueError("Empty/duplicate score cohort")
    for s in samples:
        if prefix and (s["label"] != 0 or s["key"] != f"{prefix}/{s['photo_id']}"):
            raise ValueError("Only the designated normal cohort is accepted")
        if [r["region"] for r in s["regions"]] != list(fit.REGIONS):
            raise ValueError("Region inventory/order differs")
        old.validate_sample(s, s)


def score(sample, method):
    field = method["field"]
    values = []
    for row, stats in zip(sample["regions"], method["regions"], strict=True):
        if row["region"] != stats["region"]:
            raise ValueError("Region order differs")
        value = (row[field] - stats["center"]) / stats["scale"]
        if not math.isfinite(value):
            raise ValueError("Nonfinite normalized score")
        values.append(float(value))
    winner = int(np.argmax(values))
    return {"score": max(values), "alert": max(values) >= method["threshold"],
            "winning_region": method["regions"][winner]["region"]}


def fit_contract(development, calibration):
    validate_samples(development, "development")
    validate_samples(calibration, "calibration")
    if {s["photo_id"] for s in development} & {s["photo_id"] for s in calibration}:
        raise ValueError("Development/calibration overlap")
    methods = {}
    for name, (field, scale_kind) in SCORERS.items():
        x = np.array([[r[field] for r in s["regions"]] for s in development], dtype=np.float64)
        center = np.median(x, axis=0)
        scale = (np.median(np.abs(x-center), axis=0) if scale_kind == "mad"
                 else np.quantile(x, .9, axis=0, method="higher") - center)
        if not np.isfinite(scale).all() or np.any(scale <= POLICY["minimum_scale_exclusive"]):
            raise ValueError("Degenerate scale; no fallback")
        method = {"field": field, "scale_kind": scale_kind, "threshold": 0.,
                  "regions": [{"region": r, "center": float(c), "scale": float(w)}
                              for r, c, w in zip(fit.REGIONS, center, scale, strict=True)]}
        method["threshold"] = float(np.quantile([score(s, method)["score"] for s in calibration], .95, method="higher"))
        methods[name] = method
    return methods


def select_normals(rows, preparation, development, audit, count=256):
    """Select with identities and exact pixels only; never model scores or masks."""
    groups = defaultdict(dict)
    for r in rows:
        pid, region = r["photo_id"], r["region"]
        if region not in fit.REGIONS or region in groups[pid]:
            raise ValueError("Duplicate/unknown fingerprint region")
        if r["member"] != f"extracted/{region}/train/good/{pid}.png":
            raise ValueError("Non-normal fingerprint")
        groups[pid][region] = r
    if any(set(g) != set(fit.REGIONS) for g in groups.values()):
        raise ValueError("Incomplete candidate photo")
    blocked_ids = set(preparation["training_photo_ids"] + preparation["calibration_photo_ids"] + development["photo_ids"])
    blocked_ids.update(p["photo_id"] for p in audit)
    used = preparation["training_rows"] + preparation["calibration_rows"] + development["rows"]
    used += [r for p in audit for r in p["regions"]]
    blocked_rgb = {r["decoded_rgb_sha256"] for r in used}
    chosen, seen, rejected = [], set(), Counter()
    order = sorted(groups, key=lambda pid: (hashlib.sha256(f"{NAME}\0{pid}".encode()).hexdigest(), pid))
    for pid in order:
        rgb = {r["decoded_rgb_sha256"] for r in groups[pid].values()}
        if pid in blocked_ids:
            rejected["used_photo_id"] += 1
        elif rgb & blocked_rgb:
            rejected["used_exact_rgb"] += 1
        elif rgb & seen:
            rejected["selected_exact_rgb"] += 1
        else:
            chosen.append(pid)
            seen.update(rgb)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise ValueError("Insufficient unused normals; no automatic reduction")
    selected = [groups[pid][r] for pid in chosen for r in fit.REGIONS]
    return {"photo_ids": chosen, "rows": selected, "photo_ids_sha256": fit.digest(chosen),
            "rows_sha256": fit.digest(selected), "rejected_before_selection_complete": dict(rejected),
            "scope": POLICY["normal_validation_scope"]}


def replay(sample, expected, contract):
    old.validate_sample(sample, expected)
    delta = max(abs(a[field]-b[field]) for a, b in zip(sample["regions"], expected["regions"], strict=True)
                for field in ("top1", "native_raw"))
    if delta > POLICY["raw_replay_absolute_tolerance"]:
        raise ValueError(f"Raw score replay drift: {delta}")
    # Calibration boundary ties can move by rounding; never hide a decision flip.
    for method in contract.values():
        if score(sample, method)["alert"] != score(expected, method)["alert"]:
            raise ValueError("Frozen candidate/control decision changed")
    return float(delta)


def observed_metrics(samples, contract):
    validate_samples(samples)
    output = {}
    for name, method in contract.items():
        predictions = [dict(key=s["key"], photo_id=s["photo_id"], label=s["label"], **score(s, method)) for s in samples]
        counts = {k: 0 for k in ("tp", "fn", "fp", "tn")}
        for row in predictions:
            counts[("tp" if row["alert"] else "fn") if row["label"] else ("fp" if row["alert"] else "tn")] += 1
        output[name] = {"counts": counts, "threshold": method["threshold"],
                        "false_positives": [r["key"] for r in predictions if r["alert"] and not r["label"]],
                        "false_negatives": [r["key"] for r in predictions if not r["alert"] and r["label"]]}
    return output


def normal_summary(samples, contract):
    validate_samples(samples, "validation")
    methods, predictions = {}, {}
    for name, method in contract.items():
        values = [dict(key=s["key"], **score(s, method)) for s in samples]
        predictions[name] = values
        alerts = sum(v["alert"] for v in values)
        methods[name] = {"photos": len(values), "false_positives": alerts, "observed_false_positive_rate": alerts/len(values),
                         "threshold": method["threshold"], "within_descriptive_5_percent_budget": alerts/len(values) <= .05,
                         "alert_keys": [v["key"] for v in values if v["alert"]],
                         "alert_winning_regions": dict(Counter(v["winning_region"] for v in values if v["alert"])),
                         "region_exceedances": {r["region"]: sum((s["regions"][i][method["field"]]-r["center"])/r["scale"] >= method["threshold"] for s in samples)
                                                for i, r in enumerate(method["regions"])} }
    control = predictions["control_top1_mad"]
    for name, rows in predictions.items():
        methods[name]["paired_vs_control"] = {"added_alerts": sum(a["alert"] and not b["alert"] for a,b in zip(rows,control,strict=True)),
                                              "removed_alerts": sum(not a["alert"] and b["alert"] for a,b in zip(rows,control,strict=True))}
    return {"methods": methods, "predictions": predictions, "scope": POLICY["normal_validation_scope"],
            "new_defect_photos": 0, "independent_sensitivity_measured": False,
            "thresholds_refitted": False, "automatic_promotion": False}
