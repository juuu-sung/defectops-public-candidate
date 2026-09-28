"""Read-only, post-test attribution of two already saved MPI anomaly maps."""
from __future__ import annotations

import math

import numpy as np

from scripts import mpi_region_search_protocol as p

fit, ev = p.fit, p.ev
NAME = "mpi-pcb-region-map-review-v1"
REGION_ZIP_SHA256 = "c56b03bd6d573b54cbbbe1543e3179eb697652919fd41e92a96bd2b547f0a107"
CONTROLS = {
    "defect/0024": "persistent_fn", "defect/0041": "persistent_fn",
    "defect/0020": "persistent_fn", "defect/0034": "persistent_fn",
    "defect/0010": "retained_tp", "good/0012": "retained_fp",
    "good/0020": "retained_tn",
}
POLICY = {
    "schema_version": 1, "source_region_zip_sha256": REGION_ZIP_SHA256,
    "selection": "all changed decisions, then seven fixed historical controls; no score ranking",
    "display_raw_map_cap": 1.5 * ev.THRESHOLD,
    "display": "same fixed raw scale in both arms; NOT a pixel threshold or normalized A score",
    "support": "top 656 pixels; display all cutoff ties; statistics give tie-aware bounds",
    "location_excluded_originals": ["defect/0006"],
    "calibration_reference": "0411 scores and historical shared-only cards; no paired normal maps exist",
    "inference_performed": False, "training_performed": False,
    "thresholds_changed": False, "service_model_changed": False,
    "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
}


def contracts(data):
    shared, candidate = data["shared-A-calibration.json"], data["candidate-calibration.json"]
    return {p.ARMS[0]: {"normalization": shared["normalization"], "threshold": shared["thresholds"][p.trace.A]},
            p.ARMS[1]: candidate}


def select(scores, controls=CONTROLS):
    old, new = (scores[a] for a in p.ARMS)
    pairs = ev.paired_changes(old, new)
    groups = {}
    for label, name in ((1, "defect"), (0, "normal")):
        for before, after, group in ((True, False, "new_fn" if label else "cleared_fp"),
                                     (False, True, "new_tp" if label else "new_fp")):
            pair = "old_only_review" if before else "new_only_review"
            for key in pairs[name]["photo_keys"][pair]:
                groups[key] = group
    by_key = {s["key"]: (s, t) for s, t in zip(old, new, strict=True)}
    expected = {"persistent_fn": (1, False), "retained_tp": (1, True),
                "retained_fp": (0, True), "retained_tn": (0, False)}
    for key, group in controls.items():
        if key not in by_key or key in groups:
            raise ValueError(f"Fixed review control missing or changed: {key}")
        for row in by_key[key]:
            if (row["label"], row["decision"] == "REVIEW_REQUIRED") != expected[group]:
                raise ValueError(f"Fixed review control is not {group}: {key}")
        groups[key] = group
    return [{"key": s["key"], "group": groups[s["key"]], "cohort_ordinal": i+1}
            for i, s in enumerate(old) if s["key"] in groups]


def score_details(sample, contract):
    values = []
    for row, stats in zip(sample["regions"], contract["normalization"], strict=True):
        if row["region"] != stats["region"] or not math.isfinite(stats["mad"]) or stats["mad"] <= 1e-6:
            raise ValueError("Invalid or reordered region normalization")
        normalized = (row["top1"] - stats["median"]) / stats["mad"]
        boundary = stats["median"] + stats["mad"] * contract["threshold"]
        if not all(math.isfinite(v) for v in (normalized, boundary)):
            raise ValueError("Non-finite normalized score")
        values.append({"region": row["region"], "raw_top1": row["top1"],
                       "normal_median": stats["median"], "normal_mad": stats["mad"],
                       "normalized_score": normalized, "raw_equivalent_image_boundary": boundary,
                       "raw_boundary_margin": row["top1"]-boundary})
    score = max(v["normalized_score"] for v in values)
    return {"score": score, "threshold": contract["threshold"], "margin": score-contract["threshold"],
            "decision": "REVIEW_REQUIRED" if score >= contract["threshold"] else "BELOW_THRESHOLD_NOT_AUTO_PASS",
            "winning_regions": [v["region"] for v in values if v["normalized_score"] == score],
            "regions": values}


def describe(photo, samples, arrays, masks, frozen_contracts):
    """Raw-map attribution stays separate from region-normalized image decisions."""
    eligible = photo["key"] != "defect/0006"
    arms = {}
    for arm in p.ARMS:
        p.validate_top_sample(samples[arm], photo)
        a = np.asarray(arrays[arm])
        if a.shape != (12, 256, 256) or a.dtype != np.float32:
            raise ValueError("Expected ordered float32 saved maps")
        detail = score_details(samples[arm], frozen_contracts[arm])
        for i, row in enumerate(detail["regions"]):
            # One damaged mask excludes the WHOLE original from location statistics.
            mask = masks[i] if eligible else None
            row["map"] = ev.maps.region_statistics(a[i], mask, row["raw_top1"])
            row["gt_pixels"] = int(mask.sum()) if mask is not None else None
            row["display_saturated_fraction"] = float(np.mean(a[i] >= POLICY["display_raw_map_cap"]))
        gt_rows = [r for r in detail["regions"] if r["gt_pixels"]]
        winners = [r for r in detail["regions"] if r["region"] in detail["winning_regions"]]
        detail["winner_has_gt"] = any(r["gt_pixels"] for r in winners) if eligible and photo["label"] else None
        detail["gt_region_count"] = len(gt_rows) if eligible else None
        arms[arm] = detail
    minimum = float(np.min(arrays[p.ARMS[1]]-arrays[p.ARMS[0]]))
    if minimum < -p.POLICY["map_monotonicity_tolerance"]:
        raise ValueError("Paired saved maps violate same-region monotonicity")
    deltas = []
    for before, after in zip(arms[p.ARMS[0]]["regions"], arms[p.ARMS[1]]["regions"], strict=True):
        delta = {"region": before["region"], "raw_top1_change": after["raw_top1"]-before["raw_top1"],
                 "raw_boundary_change": after["raw_equivalent_image_boundary"]-before["raw_equivalent_image_boundary"],
                 "normalized_score_change": after["normalized_score"]-before["normalized_score"],
                 "map_mean_change": after["map"]["map_mean"]-before["map"]["map_mean"]}
        if before["gt_pixels"]:
            bg, ag = before["map"]["gt"], after["map"]["gt"]
            delta["gt_statistics_before"] = bg
            delta["gt_statistics_after"] = ag
        deltas.append(delta)
    return {k: photo[k] for k in ("key", "photo_id", "label")} | {
        "localization_eligible": eligible, "arms": arms, "region_deltas": deltas,
        "minimum_candidate_minus_shared_saved_map": minimum,
        "reconstructed_image_used": photo["key"] == "defect/0054",
        "automatic_cause_assigned": False, "automatic_pass_allowed": False}


def calibration_reference(data):
    cs = contracts(data)
    values = {a: [score_details(s["arms"][a], cs[a]) | {"key": s["arms"][a]["key"]}
                  for s in data["raw-scores.json"]["calibration"]] for a in p.ARMS}
    refs = {}
    for arm in p.ARMS:
        row = next(v for v in values[arm] if v["key"] == "calibration/0411")
        row["strictly_higher_calibration_photos"] = sum(v["score"] > row["score"] for v in values[arm])
        row["equals_published_threshold"] = abs(row["score"]-cs[arm]["threshold"]) <= 1e-12
        refs[arm] = row
    return {"photo": "calibration/0411", "arms": refs, "paired_normal_maps_available": False,
            "maps_recomputed": False, "historical_cards_arm": p.ARMS[0],
            "note": "Numeric reference only; historical shared-only cards are not candidate calibration maps. No removal or relabeling."}


def summarize(cases):
    groups = {}
    for case in cases:
        group = groups.setdefault(case["group"], {"photo_keys": [], "location_eligible_defect_photos": 0,
                       "winner_has_no_gt": {a: 0 for a in p.ARMS}})
        group["photo_keys"].append(case["key"])
        if case["label"] and case["localization_eligible"]:
            group["location_eligible_defect_photos"] += 1
            for arm in p.ARMS:
                group["winner_has_no_gt"][arm] += not case["arms"][arm]["winner_has_gt"]
    return {"selected_photos": len(cases), "groups": groups,
            "scope": "Post-test selected cases, not whole-cohort accuracy or independent validation. Winner region containing GT does not prove top pixels hit GT.",
            "pixel_accuracy_claim_allowed": False, "automatic_cause_assigned": False}
