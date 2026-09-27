"""Bounded post-test diagnosis. No new score, calibration or model selection."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from scripts import mpi_score_probe_protocol as score

fit, ev = score.fit, score.ev
NAME = "mpi-pcb-patch-trace-v1"
SCORE_ZIP_SHA256 = "5784bf5d6cd24754788088d8e2f573a6804808934a3d7f6117b5ad75f4275795"
A = "a_region_mad_top1"
POLICY = {
    "schema_version": 1, "source_score_zip_sha256": SCORE_ZIP_SHA256,
    "selection": "all A FP; A calibration cutoff and middle; A TN middle; A FN low/middle/high and non-GT-winner; old TP and lowest newly caught TP",
    "grid_side": 64, "input_side": 256, "bank_shape": [78643, 768],
    "query_cells": "tile distance maximum and middle; GT-covered cell distance minimum/middle/maximum",
    "query_ties": "sort by distance then row-major flat index; extrema use first/last, middle uses n//2",
    "cell_gt": "any GT pixel in each 4x4 display cell, NOT receptive-field support",
    "provenance": "exact SHA256 of little-endian float32 selected bank vectors vs replayed original stride-2 training candidates",
    "ambiguous_origins": "retain every exact candidate source; never silently choose one as unique",
    "unresolved_origins": "report unresolved; no approximate match fallback",
    "nearest_neighbor_ties": "library-selected nearest index; other equidistant nonidentical bank vectors not enumerated",
    "maximum_origin_records": 100000,
    "training_performed": False, "threshold_changed": False, "automatic_pass_allowed": False,
    "service_model_changed": False, "confirmatory_claim_allowed": False,
    "test_previously_observed": True,
}


def select_cases(bundle, location_cases):
    """Use the already reviewed A result, explicitly post hoc; do not tune it."""
    contract = bundle["calibration.json"]
    photos = bundle["scores.json"][A]
    groups = defaultdict(list)
    for p in sorted(photos, key=lambda p: (p["score"], p["key"])):
        flag = p["score"] >= p["threshold"]
        groups[("tp" if flag else "fn") if p["label"] else ("fp" if flag else "tn")].append(p)
    selected = {}

    def add(key, reason):
        selected.setdefault(key, []).append(reason)

    for p in groups["fp"]:
        add(p["key"], "A_false_positive")
    cal = sorted(bundle["raw-scores.json"]["calibration"],
                 key=lambda p: (score.score(p, A, contract["normalization"]), p["key"]))
    for p in cal:
        if score.score(p, A, contract["normalization"]) == contract["thresholds"][A]:
            add(p["key"], "A_normal_calibration_cutoff")
    add(cal[len(cal)//2]["key"], "A_middle_normal_calibration_control")
    add(groups["tn"][len(groups["tn"])//2]["key"], "A_middle_test_normal_control")
    for i, reason in ((0, "lowest_A_miss"), (len(groups["fn"])//2, "middle_A_miss"), (-1, "nearest_cutoff_A_miss")):
        add(groups["fn"][i]["key"], reason)
    locations = {p["key"]: p["models"][ev.MODEL] for p in location_cases}
    raw = {p["key"]: p for p in bundle["raw-scores.json"]["test"]}
    for p in groups["fn"]:
        rows = raw[p["key"]]["regions"]
        values = [(r["top1"]-s["median"])/s["mad"] for r, s in zip(rows, contract["normalization"], strict=True)]
        gt = locations[p["key"]]["regions"][int(np.argmax(values))]["gt"]
        if gt is not None and gt["pixels"] == 0:
            add(p["key"], "A_miss_winning_region_without_GT")
    old = {p["key"]: p["score"] >= p["threshold"] for p in bundle["baseline-scores.json"]}
    for p in groups["tp"]:
        if old[p["key"]]:
            add(p["key"], "retained_old_TP_control")
    new_tp = [p for p in groups["tp"] if not old[p["key"]]]
    add(new_tp[0]["key"], "lowest_A_new_TP_control")
    return selected


def vector_hashes(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Expected finite float32 feature rows")
    rows = np.ascontiguousarray(values, dtype="<f4")
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in rows]


def select_query_cells(distances, mask):
    """Fixed 4x4 visual cells, not the larger overlapping CNN receptive fields."""
    values = np.asarray(distances)
    if values.shape != (4096,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid native patch distances")
    selected = {}

    def add(index, role):
        selected.setdefault(int(index), []).append(role)

    order = np.lexsort((np.arange(4096), values))
    add(order[-1], "tile_distance_maximum")
    add(order[len(order)//2], "tile_distance_middle_control")
    if mask is not None:
        if mask.shape != (256, 256) or mask.dtype != np.bool_:
            raise ValueError("Expected full-size boolean GT mask")
        support = mask.reshape(64, 4, 64, 4).any(axis=(1, 3)).ravel()
        gt_order = order[support[order]]
        if len(gt_order):
            for index, role in ((gt_order[0], "GT_distance_minimum"),
                                (gt_order[len(gt_order)//2], "GT_distance_middle"),
                                (gt_order[-1], "GT_distance_maximum")):
                add(index, role)
    return [{"flat_index": i, "grid_y": i//64, "grid_x": i%64, "roles": roles,
             "native_patch_distance": float(values[i])} for i, roles in selected.items()]


def check_replay(sample, expected, contract):
    score.validate_sample(sample, expected)
    delta = max(abs(r[k]-e[k]) for r, e in zip(sample["regions"], expected["regions"], strict=True)
                for k in ("top1", "native_raw"))
    a, old = score.score(sample, A, contract["normalization"]), score.score(expected, A, contract["normalization"])
    if max(delta, abs(a-old)) > 1e-4:
        raise ValueError("Frozen score replay drift; no threshold adjustment allowed")
    t = contract["thresholds"][A]
    if (a >= t) != (old >= t):
        raise ValueError("Frozen A decision changed despite small numerical drift")
    return {"maximum_absolute_score_difference": max(delta, abs(a-old)),
            "A_score": a, "A_threshold": t,
            "A_decision": "REVIEW_REQUIRED" if a >= t else "BELOW_THRESHOLD_NOT_AUTO_PASS"}


def targets_from_cases(cases):
    targets = {}
    for case in cases:
        for tile in case["tiles"]:
            for q in tile["queries"]:
                index, h = q["bank_index"], q["bank_vector_sha256"]
                if index in targets and targets[index] != h:
                    raise ValueError("Same bank index has inconsistent vector hash")
                targets[index] = h
    return {str(i): targets[i] for i in sorted(targets)}


def find_origins(features, indices, row, wanted):
    hashes = vector_hashes(features)
    if len(hashes) != len(indices):
        raise ValueError("Candidate feature count differs from stride grid")
    return [{"bank_vector_sha256": h, "photo_id": row["photo_id"], "region": row["region"],
             "member": row["member"], "member_sha256": row["member_sha256"],
             "decoded_rgb_sha256": row["decoded_rgb_sha256"],
             "grid_y": int(i)//64, "grid_x": int(i)%64, "candidate_flat_index": int(i)}
            for h, i in zip(hashes, indices, strict=True) if h in wanted]


def summarize_origins(targets, chunks, cases):
    by_hash = defaultdict(list)
    seen = set()
    wanted = set(targets.values())
    for chunk in chunks:
        for item in chunk["origins"]:
            key = (item["member"], item["candidate_flat_index"])
            if key in seen or item["bank_vector_sha256"] not in wanted:
                raise ValueError("Duplicate/unrequested provenance candidate")
            seen.add(key)
            by_hash[item["bank_vector_sha256"]].append(item)
    origins = {index: sorted(by_hash[h], key=lambda r: (r["photo_id"], r["region"], r["candidate_flat_index"]))
               for index, h in targets.items()}
    summaries = []
    for case in cases:
        counts = Counter()
        gt_counts = Counter()
        for tile in case["tiles"]:
            for q in tile["queries"]:
                regions = {r["region"] for r in origins[str(q["bank_index"])]}
                kind = ("unresolved" if not regions else "same_region_only" if regions == {tile["region"]}
                        else "cross_region_only" if tile["region"] not in regions else "mixed_source_regions")
                q["source_status"] = kind
                q["exact_source_candidates"] = len(origins[str(q["bank_index"])])
                counts[kind] += 1
                if any(role.startswith("GT_") for role in q["roles"]):
                    gt_counts[kind] += 1
        summaries.append({"key": case["key"], "query_counts": dict(counts), "GT_query_counts": dict(gt_counts)})
    return {"targets": targets, "origins_by_bank_index": origins,
            "target_bank_vectors": len(targets), "resolved_bank_vectors": sum(bool(x) for x in origins.values()),
            "ambiguous_bank_vectors": sum(len(x)>1 for x in origins.values()),
            "unresolved_bank_indices": [int(i) for i, x in origins.items() if not x],
            "cases": summaries,
            "interpretation": "Selected queries are correlated diagnostic examples, not detection metrics. Cross-region matching alone is not proof of an error or causal explanation."}
