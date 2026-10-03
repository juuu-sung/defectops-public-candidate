"""CPU arithmetic attribution of two frozen pipelines, not a candidate search.

The 16 mixed states are explanations only. They are never ranked by accuracy,
calibrated, deployed or returned as alternative classifier performance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mpi_scale_shrinkage as old
from scripts import mpi_normal_diversity_evaluation_protocol as evaluation

NAME = "mpi-diversity-score-decomposition-v1"
EVALUATION_SHA = "7d2d859341ffdd1607fe8982de11063fbd796c0a174dfbe86c095729d49ad5f4"
VALIDATION_SHA = "4f995562c08b3c768ff591025a5f22db01daad5d52020b2d296912f4814bd73a"
FACTORS = ("raw", "center", "scale", "threshold")
ORDERS = tuple(itertools.permutations(range(4)))
TOLERANCE = 1e-10


def budget(start):
    elapsed, rss = time.monotonic() - start, old.peak_rss_bytes()
    if elapsed > 1800 or rss > 2 * 1024**3:
        raise RuntimeError("Diagnostic CPU budget exceeded; no sampling or fallback")
    return {"elapsed_seconds": elapsed, "peak_rss_bytes": rss}


def method_vectors(method):
    if (method["field"] != "top1" or method["scale_kind"] != "q90"
            or [r["region"] for r in method["regions"]] != old.REGIONS):
        raise ValueError("Frozen method definition/order differs")
    return [np.asarray([r[k] for r in method["regions"]], dtype=np.float64)
            for k in ("center", "scale")] + [float(method["threshold"])]


def decompose(raw_before, raw_after, before, after):
    """Mean marginal change over all 24 factor orders, including max switches."""
    if np.shape(raw_before) != (12,) or np.shape(raw_after) != (12,):
        raise ValueError("Exactly 12 regions required")
    endpoints = [[np.asarray(raw_before, dtype=np.float64), *method_vectors(before)],
                 [np.asarray(raw_after, dtype=np.float64), *method_vectors(after)]]
    for parts in endpoints:
        if not all(np.isfinite(x).all() for x in map(np.asarray, parts)) or np.any(parts[2] <= 1e-6):
            raise ValueError("Nonfinite input or invalid scale")
    margins, winners, ties = [], [], []
    for state in range(16):
        raw, center, scale, threshold = [endpoints[(state >> i) & 1][i] for i in range(4)]
        z = (raw - center) / scale
        margin = float(np.max(z) - threshold)
        if not np.isfinite(z).all() or not np.isfinite(margin):
            raise ValueError("Nonfinite intermediate arithmetic")
        margins.append(margin)
        winners.append(old.REGIONS[int(np.argmax(z))])
        ties.append([old.REGIONS[i] for i in np.flatnonzero(z == z.max())])
    effects = []
    for order in ORDERS:
        state, row = 0, [0.] * 4
        for factor in order:
            next_state = state | (1 << factor)
            row[factor] = margins[next_state] - margins[state]
            state = next_state
        effects.append(row)
    effects = np.asarray(effects)
    means = effects.mean(axis=0)
    delta = margins[15] - margins[0]
    residual = float(means.sum() - delta)
    if abs(residual) > TOLERANCE:
        raise ValueError("Attribution additivity failed")
    return {"before_margin": margins[0], "after_margin": margins[15], "margin_change": delta,
            "contributions": dict(zip(FACTORS, map(float, means), strict=True)),
            "combined_calibration_contribution": float(means[1:].sum()),
            "order_effect_ranges": {name: [float(effects[:, i].min()), float(effects[:, i].max())]
                                    for i, name in enumerate(FACTORS)},
            "maximum_absolute_residual": abs(residual),
            "before_winning_region": winners[0], "after_winning_region": winners[15],
            "endpoint_winning_region_changed": winners[0] != winners[15],
            "mixed_state_margins_not_valid_models": margins,
            "mixed_state_winners": winners, "mixed_state_ties": ties}


def read_candidate(path):
    if old.sha(path.read_bytes()) != EVALUATION_SHA:
        raise ValueError("Reviewed evaluation ZIP identity differs")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        receipt = old.read_json(z.read("receipt.json"))
        if len(names) != len(set(names)) or set(names) != set(receipt["files"]) | {"receipt.json"} or z.testzip():
            raise ValueError("Evaluation ZIP inventory/CRC differs")
        records = {}
        for name, sha in receipt["files"].items():
            data = z.read(name)
            if old.sha(data) != sha:
                raise ValueError("Evaluation payload checksum differs")
            # Bootstrap was already verified; this diagnostic does not re-run it.
            if name in {"plan.json", "report.json", "score.json"}:
                records[name] = old.read_json(data)
    plan = records["plan.json"]
    if receipt["identity"] != plan:
        raise ValueError("Evaluation receipt identity differs")
    for name, sha in plan["source_files_lf_sha256"].items():
        if old.sha((ROOT / name).read_bytes().replace(b"\r\n", b"\n")) != sha:
            raise ValueError(f"Frozen D-stage source differs: {name}")
    return records


def paired_validate(baseline, candidate):
    arrays = [old.matrices(raw, evaluation.COUNTS) for raw in (baseline, candidate)]
    for group in evaluation.COUNTS:
        for a, b in zip(baseline[group], candidate[group], strict=True):
            if any(a[k] != b[k] for k in ("key", "photo_id", "label")):
                raise ValueError("Paired identities/labels/order differ")
            if [r["member"] for r in a["regions"]] != [r["member"] for r in b["regions"]]:
                raise ValueError("Paired image member paths differ")
    return arrays


def summarize(rows):
    groups = {
        "all_365": rows,
        "all_defects55": [r for r in rows if r["label"] == 1],
        "all_test_normals54": [r for r in rows if r["group"] == "test" and not r["label"]],
        "all_validation_normals256": [r for r in rows if r["group"] == "validation"],
        "lost_defects": [r for r in rows if r["label"] and r["before_alert"] and not r["after_alert"]],
        "gained_defects": [r for r in rows if r["label"] and not r["before_alert"] and r["after_alert"]],
    }
    for group in ("test", "validation"):
        normals = [r for r in rows if r["group"] == group and not r["label"]]
        groups[group + "_added_fp"] = [r for r in normals if not r["before_alert"] and r["after_alert"]]
        groups[group + "_removed_fp"] = [r for r in normals if r["before_alert"] and not r["after_alert"]]
    result = {}
    for name, subset in groups.items():
        result[name] = {"count": len(subset), "keys": [r["key"] for r in subset],
            "mean_contributions": {k: float(np.mean([r["contributions"][k] for r in subset])) if subset else None
                                   for k in FACTORS},
            "mean_margin_change": float(np.mean([r["margin_change"] for r in subset])) if subset else None,
            "endpoint_winner_changes": sum(r["endpoint_winning_region_changed"] for r in subset),
            "raw_negative_count": sum(r["contributions"]["raw"] < -TOLERANCE for r in subset),
            "calibration_negative_count": sum(r["combined_calibration_contribution"] < -TOLERANCE for r in subset)}
    return result


def read_original_baseline(path, spatial, plan):
    """D used the original validation record, not the later spatial replay."""
    if old.sha(path.read_bytes()) != VALIDATION_SHA:
        raise ValueError("Original baseline validation ZIP differs")
    with zipfile.ZipFile(path) as z:
        manifest = old.read_json(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {"handoff-manifest.json"} or z.testzip():
            raise ValueError("Original baseline ZIP inventory/CRC differs")
        for name, sha in manifest.items():
            if old.sha(z.read(name)) != sha:
                raise ValueError("Original baseline payload checksum differs")
        raw = old.read_json(z.read("raw-scores.json"))
    # Canonical provenance must match D exactly, not just the performance totals.
    if evaluation.fit.digest(raw) != plan["baseline_raw_sha256"]:
        raise ValueError("Original baseline digest does not match D-stage")
    a, b = paired_validate(spatial, raw)
    if any(not np.array_equal(a[g], b[g]) for g in evaluation.COUNTS):
        raise ValueError("Original and spatial-replay top1 scores differ")
    return raw


def analyze(source_zip, candidate_zip, validation_zip, start):
    cfg = old.load_config()
    spatial, _, stored, reference = old.load_source(source_zip, cfg)
    records = read_candidate(candidate_zip)
    candidate, plan = records["score.json"]["raw"], records["plan.json"]
    baseline = read_original_baseline(validation_zip, spatial, plan)
    arrays = paired_validate(baseline, candidate)
    if evaluation.fit.digest(baseline) != plan["baseline_raw_sha256"] or stored != plan["methods"]["baseline"]:
        raise ValueError("Baseline provenance does not match D-stage")
    old.replay(arrays[0], baseline, stored, reference, cfg)
    replays = {name: evaluation.contract_replay(raw, plan["methods"][name])
               for name, raw in (("baseline", baseline), ("candidate", candidate))}
    comparison, predictions = evaluation.compare(baseline, candidate, plan["methods"])
    if comparison != records["report.json"]["comparison"]:
        raise ValueError("D-stage endpoint comparison does not reproduce")
    rows, max_error = [], 0.
    for group in ("test", "validation"):
        for i, sample in enumerate(baseline[group]):
            budget(start)
            result = decompose(arrays[0][group][i], arrays[1][group][i],
                               plan["methods"]["baseline"], plan["methods"]["candidate"])
            for side, model in (("before", "baseline"), ("after", "candidate")):
                pred = predictions[model][group][i]
                max_error = max(max_error, abs(result[side + "_margin"] - (pred["score"]-plan["methods"][model]["threshold"])))
                if (result[side + "_margin"] >= 0) != pred["alert"]:
                    raise ValueError("Endpoint decision differs")
                result[side + "_alert"] = pred["alert"]
            if max_error > TOLERANCE:
                raise ValueError("Endpoint score tolerance exceeded")
            rows.append({"key": sample["key"], "label": sample["label"], "group": group, **result})
    summary = summarize(rows)
    if (summary["lost_defects"]["keys"] != ["defect/0027", "defect/0028", "defect/0044"]
            or summary["gained_defects"]["count"] != 0
            or summary["validation_added_fp"]["count"] != 13
            or summary["validation_removed_fp"]["count"] != 8):
        raise ValueError("Mandatory case inventory differs")
    return {"name": NAME, "status": "arithmetic_diagnostic_completed_not_candidate",
            "input_zip_sha256": {"spatial_replay": cfg["source_zip_sha256"], "baseline_original": VALIDATION_SHA, "candidate": EVALUATION_SHA},
            "method": "max((raw-center)/scale)-threshold; 16 states, all 24 orders equally averaged",
            "factors": FACTORS, "positive_contribution": "increases alert margin, not universally better",
            "interpretation": "arithmetic allocation conditional on two observed endpoints; not causal or independent evidence",
            "endpoint_comparison": comparison, "endpoint_normal_replays": replays,
            "maximum_endpoint_error": max_error,
            "maximum_additivity_error": max(r["maximum_absolute_residual"] for r in rows),
            "summary": summary, "rows": rows,
            "training_performed": False, "gpu_inference_performed": False,
            "images_or_masks_read": 0, "threshold_optimized": False,
            "mixed_state_performance_computed": False, "candidate_accepted": False,
            "automatic_pass_allowed": False, "service_model_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--evaluation-zip", type=Path, required=True)
    parser.add_argument("--validation-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Diagnostic output exists; preserve and review, no overwrite")
    start = time.monotonic()
    result = analyze(args.baseline_zip, args.evaluation_zip, args.validation_zip, start)
    result["resource"] = budget(start)
    result["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    identity = {"authorization": "2026-10-02 user approved stored-score diagnostic only",
                "script_lf_sha256": old.sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                "protocol_lf_sha256": old.sha((ROOT/"docs/mpi-diversity-score-decomposition-v1.md").read_bytes().replace(b"\r\n", b"\n"))}
    old.publish(args.output, identity, result)
    print(json.dumps({k: result[k] for k in ("status", "maximum_endpoint_error", "maximum_additivity_error", "resource")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
