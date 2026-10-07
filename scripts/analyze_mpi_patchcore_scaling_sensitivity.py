"""Read-only score diagnostics. No images, torch, fitting, or adopted thresholds.

Eight counterfactual contracts separate center/scale/threshold arithmetic. They
are NOT candidate searches: all combinations and all 600 original draws remain
visible. Shapley allocation describes these three numerical factors, not causal
effects of photographed objects. Leave-one-out is sensitivity only, not removal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import mpi_patchcore_scaling_protocol as p
from scripts import mpi_patchcore_scaling_evaluation as ev

INPUT_SHA = "976601795064eba599f25f6641695c0df60988c04e6c3cd4d03180900cc62c7d"
BASE_SHA = "4f995562c08b3c768ff591025a5f22db01daad5d52020b2d296912f4814bd73a"
FACTORS = ("center", "scale", "threshold")


def checked_bundle(path, expected):
    if p.fit.sha256_file(path) != expected:
        raise ValueError("Reviewed ZIP fingerprint differs")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError("Duplicate members or CRC failure")
        manifest = json.loads(z.read("handoff-manifest.json"))
        if set(names) != set(manifest) | {"handoff-manifest.json"}:
            raise ValueError("ZIP inventory differs")
        result = {}
        for name, sha in manifest.items():
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Payload fingerprint differs")
            result[name] = json.loads(raw)
    return result


def components(method):
    return (np.array([r["center"] for r in method["regions"]]),
            np.array([r["scale"] for r in method["regions"]]), method["threshold"])


def mixed(fixed, changed, bits):
    if not 0 <= bits < 8:
        raise ValueError("Expected three-factor bitmask")
    return {"threshold": changed["threshold"] if bits & 4 else fixed["threshold"],
            "regions": [{"center": b["center"] if bits & 1 else a["center"],
                         "scale": b["scale"] if bits & 2 else a["scale"]}
                        for a, b in zip(fixed["regions"], changed["regions"], strict=True)]}


def allocation(values):
    """Exact three-factor Shapley; interactions averaged over all six orders."""
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("Need all eight finite outcomes")
    result = {}
    for i, factor in enumerate(FACTORS):
        result[factor] = sum(math.factorial(s.bit_count()) * math.factorial(2-s.bit_count()) / 6
                             * (values[s | (1 << i)]-values[s])
                             for s in range(8) if not s & (1 << i))
    if not np.isclose(sum(result.values()), values[7]-values[0], atol=1e-10):
        raise ValueError("Attribution does not conserve outcome change")
    return result


def boundary_owners(values, photos, indices, method):
    """All tied original photos/regions at Q95; repeated draws keep multiplicity."""
    c, s, threshold = components(method)
    normalized = (values[indices]-c)/s
    scores = normalized.max(axis=1)
    counts = Counter(indices)
    owners = []
    for i in sorted(set(indices)):
        row = (values[i]-c)/s
        if float(row.max()) == threshold:
            owners.append({"key": photos[i]["key"], "multiplicity": counts[i],
                           "regions": [p.fit.REGIONS[j] for j in np.flatnonzero(row == threshold)]})
    if float(np.quantile(scores, .95, method="higher")) != threshold or not owners:
        raise ValueError("Boundary is not the frozen higher quantile")
    return owners


def quantiles(values):
    return np.quantile(values, [.025, .5, .975], axis=0).tolist()


def order_pivots(values, photos, indices, region):
    j = p.fit.REGIONS.index(region)
    ordered = sorted(indices, key=lambda i: (values[i, j], i))
    positions = {"median_lower": (len(indices)-1)//2, "median_upper": len(indices)//2,
                 "q90_higher": math.ceil(.9*(len(indices)-1))}
    return {name: {"key": photos[ordered[pos]]["key"], "value": float(values[ordered[pos], j])}
            for name, pos in positions.items()}


def trace_draw(raw, arrays, fixed, trial):
    di, ci = trial["development_indices"], trial["calibration_indices"]
    method = p.comparison.fit_method(arrays["development"][di], arrays["calibration"][ci])
    labels = [x["label"] for x in raw["test"]]
    variants = [p.comparison.counts(arrays, mixed(fixed, method, i), labels) for i in range(8)]
    if trial["candidate"] != {"valid": True, "counts": variants[7], "threshold": method["threshold"]}:
        raise ValueError("Saved bootstrap result does not reproduce")
    c, s, t = components(method)
    return {"iteration": trial["iteration"], "counts": variants[7], "method": method,
            "raw_boundaries": (c+s*t).tolist(),
            "boundary_owners": boundary_owners(arrays["calibration"], raw["calibration"], ci, method),
            "counterfactual_counts_bitmask_center_scale_threshold": variants,
            "miss_allocation": allocation([v["fn"] for v in variants]),
            "fp256_allocation": allocation([v["fp256"] for v in variants])}


def leave_one_out(raw, arrays):
    """Do not adopt the resulting contracts; retain every normal, including tails."""
    labels = [x["label"] for x in raw["test"]]
    rows = []
    for group in ("development", "calibration"):
        for i, photo in enumerate(raw[group]):
            values = {g: a for g, a in arrays.items()}
            values[group] = np.delete(values[group], i, axis=0)
            method = p.comparison.fit_method(values["development"], values["calibration"])
            rows.append({"key": photo["key"], "threshold": method["threshold"],
                         "counts": p.comparison.counts(arrays, method, labels)})
    return rows


def case_rows(data, baseline, raw):
    """All five new FPs plus the previously fixed 12 high-normal map summaries."""
    comp = data["evaluation/comparison.json"]
    keys = list(dict.fromkeys([k for ks in comp["new_false_positives"].values() for k in ks] +
           [x["key"] for x in data["evaluation/location.json"]["cases"] if x["role"] == "highest_normals"]))
    methods = {"baseline": baseline["freeze.json"]["contract"]["candidate_top1_q90"], "candidate": data["contract.json"]}
    sources = {"baseline": baseline["raw-scores.json"], "candidate": raw}
    by_model = {m: {x["key"]: x for xs in r.values() for x in xs} for m, r in sources.items()}
    result = []
    for key in keys:
        samples = {m: by_model[m][key] for m in sources}
        decisions = {m: p.comparison.c.scoring.score(samples[m], methods[m]) for m in sources}
        winner = decisions["candidate"]["winning_region"]
        j = p.fit.REGIONS.index(winner)
        row = {"key": key, "candidate_winning_region": winner, "models": {}}
        for m in sources:
            c, s, t = components(methods[m]); value = samples[m]["regions"][j]["top1"]
            row["models"][m] = {"decision": decisions[m], "same_region_raw_top1": value,
                                "center": float(c[j]), "scale": float(s[j]),
                                "same_region_raw_boundary": float(c[j]+s[j]*t),
                                "same_region_raw_margin": float(value-c[j]-s[j]*t)}
        location = next((x for x in data["evaluation/location.json"]["cases"] if x["key"] == key), None)
        row["map_statistics_available"] = location is not None
        row["actual_image_and_map_pixels_in_input"] = False
        result.append(row)
    return result


def analyze(data, baseline):
    raw = {**data["normal-scores.json"], **data["evaluation/scores.json"]}
    if data["plan.json"]["policy"] != p.POLICY:
        raise ValueError("Frozen recipe differs")
    for r in (raw, baseline["raw-scores.json"]):
        p.comparison.validate_raw(r, data["plan.json"]["cohorts"])
    fixed = data["contract.json"]
    p.comparison.contract_replay(raw, fixed)
    if ev.compare({"reference": baseline["raw-scores.json"], "contract": baseline["freeze.json"]["contract"]}, raw, fixed) != data["evaluation/comparison.json"]:
        raise ValueError("Frozen comparison differs")
    trials = data["evaluation/stability.json"]["trials"]
    expected = list(p.evaluation.draws())
    if set(trials) != {x[0] for x in expected} or any(len(ts) != 200 for ts in trials.values()):
        raise ValueError("Expected all original 600 draws")
    arrays = p.comparison.matrix(raw)
    traced = {}
    for scope, i, di, ci in expected:
        trial = trials[scope][i]
        if trial["scope"] != scope or trial["iteration"] != i or trial["development_indices"] != di.tolist() or trial["calibration_indices"] != ci.tolist():
            raise ValueError("Original draw identities changed")
        traced.setdefault(scope, []).append(trace_draw(raw, arrays, fixed, trial))
    summaries = {}
    dominant = Counter(x["winning_region"] for x in data["evaluation/comparison.json"]["predictions"]["candidate"]["test"] if x["label"]).most_common(1)[0][0]
    for scope, rows in traced.items():
        bad = [r for r in rows if r["counts"]["tp"] < 50]
        owners = Counter((x["key"], region) for r in bad for x in r["boundary_owners"] for region in x["regions"])
        summaries[scope] = {"draws": len(rows), "tp_below50": len(bad),
            "mean_misses": float(np.mean([r["counts"]["fn"] for r in rows])),
            "mean_miss_allocation": {f: float(np.mean([r["miss_allocation"][f] for r in rows])) for f in FACTORS},
            "bad_draw_mean_miss_allocation": {f: float(np.mean([r["miss_allocation"][f] for r in bad])) if bad else None for f in FACTORS},
            "bad_draw_boundary_owners": [{"key": k, "region": g, "draw_count": n} for (k, g), n in owners.most_common()],
            "scale_quantiles_by_region": quantiles([[x["scale"] for x in r["method"]["regions"]] for r in rows]),
            "threshold_quantiles": quantiles([r["method"]["threshold"] for r in rows]),
            "worst_draw": min(rows, key=lambda r: (r["counts"]["tp"], r["iteration"]))}
        index = summaries[scope]["worst_draw"]["iteration"]
        summaries[scope]["worst_draw_development_pivots"] = order_pivots(arrays["development"], raw["development"], trials[scope][index]["development_indices"], dominant)
    return {"status": "frozen_score_sensitivity_diagnosed", "input_sha256": INPUT_SHA,
            "baseline_sha256": BASE_SHA, "source_sha256": p.fit.sha256_file(Path(__file__)),
            "regions": p.fit.REGIONS, "fixed_contract": fixed,
            "dominant_defect_region": dominant,
            "fixed_development_pivots": order_pivots(arrays["development"], raw["development"], list(range(128)), dominant),
            "summaries": summaries, "traces": traced, "leave_one_out": leave_one_out(raw, arrays),
            "normal_cases": case_rows(data, baseline, raw),
            "model_changed": False, "threshold_changed": False, "automatic_pass_allowed": False,
            "interpretation": "Observed-score arithmetic diagnostics, not candidate adoption or causal visual attribution; no normal is removed"}


def write_new(path, report):
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("Different existing report; do not overwrite")
        return
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(checked_bundle(args.input, INPUT_SHA), checked_bundle(args.baseline, BASE_SHA))
    write_new(args.output, result)
    print(json.dumps({"status": result["status"], "draws": 600, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
