"""Read-only attribution of saved normal resampling; no new scoring candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_mpi_scale_shrinkage as p

PRIOR_REPORT_SHA = "669ef3be6c4fe3043d381f134610e5d86e58da4052a0587ee526641c36ee03dc"
PRIOR_CODE_SHA = "26126dfe8ea98d2ac90231dbc9cff2f1f05214ad135f8cab005b046dd4582530"
RULES = {"name": "mpi-normal-influence-v1", "low": "tp < 50", "middle": "50 <= tp < 53",
         "retained": "tp >= 53", "groups_from_previous_screening_cutoffs": True,
         "association": "Pearson correlation and mean resampling multiplicity; descriptive only",
         "ties": "all distinct keys/regions; membership counts may exceed trial count",
         "reference_region": "grid3-2_s1024", "reference_region_is_posthoc": True,
         "training": False, "new_candidate": False, "data_removal": False,
         "threshold_change": False, "independent_evaluation": False}


def group_mask(tp):
    tp = np.asarray(tp)
    return {"all": np.ones(len(tp), dtype=bool), "low": tp < 50,
            "middle": (tp >= 50) & (tp < 53), "retained": tp >= 53}


def corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    da, db = a-a.mean(), b-b.mean()
    denominator = float(np.linalg.norm(da) * np.linalg.norm(db))
    return None if denominator == 0 else float(np.dot(da, db) / denominator)


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return {"mean": float(a.mean()), "q025_median_q975": np.quantile(a, [.025, .5, .975]).tolist(),
            "minimum": float(a.min()), "maximum": float(a.max())}


def attributions(arrays, raw, record, name):
    method = record["methods"][name]["parameters"]
    di = np.array(record["development_indices"])
    ci = np.array(record["calibration_indices"])
    cal_z = p.normalized(arrays["calibration"], method)
    cal_scores = cal_z.max(axis=1)
    boundary_indices = sorted({int(i) for i in ci if cal_scores[i] == method["threshold"]})
    if not boundary_indices:
        raise ValueError("No calibration quantile support")
    boundary = [{"key": raw["calibration"][i]["key"], "multiplicity": int(np.sum(ci == i)),
                 "regions": [p.REGIONS[j] for j in np.flatnonzero(cal_z[i] == cal_scores[i])]}
                for i in boundary_indices]
    dev = arrays["development"][di]
    quantile = np.quantile(dev, .9, axis=0, method="higher")
    support = {r: sorted({raw["development"][int(di[i])]["key"]
                         for i in np.flatnonzero(dev[:, j] == quantile[j])})
               for j, r in enumerate(p.REGIONS)}
    bounds = (np.array(method["center"]) + np.array(method["scale"])*method["threshold"]).tolist()
    return {"tp": record["methods"][name]["counts"]["tp"],
            "counts": record["methods"][name]["counts"],
            "threshold": method["threshold"], "center": method["center"], "scale": method["scale"],
            "raw_scale": method["raw_scale"], "raw_region_boundaries": bounds,
            "calibration_boundary": boundary, "development_q90_support": support}


def associations(indices, keys, tp):
    multiplicity = np.array([np.bincount(row, minlength=len(keys)) for row in indices])
    groups = group_mask(tp)
    return [{"key": key, "correlation_multiplicity_with_tp": corr(multiplicity[:, i], tp),
             "mean_multiplicity": {g: float(multiplicity[m, i].mean()) if m.any() else None for g, m in groups.items()}}
            for i, key in enumerate(keys)]


def summarize(traces, records, raw):
    tp = np.array([t["tp"] for t in traces])
    groups = group_mask(tp)
    result = {"groups": {}, "photo_associations": {}, "regions": {}}
    for g, mask in groups.items():
        chosen = [t for t, use in zip(traces, mask) if use]
        keys, regions = Counter(), Counter()
        for t in chosen:
            keys.update({b["key"] for b in t["calibration_boundary"]})
            regions.update({r for b in t["calibration_boundary"] for r in b["regions"]})
        result["groups"][g] = {"trials": len(chosen), "tp": describe([t["tp"] for t in chosen]),
            "threshold": describe([t["threshold"] for t in chosen]),
            "calibration_boundary_photo_membership": dict(keys.most_common()),
            "calibration_boundary_region_membership": dict(regions.most_common())}
    for cohort in ("development", "calibration"):
        result["photo_associations"][cohort] = associations(
            [r[cohort+"_indices"] for r in records], [r["key"] for r in raw[cohort]], tp)
    for j, region in enumerate(p.REGIONS):
        fields = {f: np.array([t[f][j] for t in traces]) for f in ("center", "raw_scale", "scale", "raw_region_boundaries")}
        result["regions"][region] = {
            "correlation_with_tp": {f: corr(v, tp) for f, v in fields.items()},
            "groups": {g: {f: describe(v[m]) for f, v in fields.items()} for g, m in groups.items()},
            "development_q90_support_membership": {
                g: dict(Counter(k for t, selected in zip(traces, mask) if selected
                                for k in t["development_q90_support"][region]).most_common())
                for g, mask in groups.items()}}
    result["examples"] = {label: {"trial_index": int(i), **traces[i]}
                          for label, i in (("lowest_tp_first", np.argmin(tp)), ("highest_tp_first", np.argmax(tp)))}
    return result


def normal_order_neighborhood(raw, arrays):
    """Normal-only fixed rank neighborhoods, all 12 regions, no removal simulation."""
    result = {}
    dev = arrays["development"]
    q = math.ceil(.9*(len(dev)-1))
    for j, r in enumerate(p.REGIONS):
        order = np.argsort(dev[:, j], kind="stable")
        result[r] = {"q90_index_zero_based": q, "median": float(np.median(dev[:, j])),
            "q90_value": float(dev[order[q], j]), "near_q90": [
                {"rank_one_based": k+1, "key": raw["development"][int(order[k])]["key"],
                 "raw_score": float(dev[order[k], j]),
                 "gap_from_previous": float(dev[order[k], j]-dev[order[k-1], j]) if k else None}
                for k in range(max(0, q-5), min(len(dev), q+6))]}
    return result


def execute(source, prior, output):
    cfg = p.load_config()
    raw, arrays, _, _ = p.load_source(source, cfg)
    if p.sha((prior/"report.json").read_bytes()) != PRIOR_REPORT_SHA:
        raise ValueError("Reviewed report changed")
    if p.sha(Path(p.__file__).read_bytes().replace(b"\r\n", b"\n")) != PRIOR_CODE_SHA:
        raise ValueError("Reviewed computation changed")
    prior_envelope = p.read_json((prior/"identity.json").read_bytes())
    prior_id = prior_envelope["identity"]
    previous = p.read_sealed(prior/"identity.json", prior_id)
    if (p.sha(p.canonical(previous)) != prior_id or previous["code_sha256"] != PRIOR_CODE_SHA
            or previous["config_sha256"] != p.CONFIG_SHA or previous["config"] != cfg
            or previous["cohorts"] != {g:[r["key"] for r in rows] for g,rows in raw.items()}):
        raise ValueError("Prior identity/cohort binding failed")
    report = p.read_sealed(prior/"report.json", prior_id)
    provenance = {"rules": RULES, "source_zip_sha256": cfg["source_zip_sha256"],
        "prior_identity": prior_id, "prior_report_sha256": PRIOR_REPORT_SHA,
        "code_sha256": p.sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "numpy": np.__version__, "python": sys.version}
    identity = p.sha(p.canonical(provenance))
    labels = np.array([r["label"] for r in raw["test"]])
    summaries, all_traces, trial_hashes = {}, {}, {}
    with p.locked(output):
        if not (output/"identity.json").exists() and any(v.name != ".lock" for v in output.iterdir()):
            raise ValueError("Unidentified existing output; preserved")
        p.publish(output/"identity.json", identity, provenance)
        for scope in cfg["bootstrap"]["scopes"]:
            expected = {f"{i:04d}.json" for i in range(cfg["bootstrap"]["trials_per_scope"])}
            if {v.name for v in (prior/"trials"/scope).iterdir()} != expected:
                raise ValueError("Trial inventory differs")
            traces = {n: [] for n in ("control", "candidate")}
            records = []
            for i, di, ci in p.trial_indices(scope, cfg["bootstrap"], len(raw["development"]), len(raw["calibration"])):
                path = prior/"trials"/scope/f"{i:04d}.json"
                record = p.read_sealed(path, prior_id)
                if record != p.trial(arrays, labels, di, ci):
                    raise ValueError(f"Stored trial does not replay: {scope}/{i}")
                if not all(v["valid"] for v in record["methods"].values()):
                    raise ValueError("Invalid trial requires separate diagnostic; do not drop")
                trial_hashes[f"{scope}/{i:04d}.json"] = p.sha(path.read_bytes())
                records.append(record)
                for name in traces:
                    traces[name].append(attributions(arrays, raw, record, name))
            if p.summarize_trials(records, cfg["development_screening"]) != report["results"]["stability"][scope]:
                raise ValueError("Prior stability summary differs")
            summaries[scope] = {n: summarize(t, records, raw) for n, t in traces.items()}
            all_traces[scope] = traces
            print(f"Verified and traced {scope}: {len(records)} paired trials", flush=True)
        payload = {"status": "descriptive_normal_influence_complete", "rules": RULES,
            "verified_trials": len(trial_hashes), "scopes": summaries,
            "normal_development_rank_neighborhoods": normal_order_neighborhood(raw, arrays),
            "limitation": "All associations are posthoc and dependent; no causal attribution, hypothesis test, data exclusion or independent performance claim.",
            "automatic_pass_allowed": False, "service_model_changed": False}
        p.publish(output/"input-trial-hashes.json", identity, trial_hashes)
        p.publish(output/"trace.json", identity, all_traces)
        p.publish(output/"report.json", identity, payload)
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=p.ROOT/"artifacts/analyses/mpi-scale-shrinkage-v1")
    parser.add_argument("--output", type=Path, default=p.ROOT/"artifacts/analyses/mpi-normal-influence-v1")
    args = parser.parse_args()
    result = execute(args.source, args.prior, args.output)
    print(p.canonical({"status": result["status"], "verified_trials":result["verified_trials"], "output":str(args.output)}).decode())


if __name__ == "__main__":
    main()
