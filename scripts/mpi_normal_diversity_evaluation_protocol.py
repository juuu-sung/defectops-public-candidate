"""D-stage fixed-score comparison; bootstrap normals, never search test thresholds."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile

import numpy as np

from scripts import mpi_normal_diversity_fit_protocol as c

fit = c.fit
NAME = "mpi-normal-diversity-evaluation-v1"
FIT_ZIP_SHA = "582c92a5dcc3240fa689ab138a7c133fec1604160f1e89594a883da4da2ccf57"
CHECKPOINT_SHA = "6188f87130dbb8970261dddaee3520db13bd52c5cdea5b3967eb3b1e382f622b"
COUNTS = {"development": 128, "calibration": 169, "test": 109, "validation": 256}


def read_fit(path, project):
    if fit.sha256_file(path) != FIT_ZIP_SHA:
        raise ValueError("Reviewed C-stage ZIP changed")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        receipt = json.loads(z.read("receipt.json"))
        if len(names) != len(set(names)) or set(names) != {*receipt["files"], "receipt.json"}:
            raise ValueError("C-stage inventory differs")
        records = {}
        for name, sha in receipt["files"].items():
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("C-stage payload checksum differs")
            records[name] = json.loads(raw)
    plan, report = records["plan.json"], records["report.json"]
    if receipt["identity"] != plan or report["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("C-stage identity/checkpoint differs")
    for name, sha in plan["source_files_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Frozen C-stage source changed: {name}")
    if c.calibrate(records["first.json"]["development"], records["first.json"]["calibration"]) != records["contract.json"]:
        raise ValueError("C-stage normal calibration does not reproduce")
    if c.reload_check(records["first.json"], records["reload.json"], records["contract.json"]) != report["reload_verification"]:
        raise ValueError("C-stage reload differs")
    if not report["resource_review_required"] or report["attempts_with_errors"] != 1:
        raise ValueError("Reviewed C-stage resource violation must remain visible")
    return records


def validate_raw(raw, cohorts):
    if set(raw) != set(COUNTS) or set(cohorts) != set(COUNTS):
        raise ValueError("Incomplete score cohorts")
    for group, count in COUNTS.items():
        if len(raw[group]) != count or len(cohorts[group]) != count:
            raise ValueError("Cohort size changed")
        keys = [p["key"] for p in raw[group]]
        if len(set(keys)) != count:
            raise ValueError("Duplicate photo keys")
        for sample, photo in zip(raw[group], cohorts[group], strict=True):
            c.scoring.old.validate_sample(sample, photo)
        labels = [p["label"] for p in raw[group]]
        if (group == "test" and (labels.count(1), labels.count(0)) != (55, 54)) or (group != "test" and any(labels)):
            raise ValueError("Cohort labels changed")


def matrix(raw):
    return {g: np.asarray([[r["top1"] for r in s["regions"]] for s in samples], dtype=np.float64)
            for g, samples in raw.items()}


def fit_method(development, calibration):
    center = np.median(development, axis=0)
    scale = np.quantile(development, .9, axis=0, method="higher") - center
    if not np.isfinite(scale).all() or np.any(scale <= 1e-6):
        raise ValueError("Degenerate Q90 scale; no replacement draw")
    threshold = float(np.quantile(np.max((calibration-center)/scale, axis=1), .95, method="higher"))
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite threshold")
    return {"field": "top1", "scale_kind": "q90", "threshold": threshold,
            "regions": [{"region": r, "center": float(x), "scale": float(y)}
                        for r, x, y in zip(fit.REGIONS, center, scale, strict=True)]}


def scores(values, method):
    center = np.asarray([r["center"] for r in method["regions"]])
    scale = np.asarray([r["scale"] for r in method["regions"]])
    return np.max((values-center)/scale, axis=1)


def counts(values, method, labels):
    decisions = scores(values["test"], method) >= method["threshold"]
    labels = np.asarray(labels, dtype=bool)
    return {"tp": int(np.sum(decisions & labels)), "fn": int(np.sum(~decisions & labels)),
            "fp54": int(np.sum(decisions & ~labels)), "tn54": int(np.sum(~decisions & ~labels)),
            "fp256": int(np.sum(scores(values["validation"], method) >= method["threshold"]))}


def contract_replay(raw, method):
    computed = c.calibrate(raw["development"], raw["calibration"])
    delta = abs(computed["threshold"] - method["threshold"])
    flips = 0
    for a, b in zip(computed["regions"], method["regions"], strict=True):
        if a["region"] != b["region"]:
            raise ValueError("Contract region order differs")
        delta = max(delta, *(abs(a[k]-b[k]) for k in ("center", "scale")))
    for samples in raw.values():
        for sample in samples:
            a, b = c.scoring.score(sample, computed), c.scoring.score(sample, method)
            delta = max(delta, abs(a["score"]-b["score"]))
            flips += a["alert"] != b["alert"]
    if delta > 1e-12 or flips:
        raise ValueError("Frozen baseline Q90 replay failed")
    return {"maximum_absolute_difference": delta, "decision_changes": flips, "passed": True}


def prepare(project, handoffs):
    source = c.selection.spatial.prepare(project, handoffs)
    records = read_fit(handoffs/"mpi-normal-diversity-fit-v1.zip", project)
    policy = json.loads((project/"configs/mpi-normal-diversity-v1.json").read_bytes())
    if fit.digest(policy) != c.selection.POLICY_SHA256 or records["plan.json"]["policy"] != policy:
        raise ValueError("Preregistered policy changed")
    if records["plan.json"]["model_config"] != source["config"]:
        raise ValueError("Model recipe changed")
    cohorts, baseline = source["cohorts"], source["reference"]
    validate_raw(baseline, cohorts)
    for group in ("development", "calibration"):
        for sample, photo in zip(records["first.json"][group], cohorts[group], strict=True):
            c.scoring.old.validate_sample(sample, photo)
    for group in ("development", "calibration", "validation"):
        if fit.digest(cohorts[group]) != policy["source_sha256"][group+"_rows"]:
            raise ValueError("Frozen photo order changed")
    methods = {"baseline": source["contract"]["candidate_top1_q90"], "candidate": records["contract.json"]}
    replay = contract_replay(baseline, methods["baseline"])
    labels = [p["label"] for p in baseline["test"]]
    baseline_counts = counts(matrix(baseline), methods["baseline"], labels)
    if baseline_counts != {"tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14}:
        raise ValueError("Reviewed baseline results changed")
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    identity = {"name": NAME, "stage": "D_observed_comparison_not_independent_final",
                "fit_zip_sha256": FIT_ZIP_SHA, "policy": policy,
                "cohorts_sha256": fit.digest(cohorts), "baseline_raw_sha256": fit.digest(baseline),
                "methods": methods, "baseline_replay": replay,
                "git_commit": subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                             check=True, capture_output=True, text=True).stdout.strip(),
                "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}
    return source, records, identity


def predictions(raw, method):
    return {g: [{k: p[k] for k in ("key", "photo_id", "label")} | c.scoring.score(p, method)
                for p in samples] for g, samples in raw.items()}


def compare(baseline, candidate, methods):
    labels = [p["label"] for p in baseline["test"]]
    result = {m: counts(matrix(raw), methods[m], labels) for m, raw in (("baseline", baseline), ("candidate", candidate))}
    pred = {m: predictions(raw, methods[m]) for m, raw in (("baseline", baseline), ("candidate", candidate))}
    lost, gained = [], []
    for a, b in zip(pred["baseline"]["test"], pred["candidate"]["test"], strict=True):
        if a["key"] != b["key"] or a["label"] != b["label"]:
            raise ValueError("Paired test inventory differs")
        if a["label"] and a["alert"] and not b["alert"]:
            lost.append(a["key"])
        if a["label"] and not a["alert"] and b["alert"]:
            gained.append(a["key"])
    metrics = result["candidate"]
    gates = {"tp_at_least_53": metrics["tp"] >= 53, "lost_baseline_detections_zero": not lost,
             "fp54_at_most_2": metrics["fp54"] <= 2, "fp256_at_most_12": metrics["fp256"] <= 12}
    return {"counts": result, "lost_baseline_detections": lost, "gained_detections": gained,
            "gates": gates, "passed": all(gates.values())}, pred


def bootstrap(baseline, candidate, policy, check=lambda: None):
    """Same original-photo draws for both banks; no test/mask resampling or fitting."""
    b = policy["acceptance"]["bootstrap"]
    arrays = {"baseline": matrix(baseline), "candidate": matrix(candidate)}
    labels = [p["label"] for p in baseline["test"]]
    results, summaries = {}, {}
    for scope, seed in zip(b["scopes"], b["seeds"], strict=True):
        rng = np.random.Generator(np.random.PCG64(seed))
        trials = []
        for i in range(b["repetitions_per_scope"]):
            check()
            di = rng.integers(0, 128, 128) if scope in ("development_only", "both") else np.arange(128)
            ci = rng.integers(0, 169, 169) if scope in ("calibration_only", "both") else np.arange(169)
            trial = {"index": i, "development_indices": di.tolist(), "calibration_indices": ci.tolist(), "models": {}}
            for model, values in arrays.items():
                try:
                    method = fit_method(values["development"][di], values["calibration"][ci])
                    trial["models"][model] = {"valid": True, "method": method, "counts": counts(values, method, labels)}
                except ValueError as exc:
                    trial["models"][model] = {"valid": False, "error": str(exc)}
            trial["valid"] = all(x["valid"] for x in trial["models"].values())
            trials.append(trial)
        results[scope] = trials
        summaries[scope] = summarize(trials, b)
    return {"policy": b, "scopes": results, "summary": summaries,
            "passed": all(s["passed"] for s in summaries.values()),
            "interpretation": "normal-calibration sensitivity on observed photos; not generalization confidence intervals"}


def summarize(trials, policy):
    invalid = sum(not t["valid"] for t in trials)
    if invalid or len(trials) != policy["repetitions_per_scope"]:
        return {"invalid_repetitions": invalid, "quantiles": None, "passed": False,
                "reason": "No dropping or replacing invalid/missing paired repetitions"}
    q = {m: {k: np.quantile([t["models"][m]["counts"][k] for t in trials], [.025, .5, .975], method="linear").tolist()
             for k in ("tp", "fp54", "fp256")} for m in ("baseline", "candidate")}
    b, a = q["baseline"], q["candidate"]
    widths = {m: q[m]["tp"][2]-q[m]["tp"][0] for m in q}
    gates = {"tp_q025": a["tp"][0] >= policy["minimum_tp_q025"],
             "tp_width_halved": widths["candidate"] <= policy["maximum_tp_q975_minus_q025_relative_to_baseline"]*widths["baseline"],
             "fp54_q975": a["fp54"][2] <= policy["maximum_fp54_q975"],
             "fp256_q975": a["fp256"][2] <= policy["maximum_fp256_q975"]}
    return {"invalid_repetitions": 0, "quantiles": q, "tp_width": widths, "gates": gates, "passed": all(gates.values())}
