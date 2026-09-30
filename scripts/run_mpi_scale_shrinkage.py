"""Frozen MPI-C1-01 CPU-only comparison. Never loads images or a model."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mpi-scale-shrinkage-v1.json"
CONFIG_SHA = "890041c2140ca3a6176fb798d4fbd6a8b32b7f04204fa765c49a7ee029bc1cb9"
REGIONS = [f"grid{i}-{j}_s1024" for i in range(1, 5) for j in range(1, 4)]


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(data, parse_constant=invalid)


def load_config(path=CONFIG):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if sha(data) != CONFIG_SHA:
        raise ValueError("Preregistered configuration changed; no automatic variant")
    return read_json(data)


def matrices(raw, counts):
    if set(raw) != set(counts):
        raise ValueError("Cohort inventory differs")
    arrays, used_keys, normal_ids = {}, set(), set()
    for group, n in counts.items():
        rows = raw[group]
        if len(rows) != n:
            raise ValueError("Cohort count differs")
        for row in rows:
            if row["key"] in used_keys or type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError("Duplicate identity or invalid label")
            used_keys.add(row["key"])
            if [r["region"] for r in row["regions"]] != REGIONS:
                raise ValueError("Region order differs")
            if group != "test":
                if (row["label"] != 0 or row["key"] != f"{group}/{row['photo_id']}"
                        or row["photo_id"] in normal_ids):
                    raise ValueError("Normal role/identity overlap")
                normal_ids.add(row["photo_id"])
            elif row["key"] != f"{'defect' if row['label'] else 'good'}/{row['photo_id']}":
                raise ValueError("Test identity differs")
        arrays[group] = np.array([[r["top1"] for r in row["regions"]] for row in rows], dtype=np.float64)
        if not np.isfinite(arrays[group]).all() or (arrays[group] < 0).any():
            raise ValueError("Invalid raw score")
    return arrays


def load_source(path, cfg):
    if sha(path.read_bytes()) != cfg["source_zip_sha256"]:
        raise ValueError("Source ZIP identity differs")
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError("Duplicate member or ZIP CRC failure")
        manifest = read_json(z.read("handoff-manifest.json"))
        if set(z.namelist()) != set(manifest) | {"handoff-manifest.json"}:
            raise ValueError("Payload inventory differs")
        for name, digest in manifest.items():
            if sha(z.read(name)) != digest:
                raise ValueError(f"Payload changed: {name}")
        for name, key in [("raw-scores.json", "source_raw_scores_sha256"),
                          ("control.json", "source_control_sha256")]:
            if sha(z.read(name)) != cfg[key]:
                raise ValueError(f"Pinned payload changed: {name}")
        raw = read_json(z.read("raw-scores.json"))
        control = read_json(z.read("control.json"))[cfg["baseline"]]
        reference = read_json(z.read("report.json"))["comparison"]
    arrays = matrices(raw, cfg["cohort_counts"])
    if sum(row["label"] for row in raw["test"]) != 55:
        raise ValueError("Observed test label counts differ")
    return raw, arrays, control, reference


def fit_method(dev, cal, weight):
    """Labels cannot enter this function. Both arrays must be normal scores."""
    if (dev.ndim != 2 or cal.ndim != 2 or dev.shape[1] != 12 or cal.shape[1] != 12
            or not len(dev) or not len(cal) or not np.isfinite(dev).all() or not np.isfinite(cal).all()):
        raise ValueError("Invalid normal arrays")
    center = np.median(dev, axis=0)
    raw_scale = np.quantile(dev, .9, axis=0, method="higher") - center
    if not np.isfinite(raw_scale).all() or (raw_scale <= 1e-6).any():
        raise ValueError("Degenerate raw scale; no floor, dropping or redraw")
    pool = float(np.median(raw_scale))
    scale = (1 - weight) * raw_scale + weight * pool
    threshold = float(np.quantile(((cal - center) / scale).max(axis=1), .95, method="higher"))
    return {"center": center.tolist(), "raw_scale": raw_scale.tolist(), "pool_target": pool,
            "scale": scale.tolist(), "threshold": threshold}


def normalized(x, method):
    return (x - np.array(method["center"])) / np.array(method["scale"])


def counts(arrays, labels, method):
    pred = normalized(arrays["test"], method).max(axis=1) >= method["threshold"]
    return {"tp": int(np.sum(pred & (labels == 1))), "fn": int(np.sum(~pred & (labels == 1))),
            "fp": int(np.sum(pred & (labels == 0))), "tn": int(np.sum(~pred & (labels == 0))),
            "fp_validation": int(np.sum(normalized(arrays["validation"], method).max(axis=1) >= method["threshold"]))}


def replay(arrays, raw, stored, reference, cfg):
    baseline = fit_method(arrays["development"], arrays["calibration"], 0.)
    if [r["region"] for r in stored["regions"]] != REGIONS or stored["field"] != "top1":
        raise ValueError("Stored control contract differs")
    previous = {"center": [r["center"] for r in stored["regions"]],
                "scale": [r["scale"] for r in stored["regions"]], "threshold": stored["threshold"]}
    deltas = [abs(baseline["threshold"] - previous["threshold"])]
    for field in ("center", "scale"):
        deltas.append(float(np.max(np.abs(np.array(baseline[field]) - previous[field]))))
    flips = 0
    for a in arrays.values():
        old, new = normalized(a, previous), normalized(a, baseline)
        deltas.append(float(np.max(np.abs(old - new))))
        flips += int(np.sum((old.max(axis=1) >= previous["threshold"]) != (new.max(axis=1) >= baseline["threshold"])))
    rows = predictions(raw, arrays, baseline)
    test = rows["test"]
    labels = np.array([r["label"] for r in raw["test"]])
    metric = counts(arrays, labels, baseline)
    expected = reference["observed_test"]["control_q90"]
    if ({k: metric[k] for k in ("tp", "fn", "fp", "tn")} != expected["counts"]
            or [r["key"] for r in test if r["label"] == 1 and not r["alert"]] != expected["false_negatives"]
            or [r["key"] for r in test if r["label"] == 0 and r["alert"]] != expected["false_positives"]
            or [r["key"] for r in rows["validation"] if r["alert"]]
            != reference["observed_normals"]["control_q90"]["alert_keys"]):
        raise ValueError("Source decisions/counts not reproduced")
    if max(deltas) > cfg["source_replay_atol"] or flips:
        raise ValueError("Source replay tolerance/decision gate failed")
    return baseline, {"passed": True, "maximum_absolute_difference": max(deltas),
                      "decision_changes": flips, "photos": sum(map(len, raw.values()))}


def predictions(raw, arrays, method):
    result = {}
    for group, a in arrays.items():
        z = normalized(a, method)
        result[group] = [{"key": r["key"], "label": r["label"], "score": float(v.max()),
                          "margin": float(v.max() - method["threshold"]),
                          "alert": bool(v.max() >= method["threshold"]),
                          "winning_region": REGIONS[int(v.argmax())], "region_scores": v.tolist()}
                         for r, v in zip(raw[group], z, strict=True)]
    return result


def trial_indices(scope, cfg, ndev, ncal):
    rng = np.random.Generator(np.random.PCG64(cfg["scope_seeds"][scope]))
    for i in range(cfg["trials_per_scope"]):
        di = rng.integers(0, ndev, ndev) if scope in ("development_only", "both") else np.arange(ndev)
        ci = rng.integers(0, ncal, ncal) if scope in ("calibration_only", "both") else np.arange(ncal)
        yield i, di, ci


def trial(arrays, labels, di, ci):
    record = {"development_indices": di.tolist(), "calibration_indices": ci.tolist(), "methods": {}}
    for name, weight in (("control", 0.), ("candidate", .5)):
        try:
            method = fit_method(arrays["development"][di], arrays["calibration"][ci], weight)
            record["methods"][name] = {"valid": True, "parameters": method, "counts": counts(arrays, labels, method)}
        except ValueError as exc:
            record["methods"][name] = {"valid": False, "error": str(exc)}
    return record


def summarize_trials(records, gate):
    result = {}
    for name in ("control", "candidate"):
        valid = [r["methods"][name]["counts"] for r in records if r["methods"][name]["valid"]]
        invalid = len(records) - len(valid)
        # An invalid trial makes the full-distribution summary undefined; never drop it.
        result[name] = {"invalid_trials": invalid, "quantiles": None if invalid else {
            k: np.quantile([v[k] for v in valid], [.025, .5, .975], method="linear").tolist()
            for k in ("tp", "fp", "fp_validation")}}
    tests = {"no_invalid_trials": all(v["invalid_trials"] == 0 for v in result.values())}
    if tests["no_invalid_trials"]:
        a, b = result["control"]["quantiles"], result["candidate"]["quantiles"]
        widths = {n: v["quantiles"]["tp"][2] - v["quantiles"]["tp"][0] for n, v in result.items()}
        tests.update(tp_width_at_most_half=widths["candidate"] <= gate["maximum_tp_interval_width_ratio_to_paired_control"] * widths["control"],
                     tp_lower=b["tp"][0] >= gate["minimum_bootstrap_tp_q025"],
                     fp_test_upper=b["fp"][2] <= gate["maximum_bootstrap_fp_test_q975"],
                     fp_validation_upper=b["fp_validation"][2] <= gate["maximum_bootstrap_fp_validation_q975"])
        result["tp_interval_widths"] = widths
    return {"methods": result, "conditions": tests, "passed": all(tests.values())}


def publish(path, identity, payload):
    """No overwrite: a repeated result must match its identity and exact contents."""
    envelope = {"identity": identity, "payload_sha256": sha(canonical(payload)), "payload": payload}
    if path.exists():
        if read_json(path.read_bytes()) != envelope:
            raise ValueError(f"Existing evidence changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(envelope)); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)  # Atomic publication; fails if a concurrent target exists.
    finally:
        os.unlink(temporary)


def read_sealed(path, identity):
    value = read_json(path.read_bytes())
    if value["identity"] != identity or value["payload_sha256"] != sha(canonical(value["payload"])):
        raise ValueError(f"Broken evidence seal: {path}")
    return value["payload"]


@contextmanager
def locked(root):
    import fcntl  # This runner is deliberately Mac/Linux CPU-only.
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    raise RuntimeError("Unsupported RSS units")


def check_budget(start, previous_seconds, cfg):
    elapsed = previous_seconds + time.monotonic() - start
    rss = peak_rss_bytes()
    if elapsed > cfg["maximum_elapsed_seconds"] or rss > cfg["maximum_process_peak_rss_bytes"]:
        raise RuntimeError("CPU budget exceeded; preserve partial results")
    return elapsed, rss


def execute(source, output):
    start = time.monotonic()
    cfg = load_config()
    raw, arrays, stored, reference = load_source(source, cfg)
    env = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
           "rss_measurement": "resource.RUSAGE_SELF.ru_maxrss; Darwin bytes, Linux KiB converted to bytes"}
    identity_record = {"config_sha256": CONFIG_SHA, "source_zip_sha256": cfg["source_zip_sha256"],
                       "code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                       "environment": env, "config": cfg,
                       "cohorts": {g: [r["key"] for r in rows] for g, rows in raw.items()}}
    identity = sha(canonical(identity_record))
    with locked(output):
        if not (output / "identity.json").exists() and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Unidentified existing output; preserved")
        publish(output / "identity.json", identity, identity_record)
        old_attempts = [read_sealed(p, identity) for p in sorted((output / "attempts").glob("*.json"))]
        unfinished = [p.name for p in (output / "attempt-starts").glob("*.json")
                      if not (output / "attempts" / p.name).exists()]
        for p in (output / "attempt-starts").glob("*.json"):
            read_sealed(p, identity)
        previous_seconds = sum(v["elapsed_seconds"] for v in old_attempts)
        attempt_id = uuid.uuid4().hex + ".json"
        publish(output / "attempt-starts" / attempt_id, identity, {"started_unix_seconds": time.time()})
        try:
            check_budget(start, previous_seconds, cfg["cpu_budget"])
            control, replay_report = replay(arrays, raw, stored, reference, cfg)
            publish(output / "replay.json", identity, replay_report)
            # Freeze normal-only parameters before opening the candidate's test comparison.
            candidate = fit_method(arrays["development"], arrays["calibration"], .5)
            methods = {"control": control, "candidate": candidate}
            publish(output / "parameters.json", identity, methods)
            pred = {n: predictions(raw, arrays, m) for n, m in methods.items()}
            publish(output / "predictions.json", identity, pred)
            labels = np.array([r["label"] for r in raw["test"]])
            metrics = {n: counts(arrays, labels, m) for n, m in methods.items()}
            changes = {}
            for group in ("test", "validation"):
                changes[group] = {"added_alerts": [], "removed_alerts": []}
                for a, b in zip(pred["control"][group], pred["candidate"][group], strict=True):
                    if a["alert"] != b["alert"]:
                        changes[group]["added_alerts" if b["alert"] else "removed_alerts"].append({"key": a["key"], "label": a["label"]})
            gate = cfg["development_screening"]
            point_conditions = {"minimum_tp": metrics["candidate"]["tp"] >= gate["minimum_tp"],
                "no_lost_tp": not any(r["label"] for r in changes["test"]["removed_alerts"]),
                "fp_test": metrics["candidate"]["fp"] <= gate["maximum_fp_test_normal"],
                "fp_validation": metrics["candidate"]["fp_validation"] <= gate["maximum_fp_validation_normal"]}
            point = {"counts": metrics, "changes": changes, "conditions": point_conditions, "passed": all(point_conditions.values()),
                     "raw_region_boundaries": {n: (np.array(m["center"]) + np.array(m["scale"]) * m["threshold"]).tolist() for n, m in methods.items()}}
            publish(output / "point-comparison.json", identity, point)
            scopes = {}
            for scope in cfg["bootstrap"]["scopes"]:
                records = []
                for i, di, ci in trial_indices(scope, cfg["bootstrap"], len(arrays["development"]), len(arrays["calibration"])):
                    check_budget(start, previous_seconds, cfg["cpu_budget"])
                    record = trial(arrays, labels, di, ci)
                    publish(output / "trials" / scope / f"{i:04d}.json", identity, record)
                    records.append(record)
                    if (i + 1) % 250 == 0:
                        print(f"{scope}: {i+1}/1000", flush=True)
                scopes[scope] = summarize_trials(records, gate)
            elapsed, rss = check_budget(start, previous_seconds, cfg["cpu_budget"])
            peak = max([rss] + [v["peak_rss_bytes"] for v in old_attempts])
            cost_passed = not unfinished and peak <= cfg["cpu_budget"]["maximum_process_peak_rss_bytes"]
            stable = all(v["passed"] for v in scopes.values())
            core = {"status": "exploratory_comparison_complete", "experiment_id": cfg["experiment_id"],
                    "replay": replay_report, "point": point, "stability": scopes,
                    "stability_passed": stable,
                    "decision_without_cost": "research_candidate_only" if point["passed"] and stable else "not_adopted",
                    "training_performed": False, "service_model_changed": False, "automatic_pass_allowed": False,
                    "independent_performance_claim_allowed": False}
            report_path = output / "report.json"
            if report_path.exists():
                report = read_sealed(report_path, identity)
                if report["results"] != core:
                    raise ValueError("Completed summary differs from reconstructed evidence")
            else:
                report = {"results": core, "research_screening_passed": point["passed"] and stable and cost_passed,
                          "decision": "research_candidate_only" if point["passed"] and stable and cost_passed else "not_adopted",
                          "cost": {"passed": cost_passed, "unfinished_prior_attempts": unfinished,
                          "cumulative_elapsed_seconds": elapsed if not unfinished else None,
                          "measured_elapsed_seconds": elapsed, "peak_rss_bytes": peak,
                          "scope": "source verification through 3000 persisted trials and summary; excludes final report/receipt serialization; not inference latency"}}
                publish(report_path, identity, report)
            return report
        except BaseException as exc:
            publish(output / "failures" / f"{uuid.uuid4().hex}.json", identity,
                    {"error": f"{type(exc).__name__}: {exc}", "automatic_pass_allowed": False})
            raise
        finally:
            publish(output / "attempts" / attempt_id, identity,
                    {"elapsed_seconds": time.monotonic() - start, "peak_rss_bytes": peak_rss_bytes()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/analyses/mpi-scale-shrinkage-v1")
    args = parser.parse_args()
    report = execute(args.source, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
