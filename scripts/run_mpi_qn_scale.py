"""MPI-C2-01: one frozen CPU-only Qn comparison, no images or model loading."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mpi_scale_shrinkage as common

CONFIG = ROOT / "configs/mpi-qn-scale-v1.json"
CONFIG_SHA = "bb80eb3824c27fad250c4dc840ea35f60b7a044d18c1e573af13dcc1fa104b46"
SOURCE_FILES = ("scripts/run_mpi_qn_scale.py", "scripts/run_mpi_scale_shrinkage.py",
                "configs/mpi-qn-scale-v1.json", "tests/test_mpi_qn_scale.py")


def load_config(path=CONFIG):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if common.sha(data) != CONFIG_SHA:
        raise ValueError("Frozen Qn configuration changed")
    return common.read_json(data)


def qn_scale(x):
    """Unscaled kth pair distance; indices are draw positions, not unique IDs."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid Qn input")
    h = len(x) // 2 + 1
    k = h * (h - 1) // 2 - 1
    i, j = np.triu_indices(len(x), 1)
    distances = np.abs(x[i] - x[j])
    scale = float(np.partition(distances, k)[k])
    if not np.isfinite(distances).all() or not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Degenerate Qn scale; no floor, drop or redraw")
    return scale


def fit_qn(dev, cal):
    dev, cal = np.asarray(dev, dtype=np.float64), np.asarray(cal, dtype=np.float64)
    if dev.shape != (128, 12) or cal.shape != (169, 12):
        raise ValueError("Qn requires frozen 128/169 by 12 normal matrices")
    if not np.isfinite(dev).all() or not np.isfinite(cal).all():
        raise ValueError("Nonfinite normal scores")
    center = np.median(dev, axis=0)
    scale = np.array([qn_scale(dev[:, r]) for r in range(12)])
    threshold = float(np.quantile(((cal-center)/scale).max(axis=1), .95, method="higher"))
    if not np.isfinite(threshold):
        raise ValueError("Nonfinite threshold")
    return {"center": center.tolist(), "scale": scale.tolist(), "threshold": threshold}


def load_source(path, cfg):
    if common.sha(path.read_bytes()) != cfg["source_zip_sha256"]:
        raise ValueError("Pinned source ZIP differs")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        manifest = common.read_json(z.read("handoff-manifest.json"))
        if len(names) != len(set(names)) or set(names) != set(manifest) | {"handoff-manifest.json"} or z.testzip():
            raise ValueError("Source inventory/CRC differs")
        for name, sha in manifest.items():
            if common.sha(z.read(name)) != sha:
                raise ValueError("Source payload checksum differs")
        for name, key in (("raw-scores.json", "source_raw_scores_sha256"),
                          ("freeze.json", "source_freeze_sha256")):
            if common.sha(z.read(name)) != cfg[key]:
                raise ValueError("Pinned payload differs")
        raw, freeze, report = [common.read_json(z.read(n)) for n in
                               ("raw-scores.json", "freeze.json", "report.json")]
    arrays = common.matrices(raw, cfg["cohort_counts"])
    if sum(r["label"] for r in raw["test"]) != 55:
        raise ValueError("Test labels differ")
    method = freeze["contract"][cfg["baseline"]]
    reference = {"observed_test": {"control_q90": freeze["observed_test_reference"][cfg["baseline"]]},
                 "observed_normals": {"control_q90": report["normal_validation"]["methods"][cfg["baseline"]]}}
    return raw, arrays, method, reference


def trial(arrays, labels, di, ci):
    record = {"development_indices": di.tolist(), "calibration_indices": ci.tolist(), "methods": {}}
    for name, fit in (("control", lambda d, c: common.fit_method(d, c, 0.)), ("candidate", fit_qn)):
        try:
            method = fit(arrays["development"][di], arrays["calibration"][ci])
            record["methods"][name] = {"valid": True, "parameters": method,
                                      "counts": common.counts(arrays, labels, method)}
        except ValueError as exc:
            record["methods"][name] = {"valid": False, "error": str(exc)}
    return record


def compare(raw, arrays, methods, gate):
    predictions = {name: common.predictions(raw, arrays, m) for name, m in methods.items()}
    labels = np.array([r["label"] for r in raw["test"]])
    metrics = {name: common.counts(arrays, labels, m) for name, m in methods.items()}
    changes = {}
    for group in ("test", "validation"):
        changes[group] = {"added_alerts": [], "removed_alerts": []}
        for a, b in zip(predictions["control"][group], predictions["candidate"][group], strict=True):
            if a["alert"] != b["alert"]:
                changes[group]["added_alerts" if b["alert"] else "removed_alerts"].append(
                    {"key": a["key"], "label": a["label"]})
    conditions = {"minimum_tp": metrics["candidate"]["tp"] >= gate["minimum_tp"],
                  "no_lost_tp": not any(r["label"] for r in changes["test"]["removed_alerts"]),
                  "fp_test": metrics["candidate"]["fp"] <= gate["maximum_fp_test_normal"],
                  "fp_validation": metrics["candidate"]["fp_validation"] <= gate["maximum_fp_validation_normal"]}
    return {"counts": metrics, "changes": changes, "conditions": conditions,
            "passed": all(conditions.values()), "raw_region_boundaries": {
                n: (np.array(m["center"]) + np.array(m["scale"])*m["threshold"]).tolist()
                for n, m in methods.items()}}, predictions


def source_identity(cfg):
    """Require a committed source snapshot before any candidate data calculation."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {}
    for name in SOURCE_FILES:
        actual = (ROOT/name).read_bytes().replace(b"\r\n", b"\n")
        committed = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT).replace(b"\r\n", b"\n")
        if actual != committed:
            raise ValueError(f"Execution source not committed: {name}")
        hashes[name] = common.sha(actual)
    if platform.python_version() != "3.11.9" or np.__version__ != "2.3.3" or sys.platform != "darwin":
        raise ValueError("Execution environment differs from fixed Mac CPU plan")
    # Unrelated documentation commits do not invalidate resume; relevant bytes do.
    return {"config": cfg, "source_lf_sha256": hashes, "environment": {
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
        "machine": platform.machine()}, "authorization": "2026-10-03 user approved implementation and CPU comparison"}, commit


def attempt_history(output, identity):
    starts = {p.name: common.read_sealed(p, identity) for p in (output/"attempt-starts").glob("*.json")}
    ends = {p.name: common.read_sealed(p, identity) for p in (output/"attempts").glob("*.json")}
    if starts.keys() != ends.keys():
        raise RuntimeError("Unfinished or unmatched attempt; unknown cost blocks automatic resume")
    if any(v["budget_exceeded"] for v in ends.values()):
        raise RuntimeError("Prior resource violation blocks automatic resume")
    return sum(v["elapsed_seconds"] for v in ends.values()), max([0]+[v["peak_rss_bytes"] for v in ends.values()])


def budget(start, previous, cfg):
    elapsed = previous + time.monotonic() - start
    rss = common.peak_rss_bytes()
    if elapsed > cfg["maximum_cumulative_elapsed_seconds"] or rss > cfg["maximum_process_peak_rss_bytes"]:
        raise RuntimeError("Qn CPU resource budget exceeded; no automatic shrinking")
    return elapsed, rss


def reuse_trial(path, identity, arrays, labels, di, ci):
    if path.exists():
        record = common.read_sealed(path, identity)
        if record["development_indices"] != di.tolist() or record["calibration_indices"] != ci.tolist():
            raise ValueError("Checkpoint draw indices differ")
        return record
    record = trial(arrays, labels, di, ci)
    common.publish(path, identity, record)
    return record


def execute(source, output):
    start = time.monotonic()
    cfg = load_config()
    info, commit = source_identity(cfg)
    identity = common.sha(common.canonical(info))
    with common.locked(output):
        if not (output/"identity.json").exists() and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Unidentified output preserved")
        common.publish(output/"identity.json", identity, info)
        previous, previous_peak = attempt_history(output, identity)
        attempt = uuid.uuid4().hex + ".json"
        common.publish(output/"attempt-starts"/attempt, identity,
                       {"started_unix_seconds": time.time(), "source_commit": commit})
        error = None
        try:
            budget(start, previous, cfg["cpu_budget"])
            raw, arrays, stored, ref = load_source(source, cfg)
            control, replay = common.replay(arrays, raw, stored, ref, cfg)
            common.publish(output/"replay.json", identity, replay)
            print("Baseline replay verified; freezing normal-only Qn parameters.", flush=True)
            # Persist normal-only parameters before computing candidate test metrics.
            methods = {"control": control, "candidate": fit_qn(arrays["development"], arrays["calibration"])}
            common.publish(output/"parameters.json", identity, methods)
            point, predictions = compare(raw, arrays, methods, cfg["development_screening"])
            common.publish(output/"predictions.json", identity, predictions)
            common.publish(output/"point-comparison.json", identity, point)
            print("Fixed comparison saved; running all 3000 paired trials regardless of point gate.", flush=True)
            labels = np.array([r["label"] for r in raw["test"]])
            scopes, trial_hashes = {}, {}
            for scope in cfg["bootstrap"]["scopes"]:
                records = []
                for i, di, ci in common.trial_indices(scope, cfg["bootstrap"], 128, 169):
                    budget(start, previous, cfg["cpu_budget"])
                    path = output/"trials"/scope/f"{i:04d}.json"
                    record = reuse_trial(path, identity, arrays, labels, di, ci)
                    trial_hashes[f"{scope}/{i:04d}"] = common.sha(common.canonical(record))
                    records.append(record)
                    if (i+1) % 250 == 0:
                        print(f"{scope}: {i+1}/1000; cumulative {previous+time.monotonic()-start:.1f}s", flush=True)
                scopes[scope] = common.summarize_trials(records, cfg["development_screening"])
                common.publish(output/f"summary-{scope}.json", identity, scopes[scope])
            common.publish(output/"trial-hashes.json", identity, trial_hashes)
            budget(start, previous, cfg["cpu_budget"])
            results = {"replay": replay, "point": point, "stability": scopes,
                       "stability_passed": all(s["passed"] for s in scopes.values()),
                       "training_performed": False, "service_model_changed": False,
                       "automatic_pass_allowed": False, "independent_performance_claim_allowed": False}
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed, rss = time.monotonic()-start, max(previous_peak, common.peak_rss_bytes())
            exceeded = (previous+elapsed > cfg["cpu_budget"]["maximum_cumulative_elapsed_seconds"] or
                        rss > cfg["cpu_budget"]["maximum_process_peak_rss_bytes"])
            common.publish(output/"attempts"/attempt, identity, {"elapsed_seconds": elapsed,
                "cumulative_seconds": previous+elapsed, "peak_rss_bytes": rss,
                "budget_exceeded": exceeded, "error": error})
        # Cost includes all intermediate writes and the completed calculation; not final publication.
        if exceeded:
            raise RuntimeError("Final resource budget exceeded; partial outputs preserved")
        passed = point["passed"] and results["stability_passed"]
        report = {"status": "exploratory_comparison_complete", "results": results,
                  "decision": "research_candidate_only" if passed else "not_adopted",
                  "cost": {"passed": True, "cumulative_elapsed_seconds": previous+elapsed, "peak_rss_bytes": rss},
                  "experiment_id": cfg["experiment_id"], "identity": identity}
        if (output/"report.json").exists():
            old = common.read_sealed(output/"report.json", identity)
            if old["results"] != results:
                raise ValueError("Reconstructed summary differs")
            return old
        common.publish(output/"report.json", identity, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/analyses/mpi-qn-scale-v1")
    args = parser.parse_args()
    try:
        report = execute(args.source, args.output)
        print(common.canonical(report).decode())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Files preserved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
