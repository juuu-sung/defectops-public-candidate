"""Fixed raw-feature nearest-three candidate: normal calibration then small regression gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_mpi_unit_feature as cache
from scripts.mpi_knn3_screen_runtime import MeanThreeRuntime

fit, ev, comparison, inputs, evaluation = cache.fit, cache.ev, cache.comparison, cache.inputs, cache.evaluation
scoring = comparison.c.scoring
NAME = "mpi-knn3-screen-v1"
POLICY_SHA256 = "d51a092bea2b1aeed2519f50a45d7d78cbd37e030a6b81dc14689c8eb8160a7c"


def select_regression(source, policy):
    method = source["contract"]["candidate_top1_q90"]
    positives = [p for p in source["reference"]["test"] if p["label"] == 1 and scoring.score(p, method)["alert"]]
    positives.sort(key=lambda p: hashlib.sha256((policy["selection_salt"] + "\0" + p["key"]).encode()).hexdigest())
    keys = [p["key"] for p in positives[:12]]
    if len(positives) != 53 or keys != policy["regression_keys"]:
        raise ValueError("Frozen regression selection differs")
    by_key = {p["key"]: p for p in source["cohorts"]["test"]}
    return [by_key[k] for k in keys]


def prepare(project, handoffs=None):
    source = inputs.prepare(project, handoffs or project / "artifacts/handoff")
    policy = json.loads((project / "configs" / f"{NAME}.json").read_bytes())
    if fit.digest(policy) != POLICY_SHA256:
        raise ValueError("Frozen screening policy changed")
    comparison.validate_raw(source["reference"], source["cohorts"])
    baseline = source["contract"]["candidate_top1_q90"]
    replay = comparison.contract_replay(source["reference"], baseline)
    if comparison.counts(comparison.matrix(source["reference"]), baseline,
                         [p["label"] for p in source["reference"]["test"]]) != {
                             "tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14}:
        raise ValueError("Reviewed baseline changed")
    groups = {g: source["cohorts"][g] for g in ("development", "calibration")}
    groups["regression"] = select_regression(source, policy)
    groups["reload"] = groups["calibration"][:8] + groups["regression"]
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project / "configs" / f"{NAME}.json", project / "docs" / f"{NAME}.md"])
    identity = {"name": NAME, "policy": policy, "source": source["identity"],
                "checkpoint_sha256": ev.CHECKPOINT_SHA256, "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
                "baseline_replay": replay, "groups": groups,
                "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}
    return source, policy, identity


def replay_check(actual, expected, contract=None):
    scoring.old.validate_sample(actual, expected)
    delta = max(abs(a[k] - b[k]) for a, b in zip(actual["regions"], expected["regions"], strict=True)
                for k in ("native_raw", "top1"))
    flips = (contract is not None and scoring.score(actual, contract)["alert"] != scoring.score(expected, contract)["alert"])
    if delta > 1e-4 or flips:
        raise ValueError(f"Raw replay/reload differs: {delta}; decision flip={flips}")
    return delta


def score_photo(archive, photo, expected, runtime, recovered, monitor, baseline_contract):
    sample = {k: photo[k] for k in ("key", "photo_id", "label")}
    sample["regions"] = []
    control = {**sample, "regions": []}
    runtime.synchronize()
    started = time.perf_counter()
    for row in photo["regions"]:
        monitor.check()
        image = (ev.read_tile(archive, row, recovered) if photo["key"].startswith("defect/")
                 else fit.checked_image(archive, row))
        try:
            native, baseline_map, candidate_map = runtime.predict_pair(image)
            for result, values in ((control, baseline_map), (sample, candidate_map)):
                values = np.asarray(values)
                if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Invalid anomaly map")
                result["regions"].append({"region": row["region"], "member": row["member"],
                    "native_raw": float(native), "top1": float(fit.candidate_image_score(native, values, fit.CANDIDATE))})
        finally:
            image.close()
        monitor.check()
    replay_check(control, expected, baseline_contract)
    scoring.old.validate_sample(sample, photo)
    runtime.synchronize()
    sample["baseline_replay"] = control
    sample["photo_scoring_seconds"] = time.perf_counter() - started
    return sample


def regression_gate(samples, contract, keys):
    if [p["key"] for p in samples] != keys or len(samples) != 12 or any(p["label"] != 1 for p in samples):
        raise ValueError("Regression inventory changed")
    predictions = comparison.predictions({"regression": samples}, contract)["regression"]
    lost = [p["key"] for p in predictions if not p["alert"]]
    return {"passed": not lost, "retained": 12-len(lost), "selected_prior_detections": 12,
            "lost_keys": lost, "predictions": predictions, "performance_estimate": False}


def execute(root, archive, checkpoint, recovery_path, source, policy, identity, handoff,
            runtime_factory=MeanThreeRuntime, monitor_factory=evaluation.Monitor):
    signature = fit.digest(identity)
    groups = identity["groups"]
    environment = source["records"]["reload.json"]["receipt"]["runtime"]
    baseline = source["contract"]["candidate_top1_q90"]
    references = {p["key"]: p for samples in source["reference"].values() for p in samples}
    with fit.locked_run(root):
        if not (root / "identity").exists() and any((root / g).exists() for g in (*groups, "result", "transform")):
            raise ValueError("Outputs without identity; preserved")
        inputs.v.publish(root, "identity", signature, {"plan.json": identity})
        cached = {}
        for group, photos in groups.items():
            (root / group).mkdir(exist_ok=True)
            for i, photo in enumerate(photos):
                if (root / group / f"photo-{i:03d}").exists():
                    sample = cache.read_sample(root, group, i, signature, photo, environment)
                    replay_check(sample["baseline_replay"], references[photo["key"]], baseline)
                    cached[group, i] = sample
        normal_complete = all((g, i) in cached for g in ("development", "calibration") for i in range(len(groups[g])))
        if (root / "calibrated").exists() and not normal_complete:
            raise ValueError("Calibration without full normal evidence")
        if any(g == "regression" for g, _ in cached) and not (root / "calibrated").exists():
            raise ValueError("Regression before normal calibration")
        if any(g == "reload" for g, _ in cached) and not (root / "screen-gate").exists():
            raise ValueError("Reload before screening gate")
        names = ["report.json", "scores.json", "contract.json", "transform.json", "plan.json", "attempts.json",
                 "sample-receipts.json", "runtime-gates.json", "handoff-manifest.json"]
        if (root / "result").exists():
            fit.read_phase(root, "result", signature, names)
            report = json.loads((root / "result/report.json").read_bytes())
            if len(cached) != report["scored_photos"]:
                raise ValueError("Completed report missing score caches")
            inputs.v.probe.export(root / "result", handoff)
            return report
        attempts = evaluation.attempts(root)
        if any(a.get("resource_violation") for a in attempts):
            raise ValueError("Prior resource violation requires review")
        prior = sum(a["seconds"] for a in attempts)
        if prior >= policy["resource_plan"]["comparison_seconds"]:
            raise ValueError("Cumulative time cap exceeded")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root / "attempts"))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        runtime = monitor = None
        failure = None
        started = time.perf_counter()
        try:
            runtime = runtime_factory(source["config"])
            with monitor_factory(runtime, prior, started, policy) as monitor:
                def load():
                    runtime.load(checkpoint, source["records"]["fit.json"]["checkpoint_context"])
                    if runtime.environment() != environment:
                        raise ValueError("GPU environment differs from baseline")
                    monitor.check()
                    gate = ev.normal_probe(archive, source["preparation"], source["records"]["reload.json"]["photos"], runtime)
                    if not gate["passed"]:
                        raise ValueError("Original normal replay failed")
                    monitor.check()
                    transform = runtime.activate()
                    inputs.v.publish(root, "transform", signature, {"transform.json": transform})
                    gate.update(environment=environment, transform=transform)
                    path = attempt / f"gate-{time.time_ns()}.json"
                    fit.write_json(path, gate)
                    return path

                gate_path = load()
                recovered = None

                def collect(group):
                    for i, photo in enumerate(groups[group]):
                        if (group, i) not in cached:
                            sample = score_photo(archive, photo, references[photo["key"]], runtime, recovered, monitor, baseline)
                            with fit.phase_stage(root / group, f"photo-{i:03d}") as stage:
                                fit.write_json(stage / "sample.json", sample)
                                fit.seal_phase(stage, f"photo-{i:03d}", signature,
                                    {"photo_sha256": fit.digest(photo), "environment": environment,
                                     "gate_path": gate_path.relative_to(root).as_posix(),
                                     "gate_sha256": fit.sha256_file(gate_path)}, ["sample.json"])
                            cached[group, i] = cache.read_sample(root, group, i, signature, photo, environment)
                        if (i+1) % 10 == 0 or i+1 == len(groups[group]):
                            print(f"{group} {i+1}/{len(groups[group])}", flush=True)
                    return [cached[group, i] for i in range(len(groups[group]))]

                print("1/3 Recalibrate unchanged Q90 recipe using all 297 designated normals.", flush=True)
                raw = {g: collect(g) for g in ("development", "calibration")}
                contract = comparison.c.calibrate(**raw)
                inputs.v.publish(root, "calibrated", signature, {"contract.json": contract})
                recovered = ev.recovered_bytes(recovery_path, source["previous"])
                print("2/3 Only 12 fixed previously detected defects. No full evaluation or masks.", flush=True)
                raw["regression"] = collect("regression")
                gate = regression_gate(raw["regression"], contract, policy["regression_keys"])
                inputs.v.publish(root, "screen-gate", signature, {"gate.json": gate})
                reload = {"status": "not_run_screen_failed", "passed": False}
                raw["reload"] = []
                if gate["passed"]:
                    print("3/3 Reload and verify 8 fixed normals plus the same 12 defect cases.", flush=True)
                    gate_path = load()
                    raw["reload"] = collect("reload")
                    by_key = {p["key"]: p for g in ("development", "calibration", "regression") for p in raw[g]}
                    maximum = max(replay_check(p, by_key[p["key"]], contract) for p in raw["reload"])
                    reload = {"status": "small_reload_passed", "passed": True, "photos": 20,
                              "maximum_raw_difference": maximum, "decision_flips": 0,
                              "full_297_normal_reload_completed": False}
                elif any(g == "reload" for g, _ in cached):
                    raise ValueError("Reload evidence after failed screening gate")
                monitor.check()
                runtime.activate()  # recheck raw bank fingerprint without changing it
                if fit.sha256_file(checkpoint) != identity["checkpoint_sha256"]:
                    raise ValueError("Original checkpoint changed")
                monitor.check()
                report = {"status": "mean_three_small_regression_completed", "screen": gate, "reload": reload,
                    "decision": "eligible_for_full_observed_comparison_review" if gate["passed"] else "stop_candidate",
                    "scored_photos": len(cached), "normal_calibration_photos": 297,
                    "test_masks_decoded": 0, "full_evaluation_run": False, "new_training": False,
                    "automatic_pass_allowed": False, "service_model_changed": False,
                    "independent_performance_claim_allowed": False,
                    "false_positive_rate_evaluated": False, "stability_evaluated": False}
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if runtime is not None:
                    runtime.release()
            except BaseException as exc:
                failure = failure or f"Release failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter() - started
                if failure is None and prior + elapsed > policy["resource_plan"]["comparison_seconds"]:
                    failure = "Resource cap exceeded including release"
                fit.write_json(attempt / "finished.json", {"seconds": elapsed, "error": failure,
                    "resource_violation": bool(failure and any(s in failure for s in ("cap exceeded", "OutOfMemory", "out of memory", "monitor failed", "sampler failed"))),
                    "memory": monitor.snapshot() if monitor is not None else None,
                    "scope": "load, original replay, dual scoring, cache I/O, small reload, release; excludes input hashing/export",
                    "completed_cached_photos": len(cached)})
        if failure:
            raise RuntimeError(failure)
        attempts = evaluation.attempts(root)
        report.update(cumulative_seconds=sum(a["seconds"] for a in attempts), attempts_with_errors=sum(bool(a["error"]) for a in attempts))
        receipts = {f"{g}/photo-{i:03d}": fit.read_phase(root / g, f"photo-{i:03d}", signature, ["sample.json"]) for g, i in cached}
        payloads = {"report.json": report, "scores.json": raw, "contract.json": contract, "plan.json": identity,
                    "transform.json": json.loads((root / "transform/transform.json").read_bytes()),
                    "attempts.json": attempts, "sample-receipts.json": receipts,
                    "runtime-gates.json": {r["gate_path"]: json.loads((root / r["gate_path"]).read_bytes()) for r in receipts.values()}}
        payloads["handoff-manifest.json"] = {n: hashlib.sha256(fit.canonical(v)).hexdigest() for n, v in payloads.items()}
        inputs.v.publish(root, "result", signature, payloads)
        inputs.v.probe.export(root / "result", handoff)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        source, policy, identity = prepare(project)
        checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
            raise ValueError("Original checkpoint/archive changed")
        with zipfile.ZipFile(args.archive) as archive:
            report = execute(project / "artifacts/analyses" / NAME, archive, checkpoint,
                project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
                source, policy, identity, project / "artifacts/handoff" / f"{NAME}.zip")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Files preserved; do not reroll or tune.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
