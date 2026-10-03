"""Final EfficientAD: fixed 12-case screen, observed comparison, warm cost, normal sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import mpi_efficientad_evaluation_protocol as e
from scripts import mpi_efficientad_fit_protocol as p
from scripts.mpi_efficientad_evaluation_runtime import Runtime, environment
from scripts.run_mpi_efficientad_probe import publish, attempts
from scripts.evaluate_mpi_normal_diversity import Monitor as SharedMonitor, EvaluationRuntime

fit, ev, comparison = e.fit, e.inputs.ev, e.comparison
V1_RUNNER_SHA = "f5e936b8335ece2fcfbb77dcefbfaaa73f436732d2c5c539ea5b6db063038f90"
V1_ERROR = "TypeError: object supporting the buffer API required"


def reviewed_v1_failure(root, identity):
    """Read-only, narrowly scoped restart approval; never erase cost or reuse old predictions."""
    if not root.exists():
        return {"present": False, "seconds": 0.}
    # This persistent file remains after normal process exit; existence is not an active lock.
    if (root / ".run.lock").read_bytes() != b"0":
        raise ValueError("Previous evaluation lock marker differs; request review")
    with fit.locked_run(root):
        return _reviewed_v1_failure(root, identity)


def _reviewed_v1_failure(root, identity):
    old = json.loads((root / "identity/value.json").read_bytes())
    signature = fit.digest(old)
    if e.value(root, "identity", signature) != old or old.get("name") != "mpi-efficientad-evaluation-v1":
        raise ValueError("Unexpected previous evaluation identity")
    for key in ("policy", "fit_zip_sha256", "fit_signature", "checkpoint_sha256", "cohorts", "contract", "environment"):
        if old[key] != identity[key]:
            raise ValueError(f"Previous evaluation {key} changed; review required")
    if old["sources_lf_sha256"].get("scripts/run_mpi_efficientad_evaluation.py") != V1_RUNNER_SHA:
        raise ValueError("Previous failure was not from the reviewed v1 implementation")
    paths = sorted(root.glob("attempt-*"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one reviewed v1 failure; preserve attempts for review")
    path = paths[0]
    record = json.loads((path / "finished.json").read_bytes())
    started = json.loads((path / "started.json").read_bytes())
    seconds = record["seconds"]
    if (started["identity"] != signature or record["status"] != "failed" or record["error"] != V1_ERROR
            or not np.isfinite(seconds) or seconds < 0 or record["cached_photos"] != 4):
        raise ValueError("Previous failure differs from the reviewed 0054 buffer error")
    if any((root / phase).exists() for phase in ("screen", "comparison", "warm-baseline", "warm-candidate", "stability", "result", "export")):
        raise ValueError("Previous v1 progressed beyond the reviewed failure")
    photos = old["cohorts"]["test"]
    selected = e.POLICY["acceptance"]["regression_keys"][:4]
    indices = [next(i for i, photo in enumerate(photos) if photo["key"] == key) for key in selected]
    if ({f.name for f in (root / "test").iterdir()} != {f"photo-{i:03d}" for i in indices}
            or any((root / "validation").iterdir())):
        raise ValueError("Previous v1 cache differs from the first four fixed cases")
    for i in indices:
        read_cache(root, "test", i, signature, photos[i], old["contract"])
    return {"present": True, "name": root.name, "identity_sha256": signature,
            "seconds": float(seconds), "attempt": record, "predictions_reused": False,
            "files_sha256": {f.relative_to(root).as_posix(): fit.sha256_file(f)
                             for f in sorted(root.rglob("*")) if f.is_file() and f.name != ".run.lock"}}


def recovery_for_photo(photo, recovered, path, previous):
    """0054 is the fifth fixed screen case. Load only its previously approved recovery."""
    if any(row["member"] == ev.REPAIR_MEMBER for row in photo["regions"]) and recovered is None:
        recovered = ev.recovered_bytes(path, previous)
        if not isinstance(recovered, bytes):
            raise ValueError("Approved 0054 recovery did not return image bytes")
    return recovered


class Monitor(SharedMonitor):
    def sample(self):
        # A stable local reference avoids a race while sequential GPU runtimes are released.
        with self.lock:
            self.rss = max(self.rss, self.process.memory_info().rss)
            runtime = self.runtime
            if runtime is not None:
                free, total = runtime.torch.cuda.mem_get_info()
                self.device = max(self.device, total-free)
            self.samples += 1


def score_photo(archive, photo, runtime, recovered, monitor, contract, check_tiles=True):
    monitor.check()
    runtime.synchronize()
    started = time.perf_counter()
    maps = []
    for row in photo["regions"]:
        if check_tiles:
            monitor.check()
        image = (ev.read_tile(archive, row, recovered) if photo["key"].startswith(("good/", "defect/"))
                 else fit.checked_image(archive, row))
        try:
            maps.append(p.validate_maps(runtime.predict(image)))
        finally:
            image.close()
    raw = np.stack(maps)
    result = e.sample(photo, p.tile_scores(raw, contract["map_quantiles"]), contract)
    runtime.synchronize()
    result["photo_seconds"] = time.perf_counter() - started
    monitor.check()
    return raw, result


def normal_gate(archive, runtime, monitor, records, source, fit_root):
    contract = records["normal-results.json"]["contract"]
    maximum_map = maximum_score = 0.
    for i, photo in enumerate(source["cohorts"]["calibration"][:8]):
        expected = np.load(fit_root / "calibration" / f"photo-{i:03d}/maps.npy", allow_pickle=False)
        raw, sample = score_photo(archive, photo, runtime, None, monitor, contract)
        maximum_map = max(maximum_map, float(np.abs(raw - expected).max()))
        previous = e.sample(photo, records["normal-results.json"]["tile_scores"]["calibration"][i], contract)
        maximum_score = max(maximum_score, abs(sample["score"] - previous["score"]))
        if maximum_map > 1e-4 or maximum_score > 1e-4 or sample["alert"] != previous["alert"]:
            raise ValueError("Final model normal replay failed; no test access")
    return {"passed": True, "photos": 8, "maximum_raw_map_difference": maximum_map,
            "maximum_score_difference": maximum_score, "decision_changes": 0, "contract_refitted": False}


def read_cache(root, group, index, signature, photo, contract):
    name = f"photo-{index:03d}"
    receipt = fit.read_phase(root / group, name, signature, ["maps.npy", "sample.json"])
    if receipt["photo_sha256"] != fit.digest(photo):
        raise ValueError("Cached photo identity differs")
    gate_path = (root / receipt["gate_path"]).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt["gate_sha256"] or not json.loads(gate_path.read_bytes())["passed"]:
        raise ValueError("Cached sample runtime gate changed")
    maps = p.validate_maps(np.load(root / group / name / "maps.npy", allow_pickle=False))
    if maps.shape != (12, 2, 256, 256):
        raise ValueError("Cached photo map inventory differs")
    sample = json.loads((root / group / name / "sample.json").read_bytes())
    expected = e.sample(photo, p.tile_scores(maps, contract["map_quantiles"]), contract)
    if any(sample[k] != v for k, v in expected.items()) or not np.isfinite(sample["photo_seconds"]) or sample["photo_seconds"] < 0:
        raise ValueError("Cached signed scores differ")
    return sample


def benchmark(kind, runtime, archive, source, records, contract, monitor):
    photos = source["cohorts"]["calibration"][:e.POLICY["warm_photos"]]
    expected = source["reference"]["calibration"]
    baseline_contract = source["contract"]["candidate_top1_q90"]

    def once(i):
        if kind == "candidate":
            # Both warm timers exclude monitor checks; the monitor still samples in its thread.
            _, sample = score_photo(archive, photos[i], runtime, None, monitor, contract, check_tiles=False)
            prior = e.sample(photos[i], records["normal-results.json"]["tile_scores"]["calibration"][i], contract)
            if abs(sample["score"] - prior["score"]) > 1e-4 or sample["alert"] != prior["alert"]:
                raise ValueError("Warm EfficientAD decision drift")
            return sample["photo_seconds"]
        from scripts.evaluate_mpi_normal_diversity import score_photo as baseline_score, replay
        monitor.check()
        sample, _, seconds = baseline_score(archive, photos[i], runtime, baseline_contract, None)
        replay(sample, expected[i], baseline_contract)
        monitor.check()
        return seconds

    for i in range(e.POLICY["warmup_photos"]):
        once(i)
    seconds = [once(i) for _ in range(e.POLICY["warm_repeats"]) for i in range(len(photos))]
    q = float(np.quantile(seconds, .95, method="higher"))
    return {"model": kind, "photo_keys": [s["key"] for s in photos], "seconds": seconds,
            "mean_seconds": float(np.mean(seconds)), "p95_higher_seconds": q,
            "memory": runtime.memory(), "scope": e.POLICY["warm_scope"],
            "candidate_budget_passed": q <= e.POLICY["acceptance"]["warm_photo_p95_seconds"] if kind == "candidate" else None}


def stability(root, signature, fit_root, source, records, monitor):
    groups = source["cohorts"]
    arrays = {g: [np.load((fit_root if g in ("development", "calibration") else root) / g /
                          f"photo-{i:03d}/maps.npy", mmap_mode="r", allow_pickle=False)
                  for i in range(len(photos))] for g, photos in groups.items()}
    baseline = comparison.matrix(source["reference"])
    labels = [s["label"] for s in groups["test"]]
    frozen = records["normal-results.json"]
    results = {g: [] for g in e.POLICY["acceptance"]["stability"]["scopes"]}
    directory = root / "stability"
    directory.mkdir(exist_ok=True)
    for scope, i, di, ci in e.draws():
        monitor.check()
        name = f"{scope}-{i:03d}"
        if (directory / name).exists():
            trial = e.value(directory, name, signature)
            if trial["development_indices"] != di.tolist() or trial["calibration_indices"] != ci.tolist():
                raise ValueError("Resampling plan changed")
        else:
            trial = {"scope": scope, "index": i, "development_indices": di.tolist(), "calibration_indices": ci.tolist()}
            for model in ("baseline", "candidate"):
                try:
                    if model == "baseline":
                        method = comparison.fit_method(baseline["development"][di], baseline["calibration"][ci])
                        trial[model] = {"valid": True, "contract": method,
                                        "counts": comparison.counts(baseline, method, labels)}
                    else:
                        # Calibration-only draws leave development maps/quantiles exactly unchanged.
                        q = (frozen["contract"]["map_quantiles"] if scope == "calibration_only" else
                             p.map_quantiles([arrays["development"][j] for j in di], monitor.check))
                        tiles = {}
                        for g in groups:
                            if scope == "calibration_only" and g in frozen["tile_scores"]:
                                tiles[g] = np.asarray(frozen["tile_scores"][g])
                            else:
                                values = []
                                for raw in arrays[g]:
                                    monitor.check()
                                    values.append(p.tile_scores(raw, q))
                                tiles[g] = np.asarray(values)
                        method = p.calibrate(tiles["development"][di], tiles["calibration"][ci], q)
                        samples = {g: [e.sample(photo, a, method) for photo, a in zip(groups[g], tiles[g], strict=True)]
                                   for g in ("test", "validation")}
                        trial[model] = {"valid": True, "contract": method,
                                        "counts": e.metrics(samples["test"], samples["validation"])}
                except ValueError as exc:
                    if not str(exc).startswith("Degenerate"):
                        raise
                    trial[model] = {"valid": False, "error": str(exc)}
            trial["valid"] = all(trial[m]["valid"] for m in ("baseline", "candidate"))
            publish(directory, name, signature, trial)
        results[scope].append(trial)
        monitor.check()
        print(f"stability {scope} {i+1}/200 saved (normal resampling, no retraining)", flush=True)
    summaries = {g: e.summarize(values) for g, values in results.items()}
    return {"trials": results, "summary": summaries, "passed": all(s["passed"] for s in summaries.values()),
            "interpretation": "Observed-photo normal-calibration sensitivity, not population confidence intervals"}


def export(root, handoff, signature, report, identity):
    samples = {g: [json.loads(f.read_bytes()) for f in sorted((root / g).glob("photo-*/sample.json"))]
               for g in ("test", "validation")}
    payloads = {"report.json": report, "plan.json": identity, "samples.json": samples,
                "attempts.json": attempts(root),
                "sample-receipts.json": {f.relative_to(root).as_posix(): json.loads(f.read_bytes())
                    for g in ("test", "validation") for f in sorted((root / g).glob("photo-*/receipt.json"))},
                "runtime-gates.json": {f.relative_to(root).as_posix(): json.loads(f.read_bytes()) for f in root.glob("attempt-*/normal-gate.json")}}
    for name in ("screen", "comparison", "warm-baseline", "warm-candidate", "stability-summary"):
        if (root / name).exists():
            payloads[name + ".json"] = e.value(root, name, signature)
    payloads["handoff-manifest.json"] = {n: hashlib.sha256(fit.canonical(v)).hexdigest() for n, v in payloads.items()}
    e.inputs.v.publish(root, "export", signature, payloads)
    e.inputs.v.probe.export(root / "export", handoff)


def execute(root, archive_path, fit_root, recovery_path, records, source, identity, handoff,
            runtime_factory=Runtime, monitor_factory=Monitor, baseline_factory=EvaluationRuntime):
    signature, contract = fit.digest(identity), identity["contract"]
    checkpoint = fit_root / "step-070000/model.pt"
    photos = source["cohorts"]
    regression = e.POLICY["acceptance"]["regression_keys"]
    selected = [next(i for i, photo in enumerate(photos["test"]) if photo["key"] == key) for key in regression]
    with fit.locked_run(root):
        if not (root / "identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output exists without identity; preserved")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        cached = {}
        for g in ("test", "validation"):
            (root / g).mkdir(exist_ok=True)
            allowed = {f"photo-{i:03d}" for i in range(len(photos[g]))}
            if any(f.name not in allowed for f in (root / g).glob("photo-*")):
                raise ValueError("Unexpected cached photo")
            for i, photo in enumerate(photos[g]):
                if (root / g / f"photo-{i:03d}").exists():
                    cached[g, i] = read_cache(root, g, i, signature, photo, contract)
        if (root / "screen").exists():
            expected = e.screen([cached["test", i] for i in selected])
            if e.value(root, "screen", signature) != expected:
                raise ValueError("Saved screen differs")
        else:
            expected = {"passed": False}
        if any(g == "validation" or i not in selected for g, i in cached) and not expected["passed"]:
            raise ValueError("Full test cache before passing 12-case screen")
        if (root / "result").exists():
            report = e.value(root, "result", signature)
            if len(cached) != report["cached_photos"]:
                raise ValueError("Completed result missing samples")
            export(root, handoff, signature, report, identity)
            return report
        prior_failure = identity.get("reviewed_v1_failure", {"seconds": 0.})
        prior = prior_failure["seconds"] + sum(a["seconds"] for a in previous)
        if prior >= e.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative 4-hour budget exhausted; review required")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        monitor = monitor_factory(None, prior, started, e.POLICY)
        runtime, status, error = None, "failed", None
        full = warm = sensitive = None
        try:
            with monitor, zipfile.ZipFile(archive_path) as archive:
                monitor.check()
                runtime = runtime_factory(checkpoint, identity["fit_signature"])
                monitor.runtime = runtime
                gate = normal_gate(archive, runtime, monitor, records, source, fit_root)
                fit.write_json(attempt / "normal-gate.json", gate)
                gate_path = (attempt / "normal-gate.json").relative_to(root).as_posix()
                # Load approved recovery lazily when a selected photo actually requires it.
                recovered = None

                def collect(g, indices):
                    nonlocal recovered
                    result = []
                    for n, i in enumerate(indices):
                        if (g, i) not in cached:
                            monitor.check()
                            recovered = recovery_for_photo(photos[g][i], recovered, recovery_path, source["previous"])
                            raw, sample = score_photo(archive, photos[g][i], runtime, recovered, monitor, contract)
                            name = f"photo-{i:03d}"
                            with fit.phase_stage(root / g, name) as stage:
                                np.save(stage / "maps.npy", raw, allow_pickle=False)
                                fit.write_json(stage / "sample.json", sample)
                                fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photos[g][i]),
                                    "gate_path": gate_path, "gate_sha256": fit.sha256_file(root / gate_path)}, ["maps.npy", "sample.json"])
                            cached[g, i] = sample
                        result.append(cached[g, i])
                        if (n+1) % 10 == 0 or n+1 == len(indices):
                            print(f"{g} {n+1}/{len(indices)}", flush=True)
                    return result

                print("1/4 Fixed 12 prior detections. Final model and threshold unchanged.", flush=True)
                screen = e.screen(collect("test", selected))
                publish(root, "screen", signature, screen)
                if screen["passed"]:
                    print("2/4 Screen 12/12 passed. Observed defects55 + normals54/256; no masks.", flush=True)
                    raw = {g: collect(g, list(range(len(photos[g])))) for g in ("test", "validation")}
                    full = e.compare(raw, source)
                    publish(root, "comparison", signature, full)
                monitor.check()
                runtime.release()
                runtime = monitor.runtime = None
                if screen["passed"]:
                    print("3/4 Paired fixed normal warm timing, 5 warmups + 20 photos x 3; models loaded sequentially.", flush=True)
                    warm = {}
                    for kind in ("baseline", "candidate"):
                        phase = "warm-" + kind
                        if (root / phase).exists():
                            warm[kind] = e.value(root, phase, signature)
                            continue
                        if kind == "candidate":
                            runtime = runtime_factory(checkpoint, identity["fit_signature"])
                        else:
                            baseline_checkpoint = fit_root.parent / "mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
                            if fit.sha256_file(baseline_checkpoint) != ev.CHECKPOINT_SHA256:
                                raise ValueError("Baseline checkpoint changed")
                            runtime = baseline_factory(source["config"])
                            runtime.load(baseline_checkpoint, source["records"]["fit.json"]["checkpoint_context"])
                        monitor.runtime = runtime
                        monitor.check()
                        warm[kind] = benchmark(kind, runtime, archive, source, records, contract, monitor)
                        monitor.check()
                        publish(root, phase, signature, warm[kind])
                        runtime.release()
                        runtime = monitor.runtime = None
                if full and full["passed"] and warm["candidate"]["candidate_budget_passed"]:
                    print("4/4 Normal sensitivity: 600 fixed paired draws; exact map quantiles are refitted. May take hours.", flush=True)
                    sensitive = stability(root, signature, fit_root, source, records, monitor)
                    publish(root, "stability-summary", signature, sensitive)
                monitor.check()
                if fit.sha256_file(checkpoint) != identity["checkpoint_sha256"]:
                    raise ValueError("EfficientAD checkpoint changed during evaluation")
                status = "completed"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            fit.write_json(attempt / "error-detail.json", {"error": error, "traceback": traceback.format_exc()})
            raise
        finally:
            try:
                if runtime is not None:
                    monitor.check()
                    runtime.release()
            except BaseException as exc:
                status, error = "failed", f"Cleanup failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter() - started
                if prior + elapsed > e.POLICY["resource_plan"]["comparison_seconds"]:
                    status, error = "failed", "Cumulative evaluation time cap exceeded"
                fit.write_json(attempt / "finished.json", {"status": status, "error": error, "seconds": elapsed,
                    "memory": monitor.snapshot(), "cached_photos": len(cached),
                    "scope": "runtime load/replay, scoring/cache, warm benchmarks, CPU bootstrap; input hashing/export excluded"})
        if error:
            raise RuntimeError(error)
        passed = bool(screen["passed"] and full["passed"] and warm["candidate"]["candidate_budget_passed"] and sensitive and sensitive["passed"])
        report = {"status": "efficientad_observed_evaluation_completed", "screen_passed": screen["passed"],
                  "full_comparison_run": full is not None, "full_accuracy_passed": full["passed"] if full else None,
                  "warm_evaluated": warm is not None, "stability_evaluated": sensitive is not None,
                  "all_gates_passed": passed, "cached_photos": len(cached),
                  "next": "review_completed_comparison" if passed else "review_failed_gate_no_automatic_retraining",
                  "stability_skip_reason": None if sensitive else "screen, accuracy or candidate warm gate did not pass",
                  "reviewed_v1_failure": prior_failure,
                  "cumulative_seconds": prior_failure["seconds"] + sum(a["seconds"] for a in attempts(root)),
                  "test_masks_decoded": 0, "training_performed": False, "threshold_changed": False,
                  "automatic_pass_allowed": False, "service_model_changed": False,
                  "independent_performance_claim_allowed": False}
        publish(root, "result", signature, report)
        export(root, handoff, signature, report, identity)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    fit_root = project / "artifacts" / p.NAME
    # Reader lock prevents a concurrent old training runner from touching the artifact.
    with fit.locked_run(fit_root):
        records, source, identity = e.prepare(project, project / "artifacts/handoff", fit_root)
        identity["reviewed_v1_failure"] = reviewed_v1_failure(
            project / "artifacts/analyses/mpi-efficientad-evaluation-v1", identity)
        if identity["reviewed_v1_failure"]["present"]:
            print("Reviewed v1 buffer error: old files preserved, prior time counted; all 12 cases rerun in v2.", flush=True)
        if environment() != identity["environment"]:
            raise ValueError("Environment differs from the reviewed fit")
        if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
            raise ValueError("MPI source archive differs")
        baseline_checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        if fit.sha256_file(baseline_checkpoint) != ev.CHECKPOINT_SHA256:
            raise ValueError("Baseline checkpoint missing or changed; required for paired warm timing")
        if shutil.disk_usage(project).free < e.POLICY["free_disk_GiB"] * 1024**3:
            raise ValueError("Need 12 GiB free disk; no automatic deletion")
        report = execute(project / "artifacts/analyses" / e.NAME, args.archive, fit_root,
            project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
            records, source, identity, project / "artifacts/handoff" / (e.NAME + ".zip"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all outputs; do not retrain or change thresholds.", file=sys.stderr)
        raise SystemExit(1)
