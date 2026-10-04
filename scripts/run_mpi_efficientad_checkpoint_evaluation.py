"""Recalibrate 10k/30k separately, screen 12, then gated comparison. Never train."""
from __future__ import annotations
import argparse
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
from scripts import mpi_efficientad_checkpoint_protocol as c
from scripts import run_mpi_efficientad_evaluation as old
from scripts.mpi_efficientad_trajectory_runtime import Runtime, environment
from scripts.run_mpi_efficientad_trajectory import export as bundle_export
from scripts.run_mpi_efficientad_probe import publish, attempts

fit, p, e = c.fit, c.p, c.e


def raw_normal(archive, photo, runtime, monitor):
    maps = []
    for row in photo["regions"]:
        monitor.check()
        with fit.checked_image(archive, row) as image:
            maps.append(p.validate_maps(runtime.predict(image)))
    return p.validate_maps(np.stack(maps))


def normal_maps(root, group, index, photo, signature):
    name = f"photo-{index:03d}"
    receipt = fit.read_phase(root/group, name, signature, ["maps.npy"])
    if receipt["photo_sha256"] != fit.digest(photo):
        raise ValueError("Normal cache cohort differs")
    raw = p.validate_maps(np.load(root/group/name/"maps.npy", mmap_mode="r", allow_pickle=False))
    if raw.shape != (12, 2, 256, 256):
        raise ValueError("Normal cache shape differs")
    return raw


def normals(root, archive, runtime, monitor, source, signature):
    arrays = {}
    for group in ("development", "calibration"):
        arrays[group] = []
        (root/group).mkdir(exist_ok=True)
        allowed = {f"photo-{i:03d}" for i in range(len(source["cohorts"][group]))}
        if any(x.name.startswith("photo-") and x.name not in allowed for x in (root/group).iterdir()):
            raise ValueError("Unexpected normal cache")
        for i, photo in enumerate(source["cohorts"][group]):
            name = f"photo-{i:03d}"
            if not (root/group/name).exists():
                raw = raw_normal(archive, photo, runtime, monitor)
                with fit.phase_stage(root/group, name) as stage:
                    np.save(stage/"maps.npy", raw, allow_pickle=False)
                    fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photo)}, ["maps.npy"])
            arrays[group].append(normal_maps(root, group, i, photo, signature))
            monitor.check()
            if (i+1) % 25 == 0 or i+1 == len(source["cohorts"][group]):
                print(f"normal {group} {i+1}/{len(source['cohorts'][group])}", flush=True)
    quantiles = p.map_quantiles(arrays["development"], monitor.check)
    scores = {}
    for group, maps in arrays.items():
        scores[group] = []
        for raw in maps:
            monitor.check()
            scores[group].append(p.tile_scores(raw, quantiles).tolist())
    result = {"contract": p.calibrate(**scores, quantiles=quantiles), "tile_scores": scores}
    publish(root, "normal-results", signature, result)  # Existing value must exactly match.
    return result


def validate_cache(root, source, signature, contract):
    cached = {}
    selected = e.POLICY["acceptance"]["regression_keys"]
    for group in ("test", "validation"):
        (root/group).mkdir(exist_ok=True)
        photos = source["cohorts"][group]
        allowed = {f"photo-{i:03d}" for i in range(len(photos))}
        if any(x.name not in allowed for x in (root/group).iterdir()):
            raise ValueError("Unexpected test cache")
        for i, photo in enumerate(photos):
            if (root/group/f"photo-{i:03d}").exists():
                cached[group, i] = old.read_cache(root, group, i, signature, photo, contract)
    screen = None
    if (root/"screen").exists():
        by_key = {s["key"]: s for (g, _), s in cached.items() if g == "test"}
        screen = e.screen([by_key[k] for k in selected])
        if e.value(root, "screen", signature) != screen:
            raise ValueError("Saved screen differs")
    if any(g == "validation" or s["key"] not in selected for (g, _), s in cached.items()):
        if not screen or not screen["passed"]:
            raise ValueError("Full cache before passing 12-case screen")
    return cached


def release(runtime, monitor):
    # Capture allocator peaks before detaching; always release even on budget failure.
    try:
        monitor.check()
    finally:
        monitor.runtime = None
        runtime.release()


def evaluate_step(root, archive_path, fit_root, recovery, source, identity, step,
                  runtime_factory=Runtime, monitor_factory=old.Monitor,
                  baseline_factory=old.EvaluationRuntime):
    if step not in c.STEPS:
        raise ValueError("Only preselected 10k/30k checkpoints are allowed")
    plan = {"execution": identity, "step": step}
    signature = fit.digest(plan)
    checkpoint = fit_root/f"step-{step:06d}/model.pt"
    expected_sha = identity["checkpoints"][str(step)]["receipt"]["files"]["model.pt"]
    if fit.sha256_file(checkpoint) != expected_sha:
        raise ValueError("Saved checkpoint changed")
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserved")
        publish(root, "identity", signature, plan)
        previous = attempts(root)
        if (root/"result").exists():
            normal = e.value(root, "normal-results", signature)
            for group in ("development", "calibration"):
                for i, photo in enumerate(source["cohorts"][group]):
                    normal_maps(root, group, i, photo, signature)
            cached = validate_cache(root, source, signature, normal["contract"])
            report = e.value(root, "result", signature)
            if len(cached) != report["cached_test_photos"]:
                raise ValueError("Completed samples missing")
            return report
        prior = sum(a["seconds"] for a in previous)
        if prior >= c.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Per-checkpoint cumulative 4-hour cap exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        monitor = monitor_factory(None, prior, started, c.POLICY)
        runtime, error, status = None, None, "failed"
        full = warm = sensitive = None
        try:
            with monitor, zipfile.ZipFile(archive_path) as archive:
                monitor.check()
                runtime = runtime_factory(checkpoint, identity["fit_signature"], step)
                monitor.runtime = runtime
                print(f"Checkpoint {step}: 1/4 normal128+169 calibration; no test inference yet.", flush=True)
                normal = normals(root, archive, runtime, monitor, source, signature)
                contract = normal["contract"]
                releasing, runtime = runtime, None
                release(releasing, monitor)
                runtime = runtime_factory(checkpoint, identity["fit_signature"], step)
                monitor.runtime = runtime
                records = {"normal-results.json": normal}
                gate = old.normal_gate(archive, runtime, monitor, records, source, root)
                fit.write_json(attempt/"normal-gate.json", gate)
                gate_path = (attempt/"normal-gate.json").relative_to(root).as_posix()
                cached = validate_cache(root, source, signature, contract)
                recovered = None

                def collect(group, indices):
                    nonlocal recovered
                    values = []
                    photos = source["cohorts"][group]
                    for n, i in enumerate(indices):
                        monitor.check()
                        if (group, i) not in cached:
                            recovered = old.recovery_for_photo(photos[i], recovered, recovery, source["previous"])
                            raw, sample = old.score_photo(archive, photos[i], runtime, recovered, monitor, contract)
                            name = f"photo-{i:03d}"
                            with fit.phase_stage(root/group, name) as stage:
                                np.save(stage/"maps.npy", raw, allow_pickle=False)
                                fit.write_json(stage/"sample.json", sample)
                                fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photos[i]),
                                    "gate_path": gate_path, "gate_sha256": fit.sha256_file(root/gate_path)}, ["maps.npy", "sample.json"])
                            cached[group, i] = sample
                        values.append(cached[group, i])
                        if (n+1) % 25 == 0 or n+1 == len(indices):
                            print(f"checkpoint {step} {group} {n+1}/{len(indices)}", flush=True)
                    return values

                selected = [next(i for i, photo in enumerate(source["cohorts"]["test"]) if photo["key"] == key)
                            for key in e.POLICY["acceptance"]["regression_keys"]]
                print(f"Checkpoint {step}: 2/4 fixed12 screen with its own threshold.", flush=True)
                screen = e.screen(collect("test", selected))
                publish(root, "screen", signature, screen)
                if screen["passed"]:
                    print(f"Checkpoint {step}: 3/4 screen passed; full observed comparison and paired warm cost.", flush=True)
                    samples = {g: collect(g, list(range(len(source["cohorts"][g])))) for g in ("test", "validation")}
                    full = e.compare(samples, source)
                    publish(root, "comparison", signature, full)
                releasing, runtime = runtime, None
                release(releasing, monitor)
                if screen["passed"]:
                    warm = {}
                    for kind in ("baseline", "candidate"):
                        phase = "warm-"+kind
                        if (root/phase).exists():
                            warm[kind] = e.value(root, phase, signature)
                            continue
                        if kind == "candidate":
                            runtime = runtime_factory(checkpoint, identity["fit_signature"], step)
                        else:
                            baseline = fit_root.parent/"mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
                            if fit.sha256_file(baseline) != old.ev.CHECKPOINT_SHA256:
                                raise ValueError("Baseline checkpoint changed")
                            runtime = baseline_factory(source["config"])
                            runtime.load(baseline, source["records"]["fit.json"]["checkpoint_context"])
                        monitor.runtime = runtime
                        warm[kind] = old.benchmark(kind, runtime, archive, source, records, contract, monitor)
                        releasing, runtime = runtime, None
                        release(releasing, monitor)
                        publish(root, phase, signature, warm[kind])
                if full and full["passed"] and warm["candidate"]["candidate_budget_passed"]:
                    print(f"Checkpoint {step}: 4/4 accuracy+warm gates passed; 600 fixed normal resamples, may take hours.", flush=True)
                    sensitive = old.stability(root, signature, root, source, records, monitor)
                    publish(root, "stability-summary", signature, sensitive)
                monitor.check()
                if fit.sha256_file(checkpoint) != expected_sha:
                    raise ValueError("Checkpoint changed during evaluation")
                status = "completed"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyboardInterrupt): status = "interrupted"
            else:
                fit.write_json(attempt/"error-detail.json", {"error": error, "traceback": traceback.format_exc()})
            raise
        finally:
            try:
                if runtime is not None: release(runtime, monitor)
            except BaseException as exc:
                status, error = "failed", f"Cleanup failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter()-started
                if prior+elapsed > c.POLICY["resource_plan"]["comparison_seconds"]:
                    status, error = "failed", "Per-checkpoint cumulative time cap exceeded"
                fit.write_json(attempt/"finished.json", {"status": status, "error": error,
                    "seconds": elapsed, "memory": monitor.snapshot(),
                    "scope": "normal inference/calibration/reload, screen/full/cache/warm/bootstrap; excludes input hash/export"})
        if error: raise RuntimeError(error)
        report = {"status": "checkpoint_evaluation_completed_exploratory", "step": step,
                  "screen_passed": screen["passed"], "retained12": screen["retained"],
                  "full_comparison_run": full is not None, "full_accuracy_passed": full["passed"] if full else None,
                  "warm_evaluated": warm is not None, "stability_evaluated": sensitive is not None,
                  "all_gates_passed": bool(sensitive and sensitive["passed"]),
                  "cached_test_photos": len(cached), "cumulative_seconds": sum(a["seconds"] for a in attempts(root)),
                  "contract_sha256": fit.digest(contract), "threshold_fitted_from_normals": True,
                  "training_performed": False, "test_masks_decoded": 0,
                  "service_model_changed": False, "automatic_pass_allowed": False,
                  "independent_performance_claim_allowed": False}
        publish(root, "result", signature, report)
        return report


def finish(root, handoff, identity, reports):
    signature = fit.digest(identity)
    payload = {"plan.json": identity, "report.json": {"status": "two_checkpoint_review_ready",
        "checkpoints": reports, "selected_model": None, "training_performed": False,
        "next": "review_both_results_no_more_checkpoint_search", "service_model_changed": False,
        "automatic_pass_allowed": False, "independent_performance_claim_allowed": False}}
    for step in c.STEPS:
        directory = root/f"step-{step:06d}"
        sig = fit.digest({"execution": identity, "step": step})
        prefix = f"step-{step:06d}-"
        payload[prefix+"attempts.json"] = attempts(directory)
        for phase in ("normal-results", "screen", "comparison", "warm-baseline", "warm-candidate", "stability-summary", "result"):
            if (directory/phase).exists(): payload[prefix+phase+".json"] = e.value(directory, phase, sig)
        payload[prefix+"samples.json"] = {g: [old.read_cache(directory, g, i, sig, photo,
            payload[prefix+"normal-results.json"]["contract"]) for i, photo in enumerate(identity["cohorts"][g])
            if (directory/g/f"photo-{i:03d}").exists()] for g in ("test", "validation")}
        payload[prefix+"receipts.json"] = {f.relative_to(directory).as_posix(): json.loads(f.read_bytes())
            for f in directory.glob("*/photo-*/receipt.json")}
        payload[prefix+"runtime-gates.json"] = {f.relative_to(directory).as_posix(): json.loads(f.read_bytes())
            for f in directory.glob("attempt-*/normal-gate.json")}
    bundle_export(root, handoff, signature, payload, [])
    return payload["report.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    fit_root = project/"artifacts"/p.NAME
    if not fit_root.is_dir(): raise ValueError("Existing fit folder required; do not retrain")
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original archive differs")
    if shutil.disk_usage(project).free < c.POLICY["free_disk_GiB"]*1024**3:
        raise ValueError("Need 30 GiB free disk; no deletion or shrinking")
    with fit.locked_run(fit_root):
        source, identity = c.prepare(project, project/"artifacts/handoff", fit_root)
        if environment() != identity["environment"]:
            raise ValueError("Environment differs from reviewed fit")
        baseline = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        if fit.sha256_file(baseline) != old.ev.CHECKPOINT_SHA256:
            raise ValueError("Original baseline checkpoint required for paired warm timing")
        root = project/"artifacts/analyses"/c.NAME
        with fit.locked_run(root):
            signature = fit.digest(identity)
            if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
                raise ValueError("Existing output has no identity; preserved")
            publish(root, "identity", signature, identity)
            reports = []
            for step in c.STEPS:
                reports.append(evaluate_step(root/f"step-{step:06d}", args.archive, fit_root,
                    project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
                    source, identity, step))
            report = finish(root, project/"artifacts/handoff"/(c.NAME+".zip"), identity, reports)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no retraining or threshold edits.", file=sys.stderr)
        raise SystemExit(1)
