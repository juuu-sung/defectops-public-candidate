"""Matched 500-update experiment: reconstruct only the detection module."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_supersimplenet_head_reset_protocol as p
from scripts.mpi_supersimplenet_head_reset_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_varying import evaluate, compare_arrays, check_targets, reload_verify
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit


def export(root, handoff, identity, report):
    signature = fit.digest(identity)
    phases = ["baseline-control"] + [f"step-{s:04d}" for s in p.POLICY["snapshots"]]
    names = ["plan.json", "report.json", "attempts.json", "initialization.json"]
    names += [f"{phase}/{n}" for phase in phases for n in p.EVIDENCE_FILES + ["receipt.json"]]
    if not (root/"bundle").exists():
        with fit.phase_stage(root, "bundle") as stage:
            fit.write_json(stage/"plan.json", identity)
            fit.write_json(stage/"report.json", report)
            fit.write_json(stage/"attempts.json", attempts(root))
            fit.write_json(stage/"initialization.json", p.v.old.e.value(root, "initialization", signature))
            for name in names[4:]:
                (stage/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/name, stage/name)
            fit.write_json(stage/"handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names})
            fit.seal_phase(stage, "bundle", signature, {}, names + ["handoff-manifest.json"])
    names += ["handoff-manifest.json"]
    fit.read_phase(root, "bundle", signature, names)
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(names) or set(z.namelist()) != set(names):
                raise ValueError("Existing handoff inventory differs")
            if any(hashlib.sha256(z.read(n)).hexdigest() != fit.sha256_file(root/"bundle"/n) for n in names):
                raise ValueError("Existing handoff differs; preserve it")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".head-reset-export-", dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage, "x", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(root/"bundle"/name, name)
    stage.rename(handoff)


def execute(root, archive, fit_root, varying_root, identity, records, references, handoff,
            runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    original, baseline_path = fit_root/"step-028800/model.pt", varying_root/"step-0500/model.pt"
    original_fp = records["report.json"]["final_fingerprints"]

    def check_originals():
        if (fit.sha256_file(original) != identity["checkpoint_sha256"] or
                fit.sha256_file(baseline_path) != identity["baseline_checkpoint_sha256"]):
            raise ValueError("Original or baseline checkpoint changed")
        p.check_previous_files(root.parent/p.PREVIOUS_NAME, identity["previous_failure"])

    check_originals()
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve and review")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        prior_seconds = identity["previous_failure"]["seconds"] + sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        control = p.v.read_phase(root, "baseline-control", signature)[0] if (root/"baseline-control").exists() else None
        initialization = (p.v.old.e.value(root, "initialization", signature)
                          if (root/"initialization").exists() else None)
        if initialization is not None:
            p.validate_initialization(initialization, original_fp)
            p.validate_same_initialization(initialization, identity["previous_failure"]["initialization"])
        trajectory = p.v.read_checkpoints(root, signature)
        if trajectory and (control is None or initialization is None):
            raise ValueError("Missing baseline or initialization receipt")
        if control is not None and control["fingerprints"] != identity["baseline_fingerprints"]:
            raise ValueError("Saved baseline fingerprint differs")
        if control is not None:
            p.validate_evaluations(control["evaluations"])
        for progress in trajectory:
            check_targets(progress["evaluations"], control)
            p.validate_progress(progress, identity, initialization)
        if (root/"result").exists():
            report = p.v.old.e.value(root, "result", signature)
            if len(trajectory) != len(p.POLICY["snapshots"]):
                raise ValueError("Missing completed checkpoints")
            export(root, handoff, identity, report)
            return report
        if prior_seconds >= p.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative time exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        runtime = None
        error, interrupted, attempted = None, False, 0
        failure_diagnostics = None
        monitor = monitor_factory(None, prior_seconds, started, p.POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as source, ExitStack() as stack:
                images = [stack.enter_context(fit.checked_image(source, r)) for r in identity["job"]["rows"]]
                if control is None:
                    print("1/3 Replaying frozen varying015500 baseline on five synthetic seeds", flush=True)
                    runtime = runtime_factory(original, identity["fit_signature"])
                    monitor.runtime = runtime
                    runtime.load_baseline(baseline_path, identity["baseline_signature"])
                    if runtime.fingerprints() != identity["baseline_fingerprints"]:
                        raise ValueError("Loaded baseline differs")
                    evaluations, flat = evaluate(runtime, images, monitor)
                    p.validate_evaluations(evaluations)
                    replay = compare_arrays(flat, references["baseline"])
                    control = {"step": 500, "losses": [], "optimizer_updates": 0, "evaluations": evaluations,
                               "fingerprints": runtime.fingerprints(), "previous_reference_max_abs_difference": replay}
                    with fit.phase_stage(root, "baseline-control") as stage:
                        np.savez_compressed(stage/"arrays.npz", **flat)
                        fit.write_json(stage/"progress.json", control)
                        fit.seal_phase(stage, "baseline-control", signature, {}, p.EVIDENCE_FILES)
                    monitor.check()
                    monitor.runtime = None
                    runtime.release()
                    runtime = None
                print("2/3 Original model copy; reproduce v1 head initialization, then 500 matched updates", flush=True)
                runtime = runtime_factory(original, identity["fit_signature"])
                monitor.runtime = runtime
                if runtime.fingerprints() != original_fp:
                    raise ValueError("Candidate did not start from original formal model")
                losses = trajectory[-1]["losses"].copy() if trajectory else []
                completed = trajectory[-1]["step"] if trajectory else -1
                if trajectory:
                    reload_verify(runtime, root, trajectory[-1], signature, images, monitor)
                else:
                    # Check formal model replay BEFORE replacing its head.
                    print("Checking original formal model before head reset...", flush=True)
                    _, original_arrays = evaluate(runtime, images, monitor)
                    original_delta = compare_arrays(original_arrays, references["original"])
                    fresh = runtime.reset_head()
                    fresh["original_replay_max_abs_difference"] = original_delta
                    p.validate_initialization(fresh, original_fp)
                    p.validate_same_initialization(fresh, identity["previous_failure"]["initialization"])
                    publish(root, "initialization", signature, fresh)
                    initialization = fresh
                    print("Head initialization matches v1. Checking step zero before ANY optimizer update...", flush=True)
                for stop in p.POLICY["snapshots"]:
                    if stop <= completed:
                        continue
                    while len(losses) < stop:
                        monitor.check()
                        if prior_updates + attempted >= p.POLICY["maximum_attempted_updates"]:
                            raise RuntimeError("Attempted-update limit; no automatic extension")
                        attempted += 1
                        row = runtime.step(images, len(losses))
                        if row["target_sha256"] != identity["baseline_training_targets"][len(losses)]:
                            raise ValueError("Training synthetic targets differ from baseline; no reroll")
                        losses.append(row)
                        if len(losses) % 25 == 0:
                            print(f"Head-reset learning {len(losses)}/500; loss={row['loss']:.6f}", flush=True)
                    evaluations, flat = evaluate(runtime, images, monitor)
                    check_targets(evaluations, control)
                    p.validate_evaluations(evaluations)
                    checks = [e["checks"]["public_batch_comparison"] for e in evaluations.values()]
                    same = max(max(d["same_batch_map_max_abs"], d["same_batch_score_max_abs"]) for d in checks)
                    cross = max(max(d["cross_batch_map_max_abs"], d["cross_batch_score_max_abs"]) for d in checks)
                    print(f"Step {stop} public check: same-batch max={same:.8g}; 16-vs-8 max={cross:.8g}", flush=True)
                    progress = {"step": stop, "losses": losses.copy(), "evaluations": evaluations,
                                "fingerprints": runtime.fingerprints()}
                    p.validate_progress(progress, identity, initialization)
                    phase = f"step-{stop:04d}"
                    with fit.phase_stage(root, phase) as stage:
                        runtime.save(stage/"model.pt", signature, stop)
                        np.savez_compressed(stage/"arrays.npz", **flat)
                        fit.write_json(stage/"progress.json", progress)
                        monitor.check()
                        fit.seal_phase(stage, phase, signature, {}, p.FILES)
                    trajectory.append(progress)
                print("3/3 Reloading final model and reproducing all five synthetic evaluations", flush=True)
                reload_delta = reload_verify(runtime, root, trajectory[-1], signature, images, monitor)
                monitor.check()
                check_originals()
        except BaseException as exc:
            error, interrupted = f"{type(exc).__name__}: {exc}", isinstance(exc, KeyboardInterrupt)
            failure_diagnostics = getattr(exc, "diagnostics", None)
            raise
        finally:
            elapsed = time.perf_counter() - started
            fit.write_json(attempt/"finished.json", {"seconds": elapsed, "attempted_updates": attempted,
                "status": "interrupted" if interrupted else ("failed" if error else "completed"),
                "error": error, "diagnostics": failure_diagnostics, "memory": monitor.snapshot()})
            monitor.runtime = None
            if runtime is not None:
                runtime.release()
        def summary(phase):
            progress, flat = p.v.read_phase(root, phase, signature, phase.startswith("step-"))
            return p.aggregate(progress["evaluations"], flat)
        report = {"status": "head_reset_comparison_completed", "completed_updates": p.POLICY["updates"],
            "attempted_updates_including_replayed_work": prior_updates + attempted,
            "cumulative_compute_seconds": prior_seconds + elapsed,
            "previous_failure_seconds": identity["previous_failure"]["seconds"],
            "previous_failure_preserved": True, "public_eval_validation": p.PUBLIC_VALIDATION,
            "baseline_heldout": summary("baseline-control"),
            "trajectory": [{"step": x["step"], "heldout": summary(f"step-{x['step']:04d}")}
                           for x in trajectory],
            "initialization": initialization,
            "unique_training_target_hashes": len({r["target_sha256"] for r in trajectory[-1]["losses"]}),
            "all_training_targets_match_baseline": True,
            "final_reload_max_abs_difference": reload_delta, "original_checkpoints_unchanged": True,
            "test_images_decoded": 0, "test_masks_decoded": 0, "threshold_changed": False,
            "service_model_changed": False, "automatic_pass_allowed": False, "automatic_model_selection": False,
            "note": "One fixed initialization seed; synthetic draws on the same eight training tiles. Not real-defect accuracy."}
        publish(root, "result", signature, report)
        export(root, handoff, identity, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original archive changed")
    if shutil.disk_usage(project).free < p.POLICY["free_disk_GiB"] * 1024**3:
        raise ValueError("Need 8 GiB free disk")
    fit_root = project/"artifacts"/p.v.q.p.NAME
    fixed_root = project/"artifacts/analyses"/p.v.old.NAME
    varying_root = project/"artifacts/analyses"/p.v.NAME
    previous_root = project/"artifacts/analyses"/p.PREVIOUS_NAME
    for path in (fit_root, fixed_root, varying_root, previous_root):
        if not path.is_dir():
            raise ValueError(f"Preserved prior folder required: {path}")
    with fit.locked_run(fit_root), fit.locked_run(fixed_root), fit.locked_run(varying_root), fit.locked_run(previous_root):
        records, identity, references = p.prepare(project, project/"artifacts/handoff", fit_root, fixed_root, varying_root)
        current = environment()
        for name in ("torch", "torchvision", "anomalib", "timm", "supersimplenet_sources"):
            if current[name] != identity["environment"][name]:
                raise ValueError(f"Runtime environment changed: {name}")
        identity["runtime_environment"] = current
        report = execute(project/"artifacts/analyses"/p.NAME, args.archive, fit_root, varying_root,
                         identity, records, references, project/"artifacts/handoff"/(p.NAME + ".zip"))
    print(json.dumps({k: v for k, v in report.items() if k not in ("trajectory", "baseline_heldout", "initialization")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; do not reroll, extend or change thresholds.", file=sys.stderr)
        raise SystemExit(1)
