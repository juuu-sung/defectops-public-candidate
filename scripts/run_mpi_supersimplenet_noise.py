"""Compare two fixed synthetic-noise training amplitudes; no actual defect test."""
from contextlib import ExitStack
import argparse
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
from scripts import mpi_supersimplenet_noise_protocol as p
from scripts.mpi_supersimplenet_noise_runtime import Runtime, environment
from scripts.run_mpi_supersimplenet_varying import compare_arrays
from scripts.run_mpi_supersimplenet_evaluation import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit


def evaluate(runtime, images, monitor):
    before = runtime.fingerprints()
    evaluations, flat = {}, {}
    for strength in p.STRENGTHS:
        entries = {}
        for seed in p.v.POLICY["evaluation_seeds"]:
            monitor.check()
            arrays, checks = runtime.evaluate_strength(images, seed, strength)
            if not checks["state_unchanged"] or checks["noise_std"] != p.STRENGTHS[strength]:
                raise ValueError("Evaluation changed state/noise configuration")
            entries[str(seed)] = {**p.summarize(arrays), "checks": checks}
            flat.update({f"{strength}__s{seed}__{k}": a for k, a in arrays.items()})
        evaluations[strength] = entries
    p.targets_match(evaluations, evaluations)
    if runtime.fingerprints() != before:
        raise ValueError("Evaluation changed model parameters/buffers")
    monitor.check()
    return evaluations, flat


def reload_verify(runtime, root, progress, signature, images, monitor):
    arm, step = progress["arm"], progress["step"]
    phase = f"{arm}-step-{step:04d}"
    runtime.load_arm(root/phase/"model.pt", signature, step, arm)
    _, actual = evaluate(runtime, images, monitor)
    with np.load(root/phase/"arrays.npz", allow_pickle=False) as z:
        difference = compare_arrays(actual, {k: z[k] for k in z.files})
    if runtime.fingerprints() != progress["fingerprints"]:
        raise ValueError("Reload fingerprints differ")
    return difference


def export(root, handoff, identity, report):
    signature = fit.digest(identity)
    phases = ["baseline"] + [f"{a}-step-{s:04d}" for a in p.ARMS for s in p.POLICY["snapshots"]]
    names = ["plan.json", "report.json", "attempts.json"]
    names += [f"{phase}/{n}" for phase in phases for n in p.EVIDENCE + ["receipt.json"]]
    if not (root/"bundle").exists():
        with fit.phase_stage(root, "bundle") as stage:
            fit.write_json(stage/"plan.json", identity)
            fit.write_json(stage/"report.json", report)
            fit.write_json(stage/"attempts.json", attempts(root))
            for name in names[3:]:
                (stage/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/name, stage/name)
            fit.write_json(stage/"handoff-manifest.json", {n: fit.sha256_file(stage/n) for n in names})
            fit.seal_phase(stage, "bundle", signature, {}, names + ["handoff-manifest.json"])
    names += ["handoff-manifest.json"]
    fit.read_phase(root, "bundle", signature, names)
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if (len(z.namelist()) != len(names) or set(z.namelist()) != set(names) or
                    any(hashlib.sha256(z.read(n)).hexdigest() != fit.sha256_file(root/"bundle"/n) for n in names)):
                raise ValueError("Existing handoff differs; preserve it")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".noise-export-", dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage, "x", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(root/"bundle"/name, name)
    stage.rename(handoff)


def execute(root, archive, fit_root, varying_root, identity, records, references, handoff,
            runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    original = fit_root/"step-028800/model.pt"
    baseline_path = varying_root/"step-0500/model.pt"
    def check_originals():
        if (fit.sha256_file(original) != identity["checkpoint_sha256"] or
                fit.sha256_file(baseline_path) != identity["baseline_checkpoint_sha256"]):
            raise ValueError("Original or baseline checkpoint changed")
    check_originals()
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve and review")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        prior_seconds = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        baseline = p.read_phase(root, "baseline", signature)[0] if (root/"baseline").exists() else None
        histories = {a: p.checkpoints(root, a, signature, identity) for a in p.ARMS}
        for history in histories.values():
            if history and baseline is None:
                raise ValueError("Missing read-only baseline")
            for progress in history:
                p.targets_match(progress["evaluations"], baseline["evaluations"])
                if progress["fingerprints"]["backbone"] != records["report.json"]["final_fingerprints"]["backbone"]:
                    raise ValueError("Frozen backbone differs")
        completed = {}
        for arm in p.ARMS:
            if (root/(arm+"-result")).exists():
                if len(histories[arm]) != len(p.POLICY["snapshots"]):
                    raise ValueError("Completed arm has missing checkpoint")
                completed[arm] = p.v.old.e.value(root, arm+"-result", signature)
        if (root/"result").exists():
            if set(completed) != set(p.ARMS):
                raise ValueError("Incomplete result")
            report = p.v.old.e.value(root, "result", signature)
            export(root, handoff, identity, report)
            return report
        if prior_seconds >= p.POLICY["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("Cumulative compute budget exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        runtime, error, interrupted, attempted = None, None, False, 0
        monitor = monitor_factory(None, prior_seconds, started, p.POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as source, ExitStack() as stack:
                images = [stack.enter_context(fit.checked_image(source, r)) for r in identity["job"]["rows"]]
                original_fingerprints = records["report.json"]["final_fingerprints"]
                if baseline is None:
                    print("1/3 Read-only std0.015 varying500 baseline: all three evaluation strengths", flush=True)
                    runtime = runtime_factory(original, identity["fit_signature"])
                    monitor.runtime = runtime
                    runtime.load_baseline(baseline_path, identity["baseline_signature"])
                    if runtime.fingerprints() != identity["baseline_fingerprints"]:
                        raise ValueError("Baseline loaded state differs")
                    evaluations, flat = evaluate(runtime, images, monitor)
                    delta = compare_arrays(p.arrays_for(flat, "std015"), references["baseline"])
                    baseline = {"evaluations": evaluations, "optimizer_updates": 0,
                        "fingerprints": runtime.fingerprints(), "previous_replay_max_abs_difference": delta}
                    with fit.phase_stage(root, "baseline") as stage:
                        np.savez_compressed(stage/"arrays.npz", **flat)
                        fit.write_json(stage/"progress.json", baseline)
                        fit.seal_phase(stage, "baseline", signature, {}, p.EVIDENCE)
                    monitor.check(); monitor.runtime = None; runtime.release(); runtime = None
                for arm in p.ARMS:
                    if arm in completed:
                        continue
                    print(f"2/3 {arm}: fresh original model + 500 updates; other learning conditions fixed", flush=True)
                    runtime = runtime_factory(original, identity["fit_signature"])
                    monitor.runtime = runtime
                    if runtime.fingerprints() != original_fingerprints:
                        raise ValueError("Arm did not start from original formal model")
                    history = histories[arm]
                    losses = history[-1]["losses"].copy() if history else []
                    last_step = history[-1]["step"] if history else -1
                    if history:
                        reload_verify(runtime, root, history[-1], signature, images, monitor)
                    for stop in p.POLICY["snapshots"]:
                        if stop <= last_step:
                            continue
                        while len(losses) < stop:
                            monitor.check()
                            if prior_updates + attempted >= p.POLICY["maximum_attempted_updates"]:
                                raise RuntimeError("Attempted-update budget exceeded; no automatic extension")
                            attempted += 1
                            row = runtime.step_at_strength(images, len(losses), arm)
                            if row["target_sha256"] != identity["baseline_training_targets"][len(losses)]:
                                raise ValueError("Training synthetic locations differ from baseline; no reroll")
                            losses.append(row)
                            if len(losses) % 50 == 0:
                                print(f"{arm} learning {len(losses)}/500; loss={row['loss']:.6f}", flush=True)
                        evaluations, flat = evaluate(runtime, images, monitor)
                        p.targets_match(evaluations, baseline["evaluations"])
                        replay = compare_arrays(p.arrays_for(flat, "std015"), references["original"]) if stop == 0 else None
                        fingerprints = runtime.fingerprints()
                        if fingerprints["backbone"] != original_fingerprints["backbone"]:
                            raise ValueError("Frozen backbone changed")
                        progress = {"arm": arm, "step": stop, "losses": losses.copy(),
                            "evaluations": evaluations, "fingerprints": fingerprints,
                            "original_replay_max_abs_difference": replay}
                        phase = f"{arm}-step-{stop:04d}"
                        with fit.phase_stage(root, phase) as stage:
                            runtime.save_arm(stage/"model.pt", signature, stop, arm)
                            np.savez_compressed(stage/"arrays.npz", **flat)
                            fit.write_json(stage/"progress.json", progress)
                            monitor.check()
                            fit.seal_phase(stage, phase, signature, {}, p.FILES)
                        history.append(progress)
                        print(f"{arm} saved {stop}: common three strengths, four held-out seeds + reference42", flush=True)
                    delta = reload_verify(runtime, root, history[-1], signature, images, monitor)
                    monitor.check(); check_originals()
                    completed[arm] = {"completed_updates": p.POLICY["updates"], "reload_max_abs_difference": delta,
                        "trajectory": [{"step": x["step"], "heldout_by_evaluation_strength":
                            {k: p.aggregate(entries) for k, entries in x["evaluations"].items()}} for x in history]}
                    publish(root, arm+"-result", signature, completed[arm])
                    monitor.runtime = None; runtime.release(); runtime = None
                check_originals()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            interrupted = isinstance(exc, KeyboardInterrupt)
            raise
        finally:
            elapsed = time.perf_counter() - started
            fit.write_json(attempt/"finished.json", {"seconds": elapsed, "attempted_updates": attempted,
                "status": "interrupted" if interrupted else ("failed" if error else "completed"),
                "error": error, "memory": monitor.snapshot()})
            monitor.runtime = None
            if runtime is not None:
                runtime.release()
        report = {"status": "noise_strength_diagnostic_completed", "arms": completed,
            "baseline_heldout_by_evaluation_strength": {k: p.aggregate(val) for k, val in baseline["evaluations"].items()},
            "attempted_updates_including_replayed_work": prior_updates + attempted,
            "cumulative_compute_seconds": prior_seconds + elapsed, "original_checkpoints_unchanged": True,
            "test_images_decoded": 0, "test_masks_decoded": 0, "threshold_changed": False,
            "service_model_changed": False, "automatic_pass_allowed": False, "automatic_model_selection": False,
            "note": "Synthetic diagnostics on the same one normal photo/eight tiles; previously observed held-out seeds. Not real-defect or unseen-photo accuracy."}
        publish(root, "result", signature, report)
        print("3/3 Packaging evidence only; images and checkpoints remain on the lab computer", flush=True)
        export(root, handoff, identity, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original archive differs")
    if shutil.disk_usage(project).free < p.POLICY["free_disk_GiB"] * 1024**3:
        raise ValueError("Need 16 GiB free disk before starting")
    fit_root = project/"artifacts"/p.v.q.p.NAME
    fixed_root = project/"artifacts/analyses"/p.v.old.NAME
    varying_root = project/"artifacts/analyses"/p.v.NAME
    with ExitStack() as stack:
        for path in (fit_root, fixed_root, varying_root):
            if not path.is_dir():
                raise ValueError(f"Preserved prior experiment folder required: {path}")
            stack.enter_context(fit.locked_run(path))
        records, identity, references = p.prepare(project, project/"artifacts/handoff", fit_root, fixed_root, varying_root)
        current = environment()
        for key in ("torch", "torchvision", "anomalib", "timm", "supersimplenet_sources"):
            if current[key] != identity["environment"][key]:
                raise ValueError(f"Runtime environment differs: {key}")
        identity["runtime_environment"] = current
        report = execute(project/"artifacts/analyses"/p.NAME, args.archive, fit_root, varying_root,
                         identity, records, references, project/"artifacts/handoff"/(p.NAME+".zip"))
    print(json.dumps({k: v for k, v in report.items() if k not in ("arms", "baseline_heldout_by_evaluation_strength")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve files; no reroll, automatic extension or threshold changes.", file=sys.stderr)
        raise SystemExit(1)
