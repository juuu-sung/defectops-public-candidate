"""Frozen D-stage workers: score, separate-process warm benchmarks, paired bootstrap."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import mpi_normal_diversity_evaluation_protocol as p
from scripts import fit_mpi_normal_diversity as c_runner
from scripts import prepare_mpi_normal_diversity as preparation

fit, bundle = p.fit, preparation.bundle_tools
JOBS = ("score", "baseline-warm", "candidate-warm", "bootstrap")


class EvaluationRuntime(fit.TorchRuntime):
    """Retain load-time allocator peaks across the frozen loader's peak reset."""
    def __init__(self, config):
        super().__init__(config)
        self.load_allocated = self.load_reserved = 0

    def reset_peak(self):
        self.synchronize()
        self.load_allocated = max(self.load_allocated, self.torch.cuda.max_memory_allocated())
        self.load_reserved = max(self.load_reserved, self.torch.cuda.max_memory_reserved())
        super().reset_peak()

    def memory(self):
        result = super().memory()
        result["torch_cuda_peak_allocated_bytes"] = max(result["torch_cuda_peak_allocated_bytes"], self.load_allocated)
        result["torch_cuda_peak_reserved_bytes"] = max(result["torch_cuda_peak_reserved_bytes"], self.load_reserved)
        return result


class Monitor:
    """100ms RSS/device sampling plus boundary allocator peaks; CPU bootstrap included."""
    def __init__(self, runtime, prior, started, policy):
        import psutil
        self.process = psutil.Process()
        self.runtime, self.prior, self.started = runtime, prior, started
        self.limits = policy["resource_plan"]
        self.stop, self.lock = threading.Event(), threading.Lock()
        self.rss = self.device = self.allocated = self.reserved = self.samples = 0
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        with self.lock:
            self.rss = max(self.rss, self.process.memory_info().rss)
            if self.runtime is not None:
                free, total = self.runtime.torch.cuda.mem_get_info()
                self.device = max(self.device, total-free)
            self.samples += 1

    def loop(self):
        while not self.stop.wait(.1):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)
                return

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def check(self):
        self.sample()
        if self.runtime is not None:
            memory = self.runtime.memory()
            self.allocated = max(self.allocated, memory["torch_cuda_peak_allocated_bytes"])
            self.reserved = max(self.reserved, memory["torch_cuda_peak_reserved_bytes"])
        if self.error:
            raise RuntimeError(f"D-stage memory monitor failed: {self.error}")
        if (self.rss > self.limits["rss_GiB"]*1024**3
                or self.reserved > self.limits["selection_and_inference_reserved_GiB"]*1024**3
                or self.prior+time.perf_counter()-self.started > self.limits["comparison_seconds"]):
            raise RuntimeError("D-stage cumulative time or memory cap exceeded; no automatic shrinking")

    def snapshot(self):
        return {"rss_sample_peak_bytes": self.rss, "device_used_sample_peak_bytes": self.device,
                "torch_cuda_peak_allocated_bytes": self.allocated, "torch_cuda_peak_reserved_bytes": self.reserved,
                "samples": self.samples, "sampling_interval_ms": 100,
                "scope": "sampled process RSS/device use; allocator peaks; CPU bootstrap uses no GPU"}

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if (self.thread.is_alive() or self.error) and args[0] is None:
            raise RuntimeError("D-stage sampler failed to finish cleanly")


def read_value(root, name, signature):
    fit.read_phase(root, name, signature, ["value.json"])
    return json.loads((root/name/"value.json").read_bytes())


def attempts(root):
    directory = root/"attempts"
    directory.mkdir(exist_ok=True)
    preparation.read_attempts(directory)  # refuses unaccounted forced termination
    result = []
    for folder in sorted(directory.glob("attempt-*")):
        record = json.loads((folder/"finished.json").read_bytes())
        record["attempt"] = folder.name
        result.append(record)
    return result


def score_photo(source, photo, runtime, method, recovered):
    runtime.synchronize()
    started = time.perf_counter()
    sample = {k: photo[k] for k in ("key", "photo_id", "label")}
    sample["regions"] = []
    for row in photo["regions"]:
        reader = p.c.selection.spatial.ev.read_tile
        image = reader(source, row, recovered) if photo["key"].startswith(("good/", "defect/")) else fit.checked_image(source, row)
        try:
            raw, values = runtime.predict(image)
            values = np.asarray(values)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid anomaly map")
            sample["regions"].append({"region": row["region"], "member": row["member"],
                "top1": float(fit.candidate_image_score(raw, values, fit.CANDIDATE)), "native_raw": float(raw)})
        finally:
            image.close()
    decision = p.c.scoring.score(sample, method)
    runtime.synchronize()
    seconds = time.perf_counter()-started
    p.c.scoring.old.validate_sample(sample, photo)
    return sample, decision, seconds


def replay(actual, expected, method):
    p.c.scoring.old.validate_sample(actual, expected)
    delta = max(abs(a[k]-b[k]) for a, b in zip(actual["regions"], expected["regions"], strict=True)
                for k in ("top1", "native_raw"))
    flip = p.c.scoring.score(actual, method)["alert"] != p.c.scoring.score(expected, method)["alert"]
    if delta > 1e-4 or flip:
        raise ValueError(f"Frozen model replay failed: difference={delta}, decision_changed={flip}")
    return delta


def normal_gate(z, runtime, monitor, source, records, model, method):
    expected = source["reference"] if model == "baseline" else records["first.json"]
    samples, deltas, times = [], [], []
    for photo, reference in zip(source["cohorts"]["development"][:16], expected["development"][:16], strict=True):
        monitor.check()
        sample, _, seconds = score_photo(z, photo, runtime, method, None)
        deltas.append(replay(sample, reference, method))
        samples.append(sample)
        times.append(seconds)
        monitor.check()
    return {"passed": True, "photos": samples, "maximum_raw_difference": max(deltas),
            "decision_changes": 0, "first_photo_seconds": times[0], "photo_seconds": times,
            "contract_refitted": False}


def read_cached(root, name, signature, photo, env):
    receipt = fit.read_phase(root/"scores", name, signature, ["sample.json"])
    if receipt["photo_sha256"] != fit.digest(photo) or receipt["environment"] != env:
        raise ValueError("Cached photo provenance differs")
    gate_path = (root/receipt["gate_path"]).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt["gate_sha256"]:
        raise ValueError("Cached score lacks unchanged runtime gate")
    gate = json.loads(gate_path.read_bytes())
    if not gate["passed"] or gate["environment"] != env:
        raise ValueError("Cached runtime gate failed")
    sample = json.loads((root/"scores"/name/"sample.json").read_bytes())
    p.c.scoring.old.validate_sample(sample, photo)
    return sample


def collect(root, attempt, z, source, records, signature, env, runtime, monitor, method, recovered):
    (root/"scores").mkdir(exist_ok=True)
    expected_names = {f"{g}-{i:03d}" for g in ("test", "validation") for i in range(len(source["cohorts"][g]))}
    if any(f.name not in expected_names and not f.name.startswith(".") for f in (root/"scores").iterdir()):
        raise ValueError("Unexpected score cache entries")
    result, write_seconds = dict(records["first.json"]), 0.
    for group in ("test", "validation"):
        result[group] = []
        for i, photo in enumerate(source["cohorts"][group]):
            monitor.check()
            name = f"{group}-{i:03d}"
            if not (root/"scores"/name).exists():
                sample, _, _ = score_photo(z, photo, runtime, method, recovered)
                monitor.check()
                gate = attempt/"gate.json"
                began_write = time.perf_counter()
                with fit.phase_stage(root/"scores", name) as stage:
                    fit.write_json(stage/"sample.json", sample)
                    fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photo), "environment": env,
                        "gate_path": gate.relative_to(root).as_posix(), "gate_sha256": fit.sha256_file(gate)}, ["sample.json"])
                write_seconds += time.perf_counter()-began_write
            result[group].append(read_cached(root, name, signature, photo, env))
            if (i+1) % 10 == 0 or i+1 == len(source["cohorts"][group]):
                print(f"candidate {group} {i+1}/{len(source['cohorts'][group])}", flush=True)
    p.validate_raw(result, source["cohorts"])
    return result, write_seconds


def benchmark(z, source, records, runtime, monitor, model, method, candidate):
    photos = source["cohorts"]["validation"][:32]
    reference = source["reference"]["validation"] if model == "baseline" else candidate["validation"]
    for photo, expected in zip(photos[:5], reference[:5], strict=True):
        monitor.check()
        sample, _, _ = score_photo(z, photo, runtime, method, None)
        replay(sample, expected, method)
        monitor.check()
    measured = []
    for repeat in range(3):
        for photo, expected in zip(photos, reference[:32], strict=True):
            monitor.check()
            sample, decision, seconds = score_photo(z, photo, runtime, method, None)
            delta = replay(sample, expected, method)
            monitor.check()
            measured.append({"repeat": repeat, "key": photo["key"], "seconds": seconds,
                             "decision": decision, "replay_maximum_difference": delta})
        print(f"{model} warm repeat {repeat+1}/3 complete", flush=True)
    seconds = [m["seconds"] for m in measured]
    p95 = float(np.quantile(seconds, .95, method="higher"))
    return {"model": model, "process_id": os.getpid(), "warmup_photos": 5, "measurements": measured,
            "count": 96, "mean_seconds": float(np.mean(seconds)), "p95_higher_seconds": p95,
            "passed": p95 <= 2, "scope": "read_decode_preprocess_synchronized_inference_map_transfer_photo_scoring",
            "excludes": ["model_load", "normal_gate", "warmup", "output_file_writes", "http", "queue_wait", "boundary_checks"]}


def run_job(job, root, archive, source, records, identity, checkpoints, recovered_path,
            runtime_factory=EvaluationRuntime, monitor_factory=Monitor, environment_fn=preparation.environment):
    signature = fit.digest(identity)
    with fit.locked_run(root/"compute-lock"):
        previous = attempts(root)
        if (root/job).exists():
            result = read_value(root, job, signature)
            matched = [a for a in previous if a["attempt"] == result["attempt"] and a["job"] == job and a["outcome"] == "completed"]
            if len(matched) != 1:
                raise ValueError("Completed job lacks cost receipt")
            return result
        prior = sum(a["seconds"] for a in previous)
        if prior >= identity["policy"]["resource_plan"]["comparison_seconds"]:
            raise RuntimeError("D-stage cumulative time exhausted; no budget reset")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root/"attempts"))
        started, runtime, monitor, error, value = time.perf_counter(), None, None, None, None
        write_seconds = 0.
        try:
            if job != "bootstrap":
                runtime = runtime_factory(source["config"])
            with monitor_factory(runtime, prior, started, identity["policy"]) as monitor:
                if job == "bootstrap":
                    candidate = read_value(root, "score", signature)["raw"]
                    p.validate_raw(candidate, source["cohorts"])
                    comparison, predictions = p.compare(source["reference"], candidate, identity["methods"])
                    sensitivity = p.bootstrap(source["reference"], candidate, identity["policy"], monitor.check)
                    value = {"comparison": comparison, "predictions": predictions, "bootstrap": sensitivity}
                else:
                    model = "baseline" if job == "baseline-warm" else "candidate"
                    context = source["records"]["fit.json"]["checkpoint_context"] if model == "baseline" else records["training.json"]["context"]
                    before = time.perf_counter()
                    runtime.load(checkpoints[model], context)
                    load_seconds = time.perf_counter()-before
                    env = environment_fn(runtime)
                    if env != records["environment.json"]:
                        raise ValueError("C-stage and D-stage environments differ")
                    monitor.check()
                    with zipfile.ZipFile(archive) as z:
                        names = Counter(z.namelist())
                        if any(names[r["member"]] != 1 for photos in source["cohorts"].values() for photo in photos for r in photo["regions"]):
                            raise ValueError("Missing/duplicate frozen image ZIP members")
                        method = identity["methods"][model]
                        gate = normal_gate(z, runtime, monitor, source, records, model, method)
                        gate.update(environment=env, model=model, model_load_seconds=load_seconds, process_id=os.getpid())
                        began_write = time.perf_counter()
                        fit.write_json(attempt/"gate.json", gate)
                        write_seconds += time.perf_counter()-began_write
                        if job == "score":
                            recovered = p.c.selection.spatial.ev.recovered_bytes(recovered_path, source["previous"])
                            raw, cache_write_seconds = collect(root, attempt, z, source, records, signature, env, runtime, monitor, method, recovered)
                            write_seconds += cache_write_seconds
                            value = {"raw": raw}
                        else:
                            candidate = read_value(root, "score", signature)["raw"]
                            value = benchmark(z, source, records, runtime, monitor, model, method, candidate)
                        value["gate"] = gate
                monitor.check()
                value["attempt"] = attempt.name
                # Preserve completed computation before release; publish only after cost accounting succeeds.
                began_write = time.perf_counter()
                fit.write_json(attempt/"computed.json", value)
                write_seconds += time.perf_counter()-began_write
                monitor.check()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if monitor is not None:
                try:
                    monitor.check()  # retain allocator peaks even when the operation failed
                except Exception:
                    pass  # preserve the original failure; snapshot contains last readable peaks
            raise
        finally:
            try:
                if runtime is not None:
                    runtime.release()
            except BaseException as exc:
                error = error or f"Release failed: {exc}"
                raise
            finally:
                elapsed = time.perf_counter()-started
                cost = {"job": job, "seconds": elapsed, "process_id": os.getpid(), "error": error,
                        "outcome": "failed" if error else "completed", "memory": monitor.snapshot() if monitor else None,
                        "artifact_write_seconds": write_seconds,
                        "artifact_write_scope": "successful gate/cache/computed JSON writes; excludes final receipts/publication/export; partial failed writes not isolated",
                        "scope": "runtime setup/load, gate, scoring/benchmark/bootstrap, cache I/O, release; excludes input hashing, final receipt/publication/export and offline waiting"}
                if not error and prior+elapsed > identity["policy"]["resource_plan"]["comparison_seconds"]:
                    cost.update(error="D-stage time cap exceeded including release", outcome="failed")
                fit.write_json(attempt/"finished.json", cost)
        if cost["outcome"] != "completed":
            raise RuntimeError(cost["error"])
        c_runner.persist(root, job, signature, value)
        return value


def export(root, records, identity, handoff):
    signature = fit.digest(identity)
    values = {job: read_value(root, job, signature) for job in JOBS}
    costs = attempts(root)
    for job, value in values.items():
        if not any(a["job"] == job and a["attempt"] == value["attempt"] and a["outcome"] == "completed" for a in costs):
            raise ValueError("Job cost receipt missing")
    resource_errors = [a for a in costs if a["error"]]
    bootstrap = values["bootstrap"]
    gates = {"fixed_comparison": bootstrap["comparison"]["passed"], "normal_sensitivity": bootstrap["bootstrap"]["passed"],
             "baseline_warm_p95": values["baseline-warm"]["passed"], "candidate_warm_p95": values["candidate-warm"]["passed"],
             "D_attempts_without_errors": not resource_errors}
    report = {"status": "normal_diversity_D_comparison_completed_exploratory",
              "comparison": bootstrap["comparison"], "bootstrap_summary": bootstrap["bootstrap"]["summary"],
              "warm_summary": {m: {k: values[m+"-warm"][k] for k in ("count", "mean_seconds", "p95_higher_seconds", "passed")}
                               for m in ("baseline", "candidate")},
              "D_gates": gates, "D_passed": all(gates.values()), "D_cumulative_seconds": sum(a["seconds"] for a in costs),
              "C_resource_review_required": records["report.json"]["resource_review_required"],
              "overall_candidate_accepted": False, "automatic_pass_allowed": False, "service_model_changed": False,
              "new_independent_evaluation": False, "training_performed": False, "test_masks_decoded": 0,
              "note": "C-stage 7 GiB violation remains unresolved even if all D gates pass. Human review required."}
    payload = {"plan.json": identity, "report.json": report, "C-report.json": records["report.json"],
               "C-attempts.json": records["attempts.json"], "D-attempts.json": costs,
               **{job+".json": value for job, value in values.items()}}
    if not (root/"result").exists():
        with fit.phase_stage(root, "result") as stage:
            for name, value in payload.items():
                fit.write_json(stage/name, value)
            bundle.seal(stage, identity)
    bundle.verify(root/"result", identity)
    for name, value in payload.items():
        if json.loads((root/"result"/name).read_bytes()) != value:
            raise ValueError("Published result differs; preserved")
    bundle.bundle(root/"result", handoff)
    return report


def worker_command(args, job):
    return [sys.executable, str(Path(__file__).resolve()), "--project", str(args.project),
            "--archive", str(args.archive), "--handoffs", str(args.handoffs), "--job", job]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, default=Path.home()/"Downloads/extracted.zip")
    parser.add_argument("--handoffs", type=Path)
    parser.add_argument("--job", choices=JOBS)
    args = parser.parse_args(argv)
    args.project = args.project.resolve()
    args.handoffs = (args.handoffs or args.project/"artifacts/handoff").resolve()
    root = args.project/"artifacts/experiments/mpi-normal-diversity-v1/evaluation"
    try:
        source, records, identity = p.prepare(args.project, args.handoffs)
        checkpoints = {"baseline": args.project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt",
                       "candidate": args.project/"artifacts/experiments/mpi-normal-diversity-v1/fit/training/model.pt"}
        for model, path in checkpoints.items():
            expected = p.CHECKPOINT_SHA if model == "candidate" else identity["policy"]["source_sha256"]["baseline_checkpoint"]
            if fit.sha256_file(path) != expected:
                raise ValueError(f"Frozen {model} checkpoint differs")
        if fit.sha256_file(args.archive) != identity["policy"]["source_sha256"]["archive"]:
            raise ValueError("MPI archive differs")
        recovered = args.project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png"
        p.c.selection.spatial.ev.recovered_bytes(recovered, source["previous"])
        root.mkdir(parents=True, exist_ok=True)
        if args.job:
            read_value(root, "identity", fit.digest(identity))
            run_job(args.job, root, args.archive, source, records, identity, checkpoints, recovered)
        else:
            with fit.locked_run(root):
                c_runner.persist(root, "identity", fit.digest(identity), identity)
                for index, job in enumerate(JOBS, 1):
                    print(f"{index}/4 D-stage {job}; no fit, tuning, automatic PASS or independent-final claim", flush=True)
                    subprocess.run(worker_command(args, job), check=True)
                report = export(root, records, identity, args.project/"artifacts/handoff"/(p.NAME+".zip"))
                print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}\nFiles preserved. Do not delete outputs, retrain, change thresholds or reset budgets.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
