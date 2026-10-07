"""Read-only data contract plus synthetic CUDA feasibility; no model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_patchcore_scaling_probe as p
from scripts.mpi_patchcore_scaling_probe_runtime import Runtime, environment
from scripts.mpi_supersimplenet_scaling_evaluation_runtime import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit


def value(root, name, signature):
    fit.read_phase(root, name, signature, ["value.json"])
    return json.loads((root/name/"value.json").read_bytes())


def export(root, handoff, signature, identity, report):
    contents = {"plan.json": identity, "report.json": report, "attempts.json": attempts(root),
                **{name+".json": value(root, name, signature) for name in ("environment", "parity", "scale")}}
    raw = {name: fit.canonical(v) for name, v in contents.items()}
    raw["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw) or any(z.read(k) != v for k, v in raw.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".patchcore-probe-export-", dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage, "x", zipfile.ZIP_DEFLATED) as z:
        for name, content in raw.items(): z.writestr(name, content)
    stage.rename(handoff)


def execute(root, handoff, identity, runtime_factory=Runtime, environment_fn=environment, monitor_factory=Monitor):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output without identity; preserve for review")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        phases = {name: value(root, name, signature) for name in ("environment", "parity", "scale") if (root/name).exists()}
        if "scale" in phases and "parity" not in phases or phases and "environment" not in phases:
            raise ValueError("Incomplete provenance chain; preserved")
        if (root/"result").exists():
            if set(phases) != {"environment", "parity", "scale"}:
                raise ValueError("Completed probe missing verified phases")
            report = value(root, "result", signature)
            export(root, handoff, signature, identity, report)
            return report
        if shutil.disk_usage(root).free < p.POLICY["free_disk_GiB"]*1024**3:
            raise RuntimeError("Need 2 GiB free disk for this synthetic probe")
        folder = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(folder/"started.json", {"identity": signature, "unix_seconds": time.time()})
        started, status, error, runtime = time.perf_counter(), "failed", None, None
        monitor = monitor_factory(None, sum(a["seconds"] for a in previous), started, p.POLICY)
        try:
            with monitor:
                env = environment_fn()
                publish(root, "environment", signature, env)
                runtime = runtime_factory(); monitor.runtime = runtime
                monitor.check()
                for name in ("parity", "scale"):
                    if name not in phases:
                        print(f"Running {name}: synthetic only, no images/model/threshold changes", flush=True)
                        result = getattr(runtime, name)(monitor.check)
                        monitor.check()
                        if result.get("passed") is not True:
                            raise ValueError(f"{name} did not pass")
                        publish(root, name, signature, result)
                    elif phases[name].get("passed") is not True:
                        raise ValueError(f"Saved {name} did not pass")
                monitor.check()
            status = "completed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            fit.write_json(folder/"finished.json", {"status": status, "error": error,
                "seconds": time.perf_counter()-started, "memory": monitor.snapshot(),
                "identity": signature, "unfinished_stage_replay": "same fixed synthetic seed; completed phases reused"})
            if runtime is not None: runtime.release()
        all_attempts = attempts(root)
        report = {"status": "synthetic_probe_completed_needs_review_before_full_fit",
                  "identity": signature, "model_training_performed": False,
                  "images_decoded": 0, "test_scores_computed": 0,
                  "automatic_pass_allowed": False, "service_model_changed": False,
                  "elapsed_attempt_seconds": sum(a["seconds"] for a in all_attempts),
                  "scope": "Small installed-reference parity plus synthetic full candidate-size resource measurement. No real feature cache, full coreset, model, normal calibration or performance result.",
                  "full_fit_runner_ready": False}
        publish(root, "result", signature, report)
        export(root, handoff, signature, identity, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    try:
        identity = p.prepare(project, project/"artifacts/handoff/mpi-supersimplenet-scaling-v1.zip")
        report = execute(project/"artifacts"/p.NAME, project/"artifacts/handoff"/(p.NAME+".zip"), identity)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed phases preserved. Rerun this command with the same code and inputs.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve outputs; do not delete or alter limits.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
