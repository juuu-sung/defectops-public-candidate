"""Phase 1: 100-update, normal-only SuperSimpleNet GPU feasibility probe."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import fit_mpi_pcb_normal_bank as fit
from scripts import mpi_supersimplenet_protocol as p
from scripts.mpi_supersimplenet_runtime import Runtime, environment
from scripts.mpi_efficientad_probe_assets import download
from scripts.run_mpi_efficientad_probe import publish, checkpoint_steps, attempts, export
from scripts.evaluate_mpi_normal_diversity import Monitor


def prepare(project, preparation_path):
    report = fit.load_preparation(preparation_path,
                                  fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml"))
    ids = sorted(report["training_photo_ids"], key=lambda x:
                 hashlib.sha256((p.NAME + "\0" + x).encode()).hexdigest())[:8]
    by_key = {(r["photo_id"], r["region"]): r for r in report["training_rows"]}
    rows = [by_key[photo, region] for photo in ids for region in fit.REGIONS]
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    return {"name": p.NAME, "policy": p.POLICY, "photo_ids": ids, "rows": rows,
            "preparation_sha256": fit.PREPARATION_SHA256,
            "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
            "weights": p.WEIGHTS,
            "batch_schedule_sha256": fit.digest([p.batch_indices(i) for i in range(100)]),
            "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
                f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in files}}


def execute(root, archive, identity, weights, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    rows = identity["rows"]
    if len(rows) != 96 or identity["policy"] != p.POLICY:
        raise ValueError("Probe data/policy mismatch")
    with fit.locked_run(root):
        if not (root / "identity").exists() and (list(root.glob("step-*")) or list(root.glob("attempt-*"))):
            raise ValueError("Artifacts without identity; preserved")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        available = checkpoint_steps(root, signature)
        if (root / "result").exists():
            fit.read_phase(root, "result", signature, ["value.json"])
            if available != [0, 25, 50, 75, 100]:
                raise ValueError("Missing completed checkpoints")
            result = json.loads((root / "result/value.json").read_bytes())
            export(root, handoff, result, identity, previous)
            return result
        prior = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        if prior >= 900:
            raise ValueError("Probe cumulative time cap exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        monitor = monitor_factory(None, prior, started, p.POLICY)
        runtime = None
        status, error, updates, last = "failed", None, 0, available[-1] if available else 0
        timings, reloads, checks = {}, [], {}
        try:
            with monitor, zipfile.ZipFile(archive) as source:
                runtime = runtime_factory(weights)
                monitor.runtime = runtime
                monitor.check()

                def predict(indices):
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(source, rows[i])) for i in indices]
                        return runtime.predict(images)

                def save(step):
                    monitor.check()
                    with fit.phase_stage(root, f"step-{step:03d}") as stage:
                        values = predict([0])
                        runtime.save(stage / "model.pt", signature, step)
                        np.save(stage / "reference.npy", values, allow_pickle=False)
                        fit.write_json(stage / "progress.json", {"step": step, "losses": losses,
                            "initial_fingerprints": initial, "current_fingerprints": runtime.fingerprints()})
                        fit.seal_phase(stage, f"step-{step:03d}", signature, {},
                                       ["model.pt", "reference.npy", "progress.json"])
                    monitor.check()

                def reload_check(step):
                    runtime.load(root / f"step-{step:03d}/model.pt", signature, step)
                    expected = np.load(root / f"step-{step:03d}/reference.npy", allow_pickle=False)
                    delta = p.replay_difference(predict([0]), expected)
                    progress = json.loads((root / f"step-{step:03d}/progress.json").read_bytes())
                    if runtime.fingerprints() != progress["current_fingerprints"]:
                        raise ValueError("Reload tensor fingerprints differ")
                    reloads.append({"step": step, **delta, "optimizer_and_all_rng_restored": True})
                    monitor.check()

                if available:
                    progress = json.loads((root / f"step-{last:03d}/progress.json").read_bytes())
                    losses, initial = progress["losses"], progress["initial_fingerprints"]
                    reload_check(last)
                else:
                    losses, initial = [], runtime.fingerprints()
                    save(0)
                timings["setup_seconds"] = time.perf_counter() - started
                train_start = time.perf_counter()
                print(f"Normal-only SuperSimpleNet: update {last}/100; batch8 (effective16); no test access.", flush=True)
                for index in range(last, 100):
                    monitor.check()
                    if prior_updates + updates >= 100:
                        raise RuntimeError("100 attempted-update cap reached; review required")
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(source, rows[i])) for i in p.batch_indices(index)]
                        updates += 1  # Count interrupted/failed GPU attempts too.
                        step_started = time.perf_counter()
                        loss_record = runtime.step(images, index)
                        loss_record["update_seconds_including_checks"] = time.perf_counter()-step_started
                        losses.append(loss_record)
                    monitor.check()
                    if (index+1) % 25 == 0:
                        save(index+1)
                        last = index+1
                        print(f"probe update {last}/100 saved", flush=True)
                        if last in (50, 100):
                            reload_check(last)
                timings["updates_checkpoints_reload_seconds"] = time.perf_counter() - train_start
                if not any(r["step"] == 100 for r in reloads):
                    reload_check(100)
                final = runtime.fingerprints()
                if final["backbone"] != initial["backbone"]:
                    raise ValueError("Frozen backbone weights/buffers changed")
                if any(final[k] == initial[k] for k in ("adaptor", "segmentation", "classification")):
                    raise ValueError("A trainable component did not change")
                synthetic_count = sum(x["synthetic_anomalous_samples"] for x in losses)
                if not 0 < synthetic_count < 1600:
                    raise ValueError("Probe did not exercise both normal and synthetic classes")
                # Verify both the squeezed batch1 and batch8 eval paths without demanding
                # cross-batch bit equality (different convolution kernels may be used).
                eight = predict(list(range(8)))
                if eight.shape != (8, 65537) or not np.isfinite(eight).all():
                    raise ValueError("Batch8 inference contract failed")
                checks = {"batch1_inference": True, "batch8_inference": True,
                          "backbone_unchanged": True, "all_trainable_components_changed": True,
                          "synthetic_anomalous_presentations": synthetic_count,
                          "effective_feature_presentations": 1600,
                          "initial_fingerprints": initial, "final_fingerprints": final}
                monitor.check()
                warm_start = time.perf_counter()
                for i in range(12):
                    monitor.check()
                    predict([i])
                timings["one_12_tile_photo_seconds"] = time.perf_counter()-warm_start
                timings["latency_scope"] = "decode+hash verification+preprocess+inference+CPU map transfer; one photo, not p95"
                monitor.check()
            status = "completed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Include allocator peaks even if an exception occurred before the next check.
            if runtime is not None:
                try:
                    m = runtime.memory()
                    monitor.allocated = max(getattr(monitor, "allocated", 0), m["torch_cuda_peak_allocated_bytes"])
                    monitor.reserved = max(getattr(monitor, "reserved", 0), m["torch_cuda_peak_reserved_bytes"])
                except Exception:
                    pass
            if runtime is None:
                # Constructor OOM must not be reported as zero GPU consumption.
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    try:
                        monitor.allocated = max(getattr(monitor, "allocated", 0), torch.cuda.max_memory_allocated())
                        monitor.reserved = max(getattr(monitor, "reserved", 0), torch.cuda.max_memory_reserved())
                    except Exception:
                        pass
            record = {"status": status, "error": error, "seconds": time.perf_counter()-started,
                      "attempted_updates": updates, "latest_saved_step": last,
                      "memory": monitor.snapshot(), "timings": timings, "reload_checks": reloads}
            fit.write_json(attempt / "finished.json", record)
        previous.append(record)
        memory = dict(record["memory"])
        for key in ("rss_sample_peak_bytes", "device_used_sample_peak_bytes",
                    "torch_cuda_peak_allocated_bytes", "torch_cuda_peak_reserved_bytes"):
            if key in memory:
                memory[key] = max(a["memory"][key] for a in previous)
        result = {"status": "normal_only_resource_probe_completed", "model": "SuperSimpleNet",
                  "identity_sha256": signature, "completed_updates": 100,
                  "cumulative_attempted_updates": sum(a["attempted_updates"] for a in previous),
                  "cumulative_probe_seconds": sum(a["seconds"] for a in previous),
                  "normal_photos": 8, "unique_normal_tiles": 96,
                  "input_batch": 8, "effective_training_feature_batch": 16,
                  "environment": identity["environment"], "timings": timings,
                  "memory": memory, "reload_checks": reloads, "checks": checks,
                  "thresholds_calculated": False, "test_images_decoded": 0, "test_masks_decoded": 0,
                  "accuracy_measured": False, "formal_training_started": False,
                  "automatic_pass_allowed": False, "service_model_changed": False}
        publish(root, "result", signature, result)
        export(root, handoff, result, identity, previous)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    started = time.perf_counter()
    identity = prepare(project, args.preparation or project / "artifacts/handoff/mpi-pcb-normal-bank-preparation-v1.zip")
    identity["environment"] = environment()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MPI archive SHA256 differs")
    if shutil.disk_usage(project).free < 6 * 1024**3:
        raise ValueError("At least 6 GiB free disk required for five checkpoints and weights")
    assets = project / "artifacts/research-assets/supersimplenet-v1"
    with fit.locked_run(assets):
        weights = download(assets, p.WEIGHTS, args.allow_download)
    print(f"Input/weight verification took {time.perf_counter()-started:.1f}s (outside GPU 15-minute cap).", flush=True)
    result = execute(project / "artifacts" / p.NAME, args.archive, identity, weights,
                     project / "artifacts/handoff" / f"{p.NAME}.zip")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all outputs; no formal fit or test.", file=sys.stderr)
        raise SystemExit(1)
