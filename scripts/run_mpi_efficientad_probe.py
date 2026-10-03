"""Normal-only EfficientAD-S feasibility probe. No calibration or defect access."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import fit_mpi_pcb_normal_bank as fit
from scripts import mpi_efficientad_probe_assets as assets
from scripts.mpi_efficientad_probe_runtime import Runtime, environment
from scripts.evaluate_mpi_normal_diversity import Monitor

NAME = "mpi-efficientad-probe-v1"
POLICY = {"purpose": "normal_only_resource_probe", "seed": 42, "model_size": "small",
          "batch_size": 1, "image_size": 256, "normal_photos": 8, "tiles": 96,
          "optimizer_updates": 100, "checkpoint_interval": 25,
          "lr": .0001, "weight_decay": .00001, "lr_schedule": "constant_probe_only",
          "map_normalization": False, "reload_absolute_tolerance": 1e-4,
          "resource_plan": {"comparison_seconds": 900, "rss_GiB": 20,
                            "selection_and_inference_reserved_GiB": 6},
          "full_training_authorized": False, "test_images_decoded": 0,
          "automatic_pass_allowed": False}


def prepare(project, preparation_path):
    config = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    report = fit.load_preparation(preparation_path, config)
    ids = sorted(report["training_photo_ids"], key=lambda p:
                 hashlib.sha256((NAME + "\0" + p).encode()).hexdigest())[:8]
    by_key = {(r["photo_id"], r["region"]): r for r in report["training_rows"]}
    rows = [by_key[p, region] for p in ids for region in fit.REGIONS]
    # Changes to an imported helper invalidate resuming an old probe.
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    return {"name": NAME, "policy": POLICY, "photo_ids": ids, "rows": rows,
            "preparation_sha256": fit.PREPARATION_SHA256,
            "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
            "sources_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}}


def publish(root, phase, signature, payload):
    if (root / phase).exists():
        fit.read_phase(root, phase, signature, ["value.json"])
        if json.loads((root / phase / "value.json").read_bytes()) != payload:
            raise ValueError(f"Existing {phase} differs; preserved")
    else:
        with fit.phase_stage(root, phase) as stage:
            fit.write_json(stage / "value.json", payload)
            fit.seal_phase(stage, phase, signature, {}, ["value.json"])


def checkpoint_steps(root, signature):
    steps = []
    for step in range(0, 101, 25):
        if (root / f"step-{step:03d}").exists():
            if step and step - 25 not in steps:
                raise ValueError("Checkpoint gap; preserved")
            fit.read_phase(root, f"step-{step:03d}", signature,
                           ["model.pt", "reference.npy", "progress.json"])
            progress = json.loads((root / f"step-{step:03d}/progress.json").read_bytes())
            if progress["step"] != step or len(progress["losses"]) != step:
                raise ValueError("Checkpoint progress mismatch")
            steps.append(step)
    return steps


def attempts(root):
    result = []
    for path in sorted(root.glob("attempt-*")):
        if not (path / "finished.json").exists():
            raise ValueError("Unfinished attempt requires review; do not delete files")
        record = json.loads((path / "finished.json").read_bytes())
        if record["status"] not in ("completed", "interrupted"):
            raise ValueError("Prior failed/resource-limited attempt needs review; no automatic retry")
        result.append(record)
    return result


def export(root, handoff, report, identity, previous):
    """A JSON-only evidence handoff. Never include images, weights or checkpoints."""
    payload = {"report.json": report, "plan.json": identity, "attempts.json": previous,
               "progress.json": json.loads((root / "step-100/progress.json").read_bytes()),
               "checkpoint-receipts.json": [json.loads(p.read_bytes()) for p in sorted(root.glob("step-*/receipt.json"))]}
    raw = {k: fit.canonical(v) for k, v in payload.items()}
    raw["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if set(z.namelist()) != set(raw) or any(z.read(k) != v for k, v in raw.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".efficientad-export-", dir=handoff.parent)) / handoff.name
    with zipfile.ZipFile(staged, "x", zipfile.ZIP_DEFLATED) as z:
        for name, value in raw.items():
            z.writestr(name, value)
    staged.rename(handoff)


def execute(root, archive, identity, asset_root, handoff, runtime_factory=Runtime, monitor_factory=Monitor):
    signature = fit.digest(identity)
    rows = identity["rows"]
    with fit.locked_run(root):
        if not (root / "identity").exists() and (list(root.glob("step-*")) or list(root.glob("attempt-*"))):
            raise ValueError("Artifacts without identity; preserved")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        available = checkpoint_steps(root, signature)
        if (root / "result").exists():
            fit.read_phase(root, "result", signature, ["value.json"])
            if available != [0, 25, 50, 75, 100]:
                raise ValueError("Completed run has missing checkpoints")
            report = json.loads((root / "result/value.json").read_bytes())
            export(root, handoff, report, identity, previous)
            return report
        prior = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        if prior >= 900:
            raise ValueError("Cumulative GPU probe time cap exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        runtime = None
        status, error, updates, last = "failed", None, 0, available[-1] if available else 0
        memory = {}
        timings = {}
        reloads = []
        monitor = monitor_factory(None, prior, started, POLICY)
        try:
            with monitor, zipfile.ZipFile(archive) as z:
                runtime = runtime_factory(asset_root / "teacher.pth", asset_root)
                monitor.runtime = runtime
                monitor.check()

                def image(index):
                    return fit.checked_image(z, rows[index % 96])

                def predict_reference():
                    with image(0) as img:
                        return runtime.predict(img)

                def reload_check(step):
                    runtime.load(root / f"step-{step:03d}/model.pt", signature, step)
                    values = predict_reference()
                    expected = np.load(root / f"step-{step:03d}/reference.npy", allow_pickle=False)
                    if expected.shape != values.shape or not np.isfinite(expected).all():
                        raise ValueError("Checkpoint reference map invalid")
                    difference = float(np.max(np.abs(expected - values)))
                    if not np.isfinite(difference) or difference > 1e-4:
                        raise ValueError(f"Probe reload map difference: {difference}")
                    reloads.append({"step": step, "maximum_raw_map_difference": difference,
                                    "optimizer_and_rng_restored": True})
                    monitor.check()

                def save(step):
                    monitor.check()
                    with fit.phase_stage(root, f"step-{step:03d}") as stage:
                        # predict is eval-only and consumes no RNG; checkpoint restores both CPU and CUDA RNG.
                        values = predict_reference()
                        runtime.save(stage / "model.pt", signature, step)
                        np.save(stage / "reference.npy", values, allow_pickle=False)
                        fit.write_json(stage / "progress.json", {"step": step, "losses": losses,
                                                                  "statistics": statistics})
                        fit.seal_phase(stage, f"step-{step:03d}", signature, {},
                                       ["model.pt", "reference.npy", "progress.json"])
                    monitor.check()

                if available:
                    progress = json.loads((root / f"step-{last:03d}/progress.json").read_bytes())
                    losses, statistics = progress["losses"], progress["statistics"]
                    reload_check(last)
                else:
                    statistics = runtime.statistics((image(i) for i in range(96)), monitor.check)
                    losses = []
                    save(0)
                timings["setup_and_statistics_seconds"] = time.perf_counter() - started
                train_start = time.perf_counter()
                print(f"Normal-only probe: starting at update {last}/100. No test access.", flush=True)
                for index in range(last, 100):
                    monitor.check()
                    if prior_updates + updates >= 100:
                        raise RuntimeError("100 attempted-update cap reached after interruption; review required")
                    updates += 1
                    with image(index) as img:
                        losses.append(runtime.step(img, index))
                    monitor.check()
                    if (index + 1) % 25 == 0:
                        save(index + 1)
                        last = index + 1
                        print(f"probe update {last}/100 saved", flush=True)
                        if last in (50, 100):
                            reload_check(last)
                timings["updates_checkpoints_reload_seconds"] = time.perf_counter() - train_start
                if not any(r["step"] == 100 for r in reloads):
                    reload_check(100)
                infer_start = time.perf_counter()
                for i in range(12):
                    monitor.check()
                    with image(i) as img:
                        runtime.predict(img)
                timings["one_12_tile_photo_seconds"] = time.perf_counter() - infer_start
                monitor.check()
                memory = monitor.snapshot()
            status = "completed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            memory = monitor.snapshot()
            memory["scope"] = "100ms sampled process RSS and whole device use; CUDA allocator lifetime peaks during probe"
            record = {"status": status, "error": error, "seconds": time.perf_counter() - started,
                      "attempted_updates": updates, "latest_saved_step": last, "memory": memory,
                      "timings": timings, "reload_checks": reloads}
            fit.write_json(attempt / "finished.json", record)
        previous.append(record)
        report = {"status": "normal_only_resource_probe_completed", "model": "EfficientAD-S",
                  "purpose": "Compatibility/resource evidence, not accuracy or production readiness",
                  "identity_sha256": signature, "completed_updates": 100,
                  "cumulative_attempted_updates": sum(a["attempted_updates"] for a in previous),
                  "cumulative_probe_seconds": sum(a["seconds"] for a in previous),
                  "normal_photos": 8, "unique_normal_tiles": 96, "auxiliary_train_images": 96,
                  "environment": identity["environment"], "timings": timings,
                  "memory": memory, "reload_checks": reloads,
                  "thresholds_calculated": False, "test_images_decoded": 0, "test_masks_decoded": 0,
                  "accuracy_measured": False, "formal_training_started": False,
                  "automatic_pass_allowed": False, "service_model_changed": False}
        publish(root, "result", signature, report)
        export(root, handoff, report, identity, previous)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    preparation = args.preparation or project / "artifacts/handoff/mpi-pcb-normal-bank-preparation-v1.zip"
    plan = prepare(project, preparation)
    plan["environment"] = environment()
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original MPI archive hash differs")
    asset_cache = project / "artifacts/research-assets/efficientad-v1"
    if shutil.disk_usage(project).free < 6 * 1024**3:
        raise ValueError("At least 6 GiB free disk required for research assets and probe checkpoints")
    with fit.locked_run(asset_cache):
        asset_root, manifest, seconds = assets.prepare(asset_cache, args.allow_download)
    plan["assets"] = manifest
    print(f"Assets verified/prepared in {seconds:.1f}s; this is separate from the 15-minute GPU cap.", flush=True)
    report = execute(project / "artifacts" / NAME, args.archive, plan, asset_root,
                     project / "artifacts/handoff" / f"{NAME}.zip")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Files preserved; no full training or test.", file=sys.stderr)
        raise SystemExit(1)
