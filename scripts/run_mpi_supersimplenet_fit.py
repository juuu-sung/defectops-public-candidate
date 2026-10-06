"""300-epoch SuperSimpleNet fit, normal-logit calibration and reload. No test reader."""
from __future__ import annotations

import argparse
from collections import Counter
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
from scripts import mpi_supersimplenet_fit_protocol as p
from scripts.mpi_supersimplenet_fit_runtime import Runtime, environment
from scripts.mpi_efficientad_probe_assets import download
from scripts.mpi_supersimplenet_protocol import WEIGHTS
from scripts.evaluate_mpi_normal_diversity import Monitor
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = p.fit
CHECKPOINT_FILES = ["model.pt", "reference.npy", "progress.json"]


def checkpoint_steps(root, signature, policy):
    steps = []
    interval, total = policy["checkpoint_interval"], policy["updates"]
    schedule = sorted(set(range(0, total + 1, interval)) | {total})
    expected_names = {f"step-{i:06d}" for i in schedule}
    if any(d.name not in expected_names for d in root.glob("step-*")):
        raise ValueError("Unexpected checkpoint directory; preserved")
    for position, step in enumerate(schedule):
        name = f"step-{step:06d}"
        if not (root / name).exists():
            continue
        if position and schedule[position-1] not in steps:
            raise ValueError("Checkpoint gap; preserved")
        fit.read_phase(root, name, signature, CHECKPOINT_FILES)
        progress = json.loads((root / name / "progress.json").read_bytes())
        if progress["step"] != step or [v["step"] for v in progress["losses"]] != list(range(1, step + 1)):
            raise ValueError("Checkpoint progress differs")
        steps.append(step)
    return steps


def normal_arrays(root, group, count):
    return [np.load(root / group / f"photo-{i:03d}" / "predictions.npy", mmap_mode="r", allow_pickle=False)
            for i in range(count)]


def export(root, handoff, identity, report):
    total = identity["policy"]["updates"]
    payload = {"report.json": report, "plan.json": identity, "attempts.json": attempts(root),
               "progress.json": json.loads((root / f"step-{total:06d}/progress.json").read_bytes()),
               "normal-results.json": json.loads((root / "calibrated/value.json").read_bytes()),
               "reload.json": json.loads((root / "reload/value.json").read_bytes()),
               "checkpoint-receipts.json": [json.loads(f.read_bytes()) for f in sorted(root.glob("step-*/receipt.json"))],
               "prediction-receipts.json": [json.loads(f.read_bytes()) for g in ("development", "calibration")
                                     for f in sorted((root / g).glob("photo-*/receipt.json"))]}
    raw = {k: fit.canonical(v) for k, v in payload.items()}
    raw["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw) or any(z.read(k) != v for k, v in raw.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".supersimplenet-fit-export-", dir=handoff.parent)) / handoff.name
    with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED) as z:
        for k, v in raw.items():
            z.writestr(k, v)
    temp.rename(handoff)


def execute(root, archive, identity, weights, handoff, runtime_factory=Runtime, monitor_factory=Monitor,
            protocol=p):
    # Explicit injection lets new signed experiments reuse the transaction engine,
    # without mutating the original protocol or changing its default runtime.
    p = protocol
    signature, policy, groups = fit.digest(identity), identity["policy"], identity["cohorts"]
    total, interval = policy["updates"], policy["checkpoint_interval"]
    per_epoch = policy.get("updates_per_epoch", 96)
    with fit.locked_run(root):
        if not (root / "identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output exists without identity; preserved")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        available = checkpoint_steps(root, signature, policy)
        counts = {g: len(groups[g]["photo_ids"]) for g in ("development", "calibration")}
        for g, count in counts.items():
            (root / g).mkdir(exist_ok=True)
            if any(d.name not in {f"photo-{i:03d}" for i in range(count)} for d in (root / g).glob("photo-*")):
                raise ValueError("Unexpected normal cache")
            for i in range(count):
                if (root / g / f"photo-{i:03d}").exists():
                    if total not in available:
                        raise ValueError("Normal calibration cache before completed fit")
                    fit.read_phase(root / g, f"photo-{i:03d}", signature, ["predictions.npy"])
        normal_done = all((root / g / f"photo-{i:03d}").exists() for g, n in counts.items() for i in range(n))
        if (root / "calibrated").exists():
            if not normal_done:
                raise ValueError("Calibration contract missing raw normal evidence")
            fit.read_phase(root, "calibrated", signature, ["value.json"])
        if (root / "reload").exists():
            if not (root / "calibrated").exists():
                raise ValueError("Reload before calibration")
            fit.read_phase(root, "reload", signature, ["value.json"])
        if (root / "result").exists():
            if not (root / "reload").exists():
                raise ValueError("Result without reload evidence")
            fit.read_phase(root, "result", signature, ["value.json"])
            report = json.loads((root / "result/value.json").read_bytes())
            export(root, handoff, identity, report)
            return report
        prior = sum(a["seconds"] for a in previous)
        prior_updates = sum(a["attempted_updates"] for a in previous)
        time_cap = policy["resource_plan"]["comparison_seconds"]
        if time_cap is not None and prior >= time_cap:
            raise RuntimeError("Cumulative time cap exhausted; review required")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        status, error, updates, last = "failed", None, 0, available[-1] if available else 0
        monitor = monitor_factory(None, prior, started, policy)
        runtime = None
        reloads = []
        try:
            with monitor, zipfile.ZipFile(archive) as z:
                inventory = Counter(z.namelist())
                if any(inventory[r["member"]] != 1 for group in groups.values() for r in group["rows"]):
                    raise ValueError("Missing or duplicate designated normal image")
                runtime = runtime_factory(weights)
                monitor.runtime = runtime
                monitor.check()
                train_rows = groups["training"]["rows"]

                def reference():
                    values = []
                    for row in train_rows[:12]:
                        monitor.check()
                        with fit.checked_image(z, row) as img:
                            values.append(runtime.predict(img))
                    return np.stack(values)

                def save(step):
                    monitor.check()
                    if hasattr(p, "check_disk"):
                        p.check_disk(root, step)
                    with fit.phase_stage(root, f"step-{step:06d}") as stage:
                        maps = reference()
                        runtime.save(stage / "model.pt", signature, step)
                        np.save(stage / "reference.npy", maps, allow_pickle=False)
                        fit.write_json(stage / "progress.json", {"step": step, "statistics": statistics, "losses": losses,
                            "monitor_raw_map_means": maps[:, 1:].mean(axis=1, dtype=np.float64).tolist(),
                            "fingerprints": runtime.fingerprints()})
                        fit.seal_phase(stage, f"step-{step:06d}", signature, {}, CHECKPOINT_FILES)
                    monitor.check()

                def reload_check(step):
                    runtime.load(root / f"step-{step:06d}/model.pt", signature, step)
                    expected = np.load(root / f"step-{step:06d}/reference.npy", allow_pickle=False)
                    actual = reference()
                    if actual.shape != expected.shape or not np.isfinite(expected).all():
                        raise ValueError("Checkpoint reference layout invalid")
                    p.validate_predictions(actual)
                    p.validate_predictions(expected)
                    difference = float(np.abs(actual - expected).max())
                    progress = json.loads((root / f"step-{step:06d}/progress.json").read_bytes())
                    if runtime.fingerprints() != progress["fingerprints"]:
                        raise ValueError("Checkpoint tensor fingerprints changed")
                    if difference > policy["reload_tolerance"]:
                        raise ValueError(f"Checkpoint raw map drift: {difference}")
                    reloads.append({"step": step, "maximum_raw_map_difference": float(np.abs(actual[:, 1:]-expected[:, 1:]).max()),
                                    "maximum_native_score_difference": float(np.abs(actual[:, 0]-expected[:, 0]).max()),
                                    "optimizer_and_rng_restored": True})
                    monitor.check()

                if available:
                    state = json.loads((root / f"step-{last:06d}/progress.json").read_bytes())
                    statistics, losses = state["statistics"], state["losses"]
                    reload_check(last)
                else:
                    print(f"1/4 Fresh initialization: {len(groups['training']['photo_ids'])} normal photos, "
                          f"{len(train_rows)} tiles. No probe checkpoint reuse.", flush=True)
                    statistics = {"initial_fingerprints": runtime.fingerprints(), "fresh_initialization": True}
                    if statistics["initial_fingerprints"] != identity["expected_fresh_fingerprints"]:
                        raise ValueError("Fresh model initialization differs from reviewed seed42 initialization")
                    losses = []
                    save(0)
                print(f"2/4 Training from {last}/{total}; final checkpoint only, no test-based early stopping.", flush=True)
                for index in range(last, total):
                    monitor.check()
                    if prior_updates + updates >= policy["maximum_attempted_updates"]:
                        raise RuntimeError("Cumulative attempted-update limit reached; no automatic extension")
                    row_indices = p.batch_indices(index)
                    with ExitStack() as stack:
                        images = [stack.enter_context(fit.checked_image(z, train_rows[i])) for i in row_indices]
                        updates += 1
                        value = runtime.step(images, index)
                    value["training_row_indices"] = list(row_indices)
                    losses.append(value)
                    monitor.check()
                    if (index+1) % per_epoch == 0 and (index+1) % interval != 0:
                        print(f"epoch {(index+1)//per_epoch}/{policy.get('epochs', 300)}; train {index+1}/{total}; elapsed={prior+time.perf_counter()-started:.1f}s", flush=True)
                    if (index + 1) % interval == 0 or index+1 == total:
                        save(index + 1)
                        last = index + 1
                        recent = losses[-interval:]
                        mean = float(np.mean([v["loss"] for v in recent]))
                        print(f"train {last}/{total}; recent mean loss={mean:.5g}; elapsed={prior+time.perf_counter()-started:.1f}s; saved", flush=True)
                        if last % (50 * per_epoch) == 0 or last == total:
                            reload_check(last)
                final_fingerprints = runtime.fingerprints()
                initial_fingerprints = statistics["initial_fingerprints"]
                if final_fingerprints["backbone"] != initial_fingerprints["backbone"]:
                    raise ValueError("Frozen backbone changed during formal training")
                if any(final_fingerprints[k] == initial_fingerprints[k] for k in ("adaptor", "segmentation", "classification")):
                    raise ValueError("Trainable model component did not change")
                synthetic_count = sum(v["synthetic_anomalous_samples"] for v in losses)
                if not 0 < synthetic_count < total * 16:
                    raise ValueError("Formal fit did not exercise both normal and synthetic classes")
                print("3/4 Saving native logits and raw maps: development128 + calibration169. No test access.", flush=True)
                for g, count in counts.items():
                    for i in range(count):
                        name = f"photo-{i:03d}"
                        if (root / g / name).exists():
                            continue
                        monitor.check()
                        maps = []
                        for row in groups[g]["rows"][i*12:(i+1)*12]:
                            monitor.check()
                            with fit.checked_image(z, row) as img:
                                maps.append(runtime.predict(img))
                        with fit.phase_stage(root / g, name) as stage:
                            np.save(stage / "predictions.npy", p.validate_predictions(np.stack(maps)), allow_pickle=False)
                            fit.seal_phase(stage, name, signature, {"photo_id": groups[g]["photo_ids"][i],
                                "rows_sha256": fit.digest(groups[g]["rows"][i*12:(i+1)*12])}, ["predictions.npy"])
                        monitor.check()
                        if (i+1) % 16 == 0 or i+1 == count:
                            print(f"{g} raw maps {i+1}/{count}", flush=True)
                if not (root / "calibrated").exists():
                    maps = {g: normal_arrays(root, g, count) for g, count in counts.items()}
                    monitor.check()
                    scores = {}
                    for g in counts:
                        scores[g] = []
                        for a in maps[g]:
                            monitor.check()
                            scores[g].append(p.tile_scores(a).tolist())
                    contract = p.calibrate(scores["development"], scores["calibration"])
                    publish(root, "calibrated", signature, {"contract": contract, "tile_scores": scores,
                        "photo_ids": {g: groups[g]["photo_ids"] for g in counts}})
                    del maps
                normal = json.loads((root / "calibrated/value.json").read_bytes())
                contract = normal["contract"]
                print("4/4 Reloading final checkpoint and verifying all 297 normals against the frozen contract.", flush=True)
                if not (root / "reload").exists():
                    reload_check(total)
                    maximum_map, maximum_native, maximum_score, changes = 0., 0., 0., 0
                    for g, count in counts.items():
                        for i in range(count):
                            monitor.check()
                            cached = np.load(root / g / f"photo-{i:03d}/predictions.npy", allow_pickle=False)
                            maps = []
                            for j, row in enumerate(groups[g]["rows"][i*12:(i+1)*12]):
                                monitor.check()
                                with fit.checked_image(z, row) as img:
                                    actual = runtime.predict(img)
                                p.validate_predictions(actual[None, :])
                                p.validate_predictions(cached[j:j+1])
                                maximum_map = max(maximum_map, float(np.abs(actual[1:]-cached[j, 1:]).max()))
                                maximum_native = max(maximum_native, float(abs(actual[0]-cached[j, 0])))
                                maps.append(actual)
                            now = p.tile_scores(np.stack(maps))
                            old = np.array(normal["tile_scores"][g][i])
                            a, b = p.photo_scores(now, contract), p.photo_scores(old, contract)
                            maximum_score = max(maximum_score, float(np.abs(now-old).max()), float(abs(a-b)))
                            changes += int((a >= contract["threshold"]) != (b >= contract["threshold"]))
                            if max(maximum_map, maximum_native, maximum_score) > policy["reload_tolerance"] or changes:
                                raise ValueError("Final normal reload drift; no recalibration")
                            if (i+1) % 32 == 0 or i+1 == count:
                                print(f"reload {g} {i+1}/{count}", flush=True)
                    publish(root, "reload", signature, {"photos": sum(counts.values()), "passed": True,
                        "maximum_raw_map_difference": maximum_map, "maximum_native_score_difference": maximum_native,
                        "maximum_standardized_score_difference": maximum_score,
                        "decision_changes": changes, "contract_refitted": False})
                monitor.check()
            status = "completed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            torch = getattr(runtime, "torch", None) if runtime is not None else sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                try:
                    monitor.allocated = max(monitor.allocated, torch.cuda.max_memory_allocated())
                    monitor.reserved = max(monitor.reserved, torch.cuda.max_memory_reserved())
                except Exception:
                    pass
            fit.write_json(attempt / "finished.json", {"status": status, "error": error,
                "seconds": time.perf_counter()-started, "attempted_updates": updates, "latest_saved_step": last,
                "memory": monitor.snapshot(), "reload_checks": reloads})
        records = attempts(root)
        memory = dict(records[-1]["memory"])
        for key in ("rss_sample_peak_bytes", "device_used_sample_peak_bytes",
                    "torch_cuda_peak_allocated_bytes", "torch_cuda_peak_reserved_bytes"):
            if key in memory:
                memory[key] = max(a["memory"][key] for a in records)
        report = {"status": "supersimplenet_fit_normal_calibrated_ready_for_review", "model": "SuperSimpleNet",
            "completed_epochs": total // per_epoch, "final_fingerprints": final_fingerprints,
            "initial_fingerprints": initial_fingerprints, "synthetic_anomalous_presentations": synthetic_count,
            "identity_sha256": signature, "completed_updates": total,
            "cumulative_attempted_updates": sum(a["attempted_updates"] for a in records),
            "cumulative_compute_seconds": sum(a["seconds"] for a in records),
            "memory": memory,
            "training_photos": len(groups["training"]["photo_ids"]), "normal_cohorts": counts,
            "environment": identity["environment"], "final_checkpoint": f"step-{total:06d}/model.pt",
            "checkpoint_outputs_native_logits_and_raw_maps": True, "normal_contract_path": "calibrated/value.json",
            "threshold": normal["contract"]["threshold"], "normal_reload_passed": True,
            "accuracy_measured": False, "test_images_decoded": 0, "test_masks_decoded": 0,
            "automatic_pass_allowed": False, "service_model_changed": False}
        publish(root, "result", signature, report)
        export(root, handoff, identity, report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    identity = p.prepare(project, project / "artifacts/handoff")
    if environment() != identity["environment"]:
        raise ValueError("Environment differs from reviewed GPU probe")
    if fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original MPI archive hash differs")
    if shutil.disk_usage(project).free < p.POLICY["free_disk_GiB"] * 1024**3:
        raise ValueError("Need 20 GiB free disk; no automatic checkpoint deletion")
    asset_root = project / "artifacts/research-assets/supersimplenet-v1"
    with fit.locked_run(asset_root):
        weights = download(asset_root, WEIGHTS, False)
    report = execute(project / "artifacts" / p.NAME, args.archive, identity, weights,
                     project / "artifacts/handoff" / (p.NAME + ".zip"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all files; do not run test.", file=sys.stderr)
        raise SystemExit(1)
