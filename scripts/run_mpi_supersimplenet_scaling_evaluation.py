"""Evaluate all frozen scaling arms, including failed regression cases; no retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from scripts import mpi_supersimplenet_scaling_evaluation_protocol as q
from scripts import run_mpi_supersimplenet_evaluation as engine
from scripts import mpi_supersimplenet_map_diagnostic as maps
from scripts import mpi_supersimplenet_map_cards as cards
from scripts.mpi_supersimplenet_scaling_evaluation_runtime import Runtime, Monitor, environment
from scripts.run_mpi_efficientad_probe import publish, attempts

fit, e, p, ev = q.fit, q.old, q.p, engine.ev
ARMS = tuple(q.scaling.ARMS)


def derived_identity(identity, arm):
    return {**identity, "arm": arm, **identity["arms"][arm]}


def collect(root, signature, archive, photos, runtime, monitor, contract, gate_path,
            recovery_path, previous, cached):
    """Persist every photo. No branch depends on accuracy or a screen result."""
    recovered = None
    for group in ("test", "validation"):
        for i, photo in enumerate(photos[group]):
            monitor.check()
            if (group, i) not in cached:
                recovered = engine.recovery_for_photo(photo, recovered, recovery_path, previous)
                raw, sample = engine.score_photo(archive, photo, runtime, recovered, monitor, contract)
                name = f"photo-{i:03d}"
                with fit.phase_stage(root/group, name) as stage:
                    np.save(stage/"predictions.npy", raw, allow_pickle=False)
                    fit.write_json(stage/"sample.json", sample)
                    fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photo),
                        "gate_path": gate_path, "gate_sha256": fit.sha256_file(root/gate_path)},
                        ["predictions.npy", "sample.json"])
                cached[group, i] = sample
            if (i+1) % 10 == 0 or i+1 == len(photos[group]):
                print(f"{root.name} {group} {i+1}/{len(photos[group])}", flush=True)
    return {g: [cached[g, i] for i in range(len(photos[g]))] for g in ("test", "validation")}


def map_entries(source, samples):
    """All defects numerically; fixed12 and high normals get image cards."""
    entries = [{"group": "test", "index": i, "role": "all_defects", "photo": photo,
                "card": photo["key"] in e.POLICY["acceptance"]["regression_keys"]}
               for i, photo in enumerate(source["cohorts"]["test"]) if photo["label"]]
    for group in ("test", "validation"):
        normal = [i for i, s in enumerate(samples[group]) if s["label"] == 0]
        selected = sorted(normal, key=lambda i: (-samples[group][i]["score"], samples[group][i]["key"]))[:3]
        entries += [{"group": group, "index": i, "role": "highest_normal", "card": True,
                     "photo": source["cohorts"][group][i]} for i in selected]
    return entries


def review_maps(project, root, signature, archive, source, samples, contract, recovery_path, monitor):
    from scripts.mpi_score_probe_protocol import EVALUATION_SHA256
    baseline = ev.checked_bundle(project/"artifacts/handoff/mpi-pcb-normal-bank-evaluation-v1.zip", EVALUATION_SHA256)
    if fit.digest(baseline["input-lock.json"]["identity"]) != baseline["evaluation.json"]["execution_sha256"]:
        raise ValueError("Baseline map provenance differs")
    source = {**source, "baseline_bundle": baseline}
    baseline_root = project/"artifacts/evaluations/mpi-pcb-normal-bank-evaluation-v1"
    (root/"maps").mkdir(exist_ok=True)
    cases = []
    entries = map_entries(source, samples)
    for n, entry in enumerate(entries):
        monitor.check()
        group, i, photo = entry["group"], entry["index"], entry["photo"]
        sample = engine.read_cache(root, group, i, signature, photo, contract)
        raw = np.load(root/group/f"photo-{i:03d}/predictions.npy", allow_pickle=False)
        entry["baseline"] = maps.baseline_record(entry, source, baseline, baseline_root)
        prior = maps.load_baseline(entry, source, baseline_root)
        masks = ev.read_masks(archive, photo) if group == "test" else [np.zeros((256, 256), dtype=np.bool_) for _ in range(12)]
        # Normal test masks are all-zero; do not present them as localization metrics.
        describe_entry = entry if photo["label"] else {**entry, "group": "normal"}
        case, arrays = maps.describe(describe_entry, raw, masks, sample, contract, prior)
        case["group"] = group
        name = f"case-{n:03d}"
        files = ["case.json"] + (["card-1.png", "card-2.png"] if entry["card"] else [])
        metadata = {"entry_sha256": fit.digest(entry), "sample_sha256": fit.digest(sample)}
        if (root/"maps"/name).exists():
            receipt = fit.read_phase(root/"maps", name, signature, files)
            if any(receipt[k] != v for k, v in metadata.items()) or json.loads((root/"maps"/name/"case.json").read_bytes()) != case:
                raise ValueError("Map diagnostic cache changed")
        else:
            with fit.phase_stage(root/"maps", name) as stage:
                fit.write_json(stage/"case.json", case)
                if entry["card"]:
                    recovered = engine.recovery_for_photo(photo, None, recovery_path, source["previous"])
                    images = []
                    try:
                        for row in photo["regions"]:
                            images.append(ev.read_tile(archive, row, recovered) if group == "test" else fit.checked_image(archive, row))
                        for page in range(2):
                            display_case = {**case, "role": root.name+" | "+case["role"]}
                            if not photo["label"]:
                                display_case["group"] = "normal"
                            with cards.render(display_case, images, arrays, masks, page) as card:
                                card.save(stage/f"card-{page+1}.png")
                    finally:
                        for image in images:
                            image.close()
                fit.seal_phase(stage, name, signature, metadata, files)
        cases.append(case)
        print(f"{root.name} saved-map review {n+1}/{len(entries)}", flush=True)
    return {"cases": cases, "defect_photos": sum(x["photo"]["label"] == 1 for x in entries),
        "normal_photos": sum(x["photo"]["label"] == 0 for x in entries),
        "location_excluded_photos": ["defect/0006"], "cards": 2*sum(x["card"] for x in entries),
        "baseline_map_available_photos": sum(c["baseline_available"] for c in cases),
        "interpretation": "Within-tile GT ranking and classifier margins, not defect probabilities or new pixel thresholds",
        "card_selection": "fixed prior12 and highest3 normal photo scores in each observed normal group"}


def execute_arm(project, root, archive_path, fit_root, recovery, records, source, identity,
                runtime_factory=Runtime, monitor_factory=Monitor):
    arm, signature, contract = identity["arm"], fit.digest(identity), identity["contract"]
    checkpoint = fit_root/f"step-{records['plan.json']['policy']['updates']:06d}/model.pt"
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
            raise ValueError("Output exists without identity; preserved")
        publish(root, "identity", signature, identity)
        previous = attempts(root)
        cached = {}
        for g in ("test", "validation"):
            (root/g).mkdir(exist_ok=True)
            allowed = {f"photo-{i:03d}" for i in range(len(source["cohorts"][g]))}
            if any(f.name not in allowed for f in (root/g).glob("photo-*")):
                raise ValueError("Unexpected photo cache")
            for i, photo in enumerate(source["cohorts"][g]):
                if (root/g/f"photo-{i:03d}").exists():
                    cached[g, i] = engine.read_cache(root, g, i, signature, photo, contract)
        if (root/"result").exists():
            if len(cached) != 365:
                raise ValueError("Completed comparison lacks full cohort")
            # Verify all downstream receipts even on completed reruns.
            for phase in ("screen", "comparison", "warm-candidate", "stability-summary", "map-summary"):
                e.value(root, phase, signature)
            return e.value(root, "result", signature)
        started = time.perf_counter()
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature, "unix_seconds": time.time()})
        monitor = monitor_factory(None, sum(a["seconds"] for a in previous), started, q.POLICY)
        runtime, status, error = None, "failed", None
        try:
            with monitor, zipfile.ZipFile(archive_path) as archive:
                runtime = runtime_factory(checkpoint, identity["fit_signature"], arm)
                monitor.runtime = runtime
                gate = engine.normal_gate(archive, runtime, monitor, records, source, fit_root)
                fit.write_json(attempt/"normal-gate.json", gate)
                gate_path = (attempt/"normal-gate.json").relative_to(root).as_posix()
                print(f"{arm}: 1/4 all55 defects + normal54/256. Prior12 is descriptive, never a stop gate.", flush=True)
                samples = collect(root, signature, archive, source["cohorts"], runtime, monitor,
                                  contract, gate_path, recovery, source["previous"], cached)
                by_key = {s["key"]: s for s in samples["test"]}
                screen = e.screen([by_key[k] for k in e.POLICY["acceptance"]["regression_keys"]])
                publish(root, "screen", signature, screen)
                comparison = e.compare(samples, source)
                publish(root, "comparison", signature, comparison)
                print(f"{arm}: 2/4 fixed normal warm benchmark, no elapsed-time cap.", flush=True)
                if not (root/"warm-candidate").exists():
                    warm = engine.benchmark("candidate", runtime, archive, source, records, contract, monitor)
                    # Retain measured latency, remove the obsolete 2-second selection gate.
                    warm.pop("candidate_budget_passed")
                    warm["elapsed_time_cap"] = None
                    publish(root, "warm-candidate", signature, warm)
                monitor.check()
                runtime.release()
                monitor.runtime = runtime = None
                print(f"{arm}: 3/4 saved map / GT attribution; scores and thresholds unchanged.", flush=True)
                diagnostic = review_maps(project, root, signature, archive, source, samples, contract, recovery, monitor)
                publish(root, "map-summary", signature, diagnostic)
                print(f"{arm}: 4/4 all600 paired normal-calibration draws regardless of performance.", flush=True)
                stability = engine.stability(root, signature, fit_root, source, records, monitor)
                stability["descriptive_summaries"] = {k: q.summarize(v) for k, v in stability["trials"].items()}
                publish(root, "stability-summary", signature, stability)
                monitor.check()
                if fit.sha256_file(checkpoint) != identity["checkpoint_sha256"]:
                    raise ValueError("Final checkpoint changed")
                status = "completed"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            fit.write_json(attempt/"error-detail.json", {"error": error, "traceback": traceback.format_exc()})
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
                fit.write_json(attempt/"finished.json", {"status": status, "error": error,
                    "seconds": time.perf_counter()-started, "memory": monitor.snapshot(),
                    "cached_photos": len(cached), "elapsed_time_cap": None})
        report = {"status": "full_scaling_arm_evaluation_completed", "arm": arm,
            "counts": comparison["counts"], "prior12_retained": screen["retained"],
            "full_comparison_run": True, "warm_evaluated": True, "stability_evaluated": True,
            "map_diagnostics_evaluated": True, "cached_photos": len(cached),
            "checkpoint_sha256": identity["checkpoint_sha256"], "contract": contract,
            "cumulative_seconds": sum(a["seconds"] for a in attempts(root)),
            "training_performed": False, "threshold_changed": False,
            "automatic_pass_allowed": False, "service_model_changed": False,
            "independent_performance_claim_allowed": False}
        publish(root, "result", signature, report)
        return report


def baseline_warm(root, archive_path, fit_root, source, identity):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        publish(root, "identity", signature, identity)
        prior = attempts(root)
        if (root/"result").exists():
            return e.value(root, "result", signature)
        started = time.perf_counter()
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        fit.write_json(attempt/"started.json", {"identity": signature})
        monitor = Monitor(None, sum(a["seconds"] for a in prior), started, q.POLICY)
        runtime, status, error = None, "failed", None
        try:
            with monitor, zipfile.ZipFile(archive_path) as archive:
                checkpoint = fit_root.parent/"mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
                if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256:
                    raise ValueError("Q90 checkpoint changed")
                runtime = engine.EvaluationRuntime(source["config"])
                monitor.runtime = runtime
                runtime.load(checkpoint, source["records"]["fit.json"]["checkpoint_context"])
                result = engine.benchmark("baseline", runtime, archive, source, None, None, monitor)
                result.pop("candidate_budget_passed")
                result["elapsed_time_cap"] = None
                monitor.check()
                runtime.release()
                monitor.runtime = runtime = None
                status = "completed"
        except KeyboardInterrupt:
            status, error = "interrupted", "KeyboardInterrupt"
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
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
                fit.write_json(attempt/"finished.json", {"status": status, "error": error,
                    "seconds": time.perf_counter()-started, "memory": monitor.snapshot()})
        publish(root, "result", signature, result)
        return result


def export(root, handoff, identity):
    """JSON evidence and selected cards only; models and raw maps stay on Windows."""
    payload = {"plan.json": fit.canonical(identity)}
    reports = {}
    for arm in ARMS:
        signature = fit.digest(derived_identity(identity, arm))
        directory = root/arm
        reports[arm] = e.value(directory, "result", signature)
        for name in ("result", "screen", "comparison", "warm-candidate", "stability-summary", "map-summary"):
            payload[arm+"/"+name+".json"] = fit.canonical(e.value(directory, name, signature))
        payload[arm+"/attempts.json"] = fit.canonical(attempts(directory))
        samples, receipts = {}, {}
        for group in ("test", "validation"):
            samples[group] = []
            for i, photo in enumerate(identity["cohorts"][group]):
                samples[group].append(engine.read_cache(directory, group, i, signature, photo, identity["arms"][arm]["contract"]))
                path = directory/group/f"photo-{i:03d}/receipt.json"
                receipts[path.relative_to(directory).as_posix()] = json.loads(path.read_bytes())
        payload[arm+"/samples.json"] = fit.canonical(samples)
        payload[arm+"/sample-receipts.json"] = fit.canonical(receipts)
        payload[arm+"/runtime-gates.json"] = fit.canonical({f.relative_to(directory).as_posix(): json.loads(f.read_bytes())
            for f in directory.glob("attempt-*/normal-gate.json")})
        for folder in sorted((directory/"maps").glob("case-*")):
            receipt = json.loads((folder/"receipt.json").read_bytes())
            fit.read_phase(directory/"maps", folder.name, signature, list(receipt["files"]))
            for name in [*receipt["files"], "receipt.json"]:
                payload[arm+"/maps/"+folder.name+"/"+name] = (folder/name).read_bytes()
    payload["warm-q90.json"] = fit.canonical(e.value(root/"q90", "result", fit.digest(identity)))
    payload["report.json"] = fit.canonical({"status": "all_three_scaling_comparisons_completed",
        "arms": reports, "baseline_counts": {"tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14},
        "exploratory_only": True, "automatic_pass_allowed": False, "service_model_changed": False})
    payload["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k, v in payload.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(payload) or set(z.namelist()) != set(payload) or any(z.read(k) != v for k, v in payload.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".scaling-evaluation-export-", dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(stage, "x", zipfile.ZIP_DEFLATED) as z:
        for name, raw in payload.items():
            z.writestr(name, raw)
    stage.rename(handoff)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--worker", choices=(*ARMS, "q90"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    project, archive_path = args.project.resolve(), args.archive.resolve()
    fit_root = project/"artifacts"/q.scaling.NAME
    root = project/"artifacts/analyses"/q.NAME
    data = q.read_bundle(project/"artifacts/handoff"/(q.scaling.NAME+".zip"))
    source, identity = q.prepare(project, data)
    if fit.sha256_file(archive_path) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original MPI archive changed")
    if environment() != identity["environment"]:
        raise ValueError("Environment differs from frozen fit")
    if fit.sha256_file(project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt") != ev.CHECKPOINT_SHA256:
        raise ValueError("Q90 checkpoint missing or changed; required for paired warm timing")
    if shutil.disk_usage(project).free < q.POLICY["free_disk_GiB"]*1024**3:
        raise ValueError("Need 20 GiB free disk; no automatic deletion")
    if args.worker:
        with fit.locked_run(fit_root):
            if args.worker == "q90":
                baseline_warm(root/"q90", archive_path, fit_root, source, identity)
            else:
                arm = args.worker
                records = q.read_arm(project, fit_root, data, arm)
                execute_arm(project, root/arm, archive_path, fit_root/arm,
                    project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
                    records, source, derived_identity(identity, arm))
        return 0
    # Coordinator lock prevents two launchers; workers take separate arm and fit locks.
    with fit.locked_run(root):
        publish(root, "identity", fit.digest(identity), identity)
        for arm in (*ARMS, "q90"):
            print(f"Launching isolated {arm} evaluation process; no elapsed-time cap.", flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--project", str(project),
                                     "--archive", str(archive_path), "--worker", arm], check=False)
            if result.returncode:
                raise RuntimeError(f"{arm} evaluation stopped; keep files and send error")
        export(root, project/"artifacts/handoff"/(q.NAME+".zip"), identity)
    print(json.dumps({"status": "all_three_scaling_comparisons_completed", "handoff": str(project/"artifacts/handoff"/(q.NAME+".zip"))}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve files; no retraining, reroll or threshold changes.", file=sys.stderr)
        raise SystemExit(1)
