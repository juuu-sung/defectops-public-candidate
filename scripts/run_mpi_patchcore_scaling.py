"""Resumable normal256 fit/calibration/reload and complete observed Q90 comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from scripts import mpi_patchcore_scaling_protocol as p
from scripts.mpi_patchcore_scaling_runtime import Runtime, environment
from scripts.mpi_supersimplenet_scaling_evaluation_runtime import Monitor
from scripts import run_mpi_supersimplenet_evaluation as common

fit, ev = p.fit, p.evaluation.inputs.ev


@contextmanager
def attempt(root, signature, policy=p.POLICY):
    previous = p.attempts(root)
    directory = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    fit.write_json(directory/"started.json", {"identity": signature, "unix_seconds": time.time()})
    started, status, error = time.perf_counter(), "failed", None
    monitor = Monitor(None, sum(a["seconds"] for a in previous), started, policy)
    try:
        with monitor:
            yield monitor
            monitor.check()
        status = "completed"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        fit.write_json(directory/"error-detail.json", {"error": error, "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            if monitor.runtime is not None:
                monitor.runtime.release()
        except BaseException as exc:
            status, error = "failed", f"Cleanup failed: {type(exc).__name__}: {exc}; prior error: {error}"
            raise
        finally:
            fit.write_json(directory/"finished.json", {"identity": signature, "status": status, "error": error,
                "seconds": time.perf_counter()-started, "memory": monitor.snapshot(), "elapsed_time_cap": None})


def initialize(root, identity):
    signature = fit.digest(identity)
    if not (root/"identity").exists() and any(x.name != ".run.lock" for x in root.iterdir()):
        raise ValueError("Output without identity; preserved for review")
    p.publish(root, "identity", signature, identity)
    p.attempts(root)
    return signature


def extract(root, signature, archive, training, runtime, baseline, monitor):
    folder = root/"features"
    folder.mkdir(exist_ok=True)
    paths, receipts = [], []
    runtime.offline_model(baseline)
    expected = {f"photo-{i:03d}" for i in range(256)}
    if any(x.name not in expected for x in folder.glob("photo-*")):
        raise ValueError("Unexpected feature-cache photo")
    for i, pid in enumerate(training["photo_ids"]):
        rows, name = training["rows"][i*12:(i+1)*12], f"photo-{i:03d}"
        monitor.check()
        if not (folder/name).exists():
            values = runtime.extract(archive, rows, monitor.check)
            with fit.phase_stage(folder, name) as stage:
                np.save(stage/"features.npy", values, allow_pickle=False)
                fit.seal_phase(stage, name, signature, {"photo_id": pid, "rows_sha256": fit.digest(rows)}, ["features.npy"])
            del values
        receipt = fit.read_phase(folder, name, signature, ["features.npy"])
        if receipt["photo_id"] != pid or receipt["rows_sha256"] != fit.digest(rows):
            raise ValueError("Feature photo identity differs")
        path = folder/name/"features.npy"
        p.array(path, (12288, 768))
        paths.append(path); receipts.append(receipt)
        if (i+1) % 8 == 0:
            print(f"normal feature cache {i+1}/256 original photos ({(i+1)*12}/3072 tiles)", flush=True)
    monitor.check()
    runtime.release()
    p.publish(root, "feature-receipts", signature, receipts)
    return paths


def score_photo(archive, photo, runtime, recovered=None):
    runtime.synchronize()
    began = time.perf_counter()
    rows, maps = [], []
    for row in photo["regions"]:
        read = ev.read_tile(archive, row, recovered) if photo["key"].startswith(("good/", "defect/")) else fit.checked_image(archive, row)
        with read as image:
            native, values = runtime.predict(image)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0) or not np.isfinite(native):
            raise ValueError("Invalid PatchCore anomaly map/score")
        rows.append({k: row[k] for k in ("region", "member")} | {
            "top1": float(fit.candidate_image_score(native, values, fit.CANDIDATE)), "native_raw": float(native)})
        maps.append(values)
    runtime.synchronize()
    sample = {k: photo[k] for k in ("key", "photo_id", "label")} | {
        "regions": rows, "photo_seconds": time.perf_counter()-began}
    p.comparison.c.scoring.old.validate_sample(sample, photo)
    return sample, np.stack(maps)


def read_sample(folder, name, signature, photo, checkpoint_sha):
    receipt = fit.read_phase(folder, name, signature, ["sample.json", "maps.npy"])
    if receipt["photo_sha256"] != fit.digest(photo) or receipt["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Cached prediction provenance differs")
    sample = json.loads((folder/name/"sample.json").read_bytes())
    p.comparison.c.scoring.old.validate_sample(sample, photo)
    maps = p.array(folder/name/"maps.npy", (12, 256, 256))
    for row, values in zip(sample["regions"], maps, strict=True):
        if np.any(values < 0) or row["top1"] != float(fit.candidate_image_score(row["native_raw"], values, fit.CANDIDATE)):
            raise ValueError("Saved map does not reproduce top1 score")
    return sample


def collect(root, phase, signature, archive, photos, runtime, monitor, checkpoint_sha,
            recovery=None, previous=None):
    folder = root/phase
    folder.mkdir(exist_ok=True)
    names = {f"photo-{i:03d}" for i in range(len(photos))}
    if any(x.name not in names for x in folder.glob("photo-*")):
        raise ValueError("Unexpected score cache photo")
    result, recovered = [], None
    for i, photo in enumerate(photos):
        monitor.check()
        name = f"photo-{i:03d}"
        if not (folder/name).exists():
            if recovery is not None:
                recovered = common.recovery_for_photo(photo, recovered, recovery, previous)
            sample, maps = score_photo(archive, photo, runtime, recovered)
            with fit.phase_stage(folder, name) as stage:
                fit.write_json(stage/"sample.json", sample)
                np.save(stage/"maps.npy", maps, allow_pickle=False)
                fit.seal_phase(stage, name, signature, {"photo_sha256": fit.digest(photo),
                    "checkpoint_sha256": checkpoint_sha}, ["sample.json", "maps.npy"])
        result.append(read_sample(folder, name, signature, photo, checkpoint_sha))
        if (i+1) % 16 == 0 or i+1 == len(photos):
            print(f"{phase} {i+1}/{len(photos)}", flush=True)
    return result


def read_normals(root, signature, cohorts, checkpoint_sha, prefix="first"):
    return {g: [read_sample(root/(prefix+"-"+g), f"photo-{i:03d}", signature, photo, checkpoint_sha)
                for i, photo in enumerate(cohorts[g])] for g in ("development", "calibration")}


def normal_gate(root, signature, archive, source, runtime, checkpoint_sha, method, count=8):
    """Every inference process checks fixed calibration images before any observed test."""
    first, replay, map_difference = [], [], 0.
    for i, photo in enumerate(source["cohorts"]["calibration"][:count]):
        name = f"photo-{i:03d}"
        expected = read_sample(root/"first-calibration", name, signature, photo, checkpoint_sha)
        actual, values = score_photo(archive, photo, runtime)
        stored = p.array(root/"first-calibration"/name/"maps.npy", (12, 256, 256))
        map_difference = max(map_difference, float(np.max(np.abs(stored-values))))
        first.append(expected); replay.append(actual)
    check = p.reload_check({"development": [], "calibration": first}, {"development": [], "calibration": replay}, method)
    if map_difference > p.POLICY["reload_absolute_tolerance"]:
        raise ValueError("Pre-evaluation normal map replay failed")
    return {**check, "map_maximum_absolute_difference": map_difference}


def fit_run(project, root, archive_path, source, identity, runtime_factory=Runtime):
    with fit.locked_run(root):
        signature = initialize(root, identity)
        model_files = ["model.pt", "context.json"]
        if (root/"result").exists():
            model = fit.read_phase(root, "model", signature, model_files)
            first = read_normals(root, signature, source["cohorts"], model["files"]["model.pt"])
            second = read_normals(root, signature, source["cohorts"], model["files"]["model.pt"], "reload")
            contract = p.value(root, "contract", signature)
            if p.reload_check(first, second, contract) != p.value(root, "reload-check", signature):
                raise ValueError("Completed reload evidence differs")
            p.comparison.contract_replay(first, contract)
            return p.value(root, "result", signature)
        with attempt(root, signature) as monitor, zipfile.ZipFile(archive_path) as archive:
            runtime = runtime_factory(source["config"]); monitor.runtime = runtime
            baseline = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
            if not (root/"model").exists():
                # Catch preprocessing/numerics drift before spending time on the large coreset.
                runtime.load(baseline, source["records"]["fit.json"]["checkpoint_context"])
                expected, actual = [], []
                for i, photo in enumerate(source["cohorts"]["calibration"][:8]):
                    monitor.check()
                    sample, _ = score_photo(archive, photo, runtime)
                    expected.append(source["reference"]["calibration"][i]); actual.append(sample)
                gate = p.reload_check({"development": [], "calibration": expected},
                    {"development": [], "calibration": actual}, source["contract"]["candidate_top1_q90"])
                if not (root/"baseline-gate").exists():
                    p.publish(root, "baseline-gate", signature, gate)
                else:
                    p.value(root, "baseline-gate", signature)
                runtime.release()
                print("1/4 Extract all 256 normal photos to disk. No test images.", flush=True)
                paths = extract(root, signature, archive, identity["training"], runtime, baseline, monitor)
                print("2/4 Project all 3,145,728 candidates and select 78,643; no time cap.", flush=True)
                projected, start, receipts = runtime.project(root/"projection", signature, paths, monitor.check)
                p.publish(root, "projection-receipts", signature, receipts)
                selected, receipts = runtime.select(root/"selection", signature, projected, start, monitor.check)
                p.publish(root, "selection-receipts", signature, receipts)
                p.publish(root, "selected-indices", signature, {"start_index": start, "count": len(selected),
                    "indices": selected, "ordering": "photo order, 12 region order, stride2 raster patch order"})
                runtime.assemble(baseline, paths, selected, monitor.check)
                context = {"category": p.NAME, "seed": 42, "manifest_sha256": fit.digest(identity["training"]),
                           "execution_config_sha256": signature}
                with fit.phase_stage(root, "model") as stage:
                    runtime.save(stage/"model.pt", context)
                    fit.write_json(stage/"context.json", context)
                    fit.seal_phase(stage, "model", signature, {"representatives": len(selected)}, model_files)
                monitor.check()
            receipt = fit.read_phase(root, "model", signature, model_files)
            checkpoint_sha = receipt["files"]["model.pt"]
            context = json.loads((root/"model/context.json").read_bytes())
            runtime.load(root/"model/model.pt", context)
            print("3/4 Separate development128 and calibration169 normals; freeze Q90 contract.", flush=True)
            first = {g: collect(root, "first-"+g, signature, archive, source["cohorts"][g], runtime,
                        monitor, checkpoint_sha) for g in ("development", "calibration")}
            arrays = p.comparison.matrix(first)
            contract = p.comparison.fit_method(arrays["development"], arrays["calibration"])
            p.publish(root, "contract", signature, contract)
            runtime.load(root/"model/model.pt", context)
            print("4/4 Reload saved model and replay all 297 normal photos; no test yet.", flush=True)
            second = {g: collect(root, "reload-"+g, signature, archive, source["cohorts"][g], runtime,
                         monitor, checkpoint_sha) for g in ("development", "calibration")}
            reload = p.reload_check(first, second, contract)
            p.publish(root, "reload-check", signature, reload)
            monitor.check()
        report = {"status": "fit_calibrated_reloaded", "training_photos": 256, "training_tiles": 3072,
            "candidate_embeddings": 3145728, "memory_bank_shape": [78643, 768],
            "checkpoint_sha256": checkpoint_sha, "context": context, "contract": contract,
            "reload": reload, "test_images_decoded": 0, "automatic_pass_allowed": False,
            "service_model_changed": False, "cumulative_seconds": sum(a["seconds"] for a in p.attempts(root))}
        p.publish(root, "result", signature, report)
        return report


def export(root, output, identity, report, eval_root=None):
    """JSON only: models, raw training features and copyrighted imagery stay local."""
    signature = fit.digest(identity)
    payload = {"plan.json": identity, "fit-report.json": report, "fit-attempts.json": p.attempts(root)}
    for name in ("baseline-gate", "feature-receipts", "projection-receipts", "selection-receipts", "selected-indices", "contract", "reload-check"):
        payload[name+".json"] = p.value(root, name, signature)
    payload["model-receipt.json"] = fit.read_phase(root, "model", signature, ["model.pt", "context.json"])
    payload["normal-scores.json"] = read_normals(root, signature, identity["cohorts"], report["checkpoint_sha256"])
    payload["reload-scores.json"] = read_normals(root, signature, identity["cohorts"], report["checkpoint_sha256"], "reload")
    if eval_root is not None:
        for name in ("comparison", "warm", "stability", "location", "result"):
            payload["evaluation/"+name+".json"] = p.value(eval_root, name, signature)
        payload["evaluation/attempts.json"] = p.attempts(eval_root)
        payload["evaluation/scores.json"] = {g: [read_sample(eval_root/g, f"photo-{i:03d}", signature,
            photo, report["checkpoint_sha256"]) for i, photo in enumerate(identity["cohorts"][g])]
            for g in ("test", "validation")}
    payload["prediction-receipts.json"] = {str(path.relative_to(root)): json.loads(path.read_bytes())
        for path in sorted(root.glob("*-*/photo-*/receipt.json"))}
    if eval_root is not None:
        payload["evaluation/prediction-receipts.json"] = {str(path.relative_to(eval_root)): json.loads(path.read_bytes())
            for path in sorted(eval_root.glob("*/photo-*/receipt.json"))}
    raw = {k: fit.canonical(v) for k, v in payload.items()}
    raw["handoff-manifest.json"] = fit.canonical({k: hashlib.sha256(v).hexdigest() for k, v in raw.items()})
    if output.exists():
        with zipfile.ZipFile(output) as z:
            if len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw) or any(z.read(k) != v for k, v in raw.items()):
                raise ValueError("Existing handoff differs; preserved")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".patchcore-scaling-export-", dir=output.parent))/output.name
    with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED) as z:
        for name, content in raw.items():
            z.writestr(name, content)
    temp.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--worker", choices=("fit", "evaluation"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    project, archive = args.project.resolve(), args.archive.resolve()
    source, identity = p.prepare(project)
    if environment() != identity["environment"]:
        raise ValueError("GPU/software/reference implementation differs from reviewed probe")
    baseline = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(archive) != identity["archive_sha256"] or fit.sha256_file(baseline) != identity["baseline_checkpoint_sha256"]:
        raise ValueError("Original archive or baseline checkpoint changed")
    root, eval_root = project/"artifacts"/p.NAME, project/"artifacts/analyses"/p.NAME
    needed = 5 if (root/"identity").exists() else p.POLICY["initial_free_disk_GiB"]
    if shutil.disk_usage(project).free < needed*1024**3:
        raise RuntimeError(f"Need {needed} GiB free disk; no automatic deletion")
    if args.worker == "fit":
        report = fit_run(project, root, archive, source, identity)
        export(root, project/"artifacts/handoff/mpi-patchcore-scaling-fit-v1.zip", identity, report)
        return 0
    if args.worker == "evaluation":
        from scripts.mpi_patchcore_scaling_evaluation import evaluate
        with fit.locked_run(root):
            report = p.value(root, "result", fit.digest(identity))
            evaluate(project, root, eval_root, archive, source, identity, report)
            export(root, project/"artifacts/handoff"/(p.NAME+".zip"), identity, report, eval_root)
        return 0
    # Separate workers ensure the large projected matrix cannot coexist with inference.
    with fit.locked_run(project/"artifacts"/(p.NAME+"-launcher")):
        for worker in ("fit", "evaluation"):
            print(f"Starting isolated {worker} process. Fixed design; no automatic service change.", flush=True)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--project", str(project),
                "--archive", str(archive), "--worker", worker], check=False)
            if result.returncode:
                raise RuntimeError(f"{worker} stopped. Preserve files; send full error for review")
    print(json.dumps({"status": "completed_exploratory", "handoff": str(project/"artifacts/handoff"/(p.NAME+".zip"))}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Wait for cleanup. Same command resumes completed checkpoints; do not git pull mid-run.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}. Preserve all files; no shrinking, reroll or threshold changes.", file=sys.stderr)
        raise SystemExit(1)
