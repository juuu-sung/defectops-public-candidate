"""C-stage: locked normal-bank fit, normal calibration and reload; never open test."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"src")]
from scripts import mpi_normal_diversity_fit_protocol as p
from scripts import prepare_mpi_normal_diversity as preparation

fit, bundle = p.fit, preparation.bundle_tools


class Budget(preparation.Monitor):
    def __init__(self, runtime, policy, prior_seconds):
        # C-stage has its own four-hour / seven-GiB budget, not the selection budget.
        limits = dict(policy["resource_plan"])
        limits["selection_seconds"] = limits["fit_calibration_reload_seconds"]
        limits["selection_and_inference_reserved_GiB"] = limits["fit_reserved_GiB"]
        super().__init__(runtime, {"resource_plan": limits}, prior_seconds)

    def check(self):
        try:
            super().check()
        except RuntimeError as exc:
            raise RuntimeError(f"C-stage resource/monitor check stopped: {exc}") from exc


class Runtime(fit.TorchRuntime):
    def fit_locked(self, images, baseline_checkpoint, monitor):
        """Reuse the old fit implementation and exact backbone tensors, no download."""
        from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy
        original_gpu = self.gpu
        constructor = original_gpu[2]
        torch = self.torch
        checkpoint = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
        prefix = "model.feature_extractor."
        weights = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
        if not weights:
            raise ValueError("Baseline feature extractor tensors missing")
        del checkpoint

        def frozen_constructor(**kwargs):
            kwargs["pre_trained"] = False
            model = constructor(**kwargs)
            extractor = model.model.feature_extractor
            extractor.load_state_dict(weights, strict=True)
            if any(not torch.equal(v.cpu(), weights[k]) for k, v in extractor.state_dict().items()):
                raise ValueError("Frozen backbone differs after loading")
            weights.clear()
            return model

        count = 0
        original_update = KCenterGreedy.update_distances

        def checked_update(sampler, *args, **kwargs):
            nonlocal count
            if count % 25 == 0:
                monitor.check()
            result = original_update(sampler, *args, **kwargs)
            count += 1
            return result

        # All patch extraction/stride/coreset code remains the frozen baseline implementation.
        self.gpu = (*original_gpu[:2], frozen_constructor, *original_gpu[3:])
        try:
            with patch.object(KCenterGreedy, "update_distances", checked_update):
                metadata = super().fit(images)
            monitor.check()
            metadata["frozen_backbone_loaded_from_baseline"] = True
            metadata["coreset_budget_check_interval_iterations"] = 25
            return metadata
        finally:
            self.gpu = original_gpu


def persist(root, name, signature, payload):
    if not (root/name).exists():
        with fit.phase_stage(root, name) as stage:
            fit.write_json(stage/"value.json", payload)
            fit.seal_phase(stage, name, signature, {}, ["value.json"])
    fit.read_phase(root, name, signature, ["value.json"])
    if json.loads((root/name/"value.json").read_bytes()) != payload:
        raise ValueError(f"Stored {name} differs; preserved")


def read_score(folder, name, signature, photo, environment):
    receipt = fit.read_phase(folder, name, signature, ["score.json"])
    if receipt["photo_sha256"] != fit.digest(photo) or receipt["environment"] != environment:
        raise ValueError("Cached score provenance differs")
    sample = json.loads((folder/name/"score.json").read_bytes())
    p.scoring.old.validate_sample(sample, photo)
    return sample


def score_photo(source, photo, runtime, monitor):
    sample = {k: photo[k] for k in ("key", "photo_id", "label")}
    sample["regions"] = []
    for row in photo["regions"]:
        monitor.check()
        with fit.checked_image(source, row) as image:
            raw, values = runtime.predict(image)
        values = np.asarray(values)
        if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid normal anomaly map")
        top = fit.candidate_image_score(raw, values, fit.CANDIDATE)
        sample["regions"].append({"region": row["region"], "member": row["member"],
                                   "top1": float(top), "native_raw": float(raw)})
    p.scoring.old.validate_sample(sample, photo)
    monitor.check()
    return sample


def cached_scores(root, phase, signature, cohorts, env):
    folder = root/phase
    folder.mkdir(exist_ok=True)
    expected = {f"{group}-{photo['photo_id']}": photo for group, photos in cohorts.items() for photo in photos}
    if any(f.name not in expected and not f.name.startswith(".") for f in folder.iterdir()):
        raise ValueError("Unexpected normal score cache entry")
    return {name: read_score(folder, name, signature, photo, env)
            for name, photo in expected.items() if (folder/name).exists()}


def collect(root, phase, source, signature, cohorts, env, runtime, monitor):
    cache = cached_scores(root, phase, signature, cohorts, env)
    result = {}
    for group, photos in cohorts.items():
        result[group] = []
        for i, photo in enumerate(photos):
            name = f"{group}-{photo['photo_id']}"
            if name not in cache:
                sample = score_photo(source, photo, runtime, monitor)
                with fit.phase_stage(root/phase, name) as stage:
                    fit.write_json(stage/"score.json", sample)
                    fit.seal_phase(stage, name, signature,
                                   {"photo_sha256": fit.digest(photo), "environment": env}, ["score.json"])
                cache[name] = read_score(root/phase, name, signature, photo, env)
            result[group].append(cache[name])
            if (i+1) % 10 == 0 or i+1 == len(photos):
                print(f"{phase} {group} {i+1}/{len(photos)} normal photos", flush=True)
    return result


def execute(root, source, baseline, chosen, cohorts, records, identity, handoff,
            runtime_factory=Runtime, monitor_factory=Budget, environment_fn=preparation.environment):
    signature = fit.digest(identity)
    config, policy = identity["model_config"], identity["policy"]
    context = {"category": "mpi_pcb", "seed": config["seed"], "manifest_sha256": p.ROWS_SHA,
               "execution_config_sha256": signature}
    p.validate_cohorts(chosen, cohorts)
    p.validate_archive(source, chosen, cohorts)
    with fit.locked_run(root):
        if not (root/"identity").exists() and any(f.name != ".run.lock" and not f.name.startswith(".identity-attempt-") for f in root.iterdir()):
            raise ValueError("Outputs without identity; preserve for review")
        persist(root, "identity", signature, identity)
        prior = preparation.read_attempts(root)
        training = root/"training"
        env = None
        if (root/"environment").exists():
            fit.read_phase(root, "environment", signature, ["value.json"])
            env = json.loads((root/"environment/value.json").read_bytes())
        if training.exists():
            fit.read_phase(root, "training", signature, ["model.pt", "metadata.json"])
            if env is None:
                raise ValueError("Training has no environment receipt")
        elif any((root/k).exists() for k in ("first", "contract", "reload", "verified", "result")):
            raise ValueError("Scores without model checkpoint")
        score_signature = fit.digest({"execution": signature, "model": fit.sha256_file(training/"model.pt")}) if training.exists() else None
        for phase in ("first", "reload"):
            if (root/phase).exists():
                cached_scores(root, phase, score_signature, cohorts, env)
        if (root/"result").exists():
            bundle.verify(root/"result", identity)
            fit.read_phase(root, "verified", signature, ["value.json"])
            for phase in ("first", "reload"):
                if len(cached_scores(root, phase, score_signature, cohorts, env)) != 297:
                    raise ValueError("Completed export has missing normal score caches; preserve for review")
            # Recheck live caches and contract, not only the export receipt.
            first = collect(root, "first", source, score_signature, cohorts, env, None, None)
            second = collect(root, "reload", source, score_signature, cohorts, env, None, None)
            contract = p.calibrate(first["development"], first["calibration"])
            persist(root, "contract", score_signature, contract)
            persist(root, "verified", signature, p.reload_check(first, second, contract))
            for name, payload in (("first", first), ("reload", second), ("contract", contract)):
                if json.loads((root/f"result/{name}.json").read_bytes()) != payload:
                    raise ValueError("Export differs from verified live scores or contract")
            bundle.bundle(root/"result", handoff)
            return json.loads((root/"result/report.json").read_bytes())
        if prior >= policy["resource_plan"]["fit_calibration_reload_seconds"]:
            raise RuntimeError("Previous attempts exhausted C-stage time budget")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        started = time.perf_counter()
        runtime = monitor = None
        outcome, error = "interrupted", None
        costs = {}
        try:
            runtime = runtime_factory(config)
            current_env = environment_fn(runtime)
            if current_env != identity["selection_environment"]:
                raise ValueError("Runtime differs from reviewed selection environment")
            if runtime.environment() != records["reload.json"]["receipt"]["runtime"]:
                raise ValueError("Runtime differs from baseline; do not silently change environment")
            if env is not None and current_env != env:
                raise ValueError("Runtime differs from cached C-stage results")
            env = current_env
            persist(root, "environment", signature, env)
            with monitor_factory(runtime, policy, prior+time.perf_counter()-started) as monitor:
                if not training.exists():
                    print("1/3 Building bank from locked 64 normal photos x 12 regions. No test data.", flush=True)
                    def images():
                        for row in chosen["training_rows"]:
                            monitor.check()
                            yield fit.checked_image(source, row)
                            monitor.check()
                    before = time.perf_counter()
                    with fit.phase_stage(root, "training") as stage:
                        metadata = runtime.fit_locked(images(), baseline, monitor)
                        if metadata["memory_bank_shape"] != [78643, 768]:
                            raise ValueError("Frozen bank shape differs")
                        runtime.save(stage/"model.pt", context)
                        monitor.check()
                        fit.write_json(stage/"metadata.json", metadata | {"context": context})
                        fit.seal_phase(stage, "training", signature, {}, ["model.pt", "metadata.json"])
                    costs["fit_and_checkpoint_seconds"] = time.perf_counter()-before
                else:
                    runtime.load(training/"model.pt", context)
                    monitor.check()
                score_signature = fit.digest({"execution": signature, "model": fit.sha256_file(training/"model.pt")})
                print("2/3 Scoring development128 and calibration169 normals; calculating fixed Q90 contract.", flush=True)
                before = time.perf_counter()
                first = collect(root, "first", source, score_signature, cohorts, env, runtime, monitor)
                contract = p.calibrate(first["development"], first["calibration"])
                persist(root, "contract", score_signature, contract)
                costs["normal_scoring_and_contract_seconds"] = time.perf_counter()-before
                monitor.check()
                print("3/3 Reloading saved model and checking all 297 normals against the FIRST contract.", flush=True)
                before = time.perf_counter()
                runtime.release()
                gc.collect()
                runtime.load(training/"model.pt", context)
                monitor.check()
                second = collect(root, "reload", source, score_signature, cohorts, env, runtime, monitor)
                verification = p.reload_check(first, second, contract)
                monitor.check()
                costs["reload_and_verification_seconds"] = time.perf_counter()-before
                persist(root, "verified", signature, verification)
                outcome = "completed"
        except BaseException as exc:
            outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if runtime is not None:
                    runtime.release()
            except BaseException as exc:
                outcome, error = "failed", f"Runtime release failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                fit.write_json(attempt/"finished.json", {"outcome": outcome, "error": error,
                    "seconds": time.perf_counter()-started, "phase_seconds_this_attempt": costs,
                    "memory": monitor.snapshot() if monitor is not None else None,
                    "scope": "runtime setup through release; excludes input hashing, export and offline waiting"})
        attempts = [json.loads(f.read_bytes()) for f in sorted(root.glob("attempt-*/finished.json"))]
        total = preparation.read_attempts(root)
        if total > policy["resource_plan"]["fit_calibration_reload_seconds"]:
            raise RuntimeError("C-stage time budget exceeded including runtime release")
        report = {"status": "normal_diversity_fit_calibrated_reload_verified_no_test",
                  "training_original_photos": 64, "training_tiles": 768,
                  "development_original_photos": 128, "calibration_original_photos": 169,
                  "memory_bank_shape": [78643, 768], "image_threshold": contract["threshold"],
                  "checkpoint_sha256": fit.sha256_file(training/"model.pt"), "checkpoint_context": context,
                  "reload_verification": verification, "cumulative_attempt_seconds": total,
                  "attempts_with_errors": sum(a["error"] is not None for a in attempts),
                  "resource_review_required": any(a["error"] is not None or a["memory"] is None for a in attempts),
                  "training_ids_sha256": p.IDS_SHA, "test_images_decoded": 0, "test_masks_decoded": 0,
                  "test_scores_computed": 0, "validation_scores_computed": 0,
                  "automatic_pass_allowed": False, "service_model_changed": False,
                  "performance_improvement_established": False}
        with fit.phase_stage(root, "result") as stage:
            for name, payload in {"report": report, "plan": identity, "selection": chosen,
                                  "environment": env, "contract": contract, "first": first, "reload": second,
                                  "attempts": attempts, "training": json.loads((training/"metadata.json").read_bytes())}.items():
                fit.write_json(stage/f"{name}.json", payload)
            bundle.seal(stage, identity)
        bundle.bundle(root/"result", handoff)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        chosen, cohorts, records, identity = p.prepare(project, project/"artifacts/handoff")
        baseline = project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        policy = identity["policy"]
        print("Checking archive and baseline checkpoint hashes before GPU work...", flush=True)
        if (fit.sha256_file(args.archive) != policy["source_sha256"]["archive"]
                or fit.sha256_file(baseline) != policy["source_sha256"]["baseline_checkpoint"]):
            raise ValueError("Original ZIP or baseline checkpoint differs")
        handoff = project/"artifacts/handoff"/f"{p.NAME}.zip"
        with zipfile.ZipFile(args.archive) as source:
            report = execute(project/"artifacts/experiments/mpi-normal-diversity-v1/fit", source,
                             baseline, chosen, cohorts, records, identity, handoff)
        print(json.dumps(report | {"handoff": str(handoff)}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Files preserved. Do not delete outputs, reroll selection or open test. Send the error for review.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
