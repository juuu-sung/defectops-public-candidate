"""Paired observed-data evaluation; every accuracy outcome receives full diagnostics."""
from __future__ import annotations

import time
import zipfile

import numpy as np

from defectops.evaluation_metrics import binary_auroc
from scripts import mpi_patchcore_scaling_protocol as p
from scripts import run_mpi_patchcore_scaling as engine
from scripts import mpi_supersimplenet_map_diagnostic as maps
from scripts.mpi_score_probe_protocol import EVALUATION_SHA256

fit, ev = p.fit, engine.ev


def compare(source, candidate, method):
    methods = {"baseline": source["contract"]["candidate_top1_q90"], "candidate": method}
    result, predictions = p.comparison.compare(source["reference"], candidate, methods)
    keys = p.evaluation.POLICY["acceptance"]["regression_keys"]
    result["prior12"] = {name: {"retained": sum(s["alert"] for s in samples["test"] if s["key"] in keys),
        "missed_keys": [s["key"] for s in samples["test"] if s["key"] in keys and not s["alert"]]}
        for name, samples in predictions.items()}
    result["image_auroc"] = {name: binary_auroc([s["label"] for s in samples["test"]],
        [s["score"] for s in samples["test"]]) for name, samples in predictions.items()}
    result["new_false_positives"], result["resolved_false_positives"] = {}, {}
    for g in ("test", "validation"):
        pairs = list(zip(predictions["baseline"][g], predictions["candidate"][g], strict=True))
        result["new_false_positives"][g] = [b["key"] for a, b in pairs if not b["label"] and not a["alert"] and b["alert"]]
        result["resolved_false_positives"][g] = [b["key"] for a, b in pairs if not b["label"] and a["alert"] and not b["alert"]]
    result["miss_improvement_observed"] = bool(result["gained_detections"]) and not result["lost_baseline_detections"]
    result["passed_accuracy_targets"] = result.pop("passed")
    result["predictions"] = predictions
    return result


def stability(root, signature, baseline, candidate, check):
    arrays = {"baseline": p.comparison.matrix(baseline), "candidate": p.comparison.matrix(candidate)}
    labels = [s["label"] for s in baseline["test"]]
    folder = root/"stability-draws"
    folder.mkdir(exist_ok=True)
    trials = {}
    for scope, i, di, ci in p.evaluation.draws():
        check()
        name = f"{scope}-{i:03d}"
        record = {"scope": scope, "iteration": i, "development_indices": di.tolist(), "calibration_indices": ci.tolist()}
        for model, values in arrays.items():
            try:
                method = p.comparison.fit_method(values["development"][di], values["calibration"][ci])
                record[model] = {"valid": True, "counts": p.comparison.counts(values, method, labels),
                                 "threshold": method["threshold"]}
            except ValueError as exc:
                record[model] = {"valid": False, "error": str(exc)}
        record["valid"] = all(record[m]["valid"] for m in arrays)
        p.publish(folder, name, signature, record)
        trials.setdefault(scope, []).append(record)
    summaries = {s: p.evaluation.summarize(t) for s, t in trials.items()}
    return {"trials": trials, "summaries": summaries,
        "descriptive_summaries": {s: p.probe.evidence.summarize(t) for s, t in trials.items()},
        "passed": all(s["passed"] for s in summaries.values()),
        "interpretation": "Paired sensitivity to normal calibration samples, not population confidence intervals; invalid draws retained"}


def warm(runtime, archive, photos, expected, method, monitor):
    """Same five warmups and 20x3 normal photos; no file writes inside timing."""
    selected = photos[:p.POLICY["warm_photos"]]
    times = []
    for i in range(p.POLICY["warmup_photos"] + len(selected)*p.POLICY["warm_repeats"]):
        monitor.check()
        index = i % len(selected) if i < p.POLICY["warmup_photos"] else (i-p.POLICY["warmup_photos"]) % len(selected)
        actual, _ = engine.score_photo(archive, selected[index], runtime)
        replay = p.reload_check({"development": [], "calibration": [expected[index]]},
            {"development": [], "calibration": [actual]}, method)
        if not replay["passed"]:
            raise ValueError("Warm score replay failed")
        if i >= p.POLICY["warmup_photos"]:
            times.append(actual["photo_seconds"])
    return {"photo_keys": [s["key"] for s in selected], "warmup_photos": 5, "repeats": 3,
        "measured_photos": len(times), "seconds": times, "mean_seconds": float(np.mean(times)),
        "p95_higher_seconds": float(np.quantile(times, .95, method="higher")),
        "photos_per_second": len(times)/sum(times), "tiles_per_photo": 12, "elapsed_time_cap": None,
        "scope": "12 checked image reads+decode+preprocess+GPU inference+map transfer+top1; no model load, JSON/disk map write, HTTP, replay check or monitor inside timer"}


def location(project, fit_root, root, signature, archive, source, candidate, method, checkpoint_sha, check):
    baseline = ev.checked_bundle(project/"artifacts/handoff/mpi-pcb-normal-bank-evaluation-v1.zip", EVALUATION_SHA256)
    baseline_root = project/"artifacts/evaluations/mpi-pcb-normal-bank-evaluation-v1"
    src = {**source, "baseline_bundle": baseline}
    entries = [("test", i, "all_defects") for i, s in enumerate(candidate["test"]) if s["label"]]
    for group in ("development", "calibration", "test", "validation"):
        order = sorted((i for i, s in enumerate(candidate[group]) if not s["label"]),
            key=lambda i: (-p.comparison.c.scoring.score(candidate[group][i], method)["score"], candidate[group][i]["key"]))[:3]
        entries.extend((group, i, "highest_normals") for i in order)
    output = []
    for group, i, role in entries:
        check()
        photo = source["cohorts"][group][i]
        directory = fit_root/("first-"+group) if group in ("development", "calibration") else root/group
        sample = engine.read_sample(directory, f"photo-{i:03d}", signature, photo, checkpoint_sha)
        values = p.array(directory/f"photo-{i:03d}/maps.npy", (12, 256, 256))
        entry = {"group": group, "photo": photo}
        entry["baseline"] = maps.baseline_record(entry, src, baseline, baseline_root)
        prior = maps.load_baseline(entry, src, baseline_root)
        eligible = bool(photo["label"]) and photo["key"] not in p.POLICY["location_excluded"]
        masks = ev.read_masks(archive, photo) if eligible else [None]*12
        tiles = []
        for j, region in enumerate(fit.REGIONS):
            mask = masks[j]
            def statistics(values):
                return {k: v for k, v in maps.map_stats(values, mask).items() if not k.startswith("display_")}
            tiles.append({"region": region, "candidate_top1": sample["regions"][j]["top1"],
                "candidate": statistics(values[j]),
                "baseline": statistics(prior[j]) if prior is not None else None})
        output.append({"group": group, "key": photo["key"], "role": role, "location_eligible": eligible,
            "baseline_available": prior is not None, "tiles": tiles,
            "candidate_decision": p.comparison.c.scoring.score(sample, method)})
    return {"cases": output, "defect_photos": 55, "normal_high_photos": 12,
        "location_excluded": p.POLICY["location_excluded"], "new_pixel_threshold": False,
        "scope": "Numerical within-tile GT ranking and top656 overlap; no mask used in score/threshold; no probability or shared display scale claim.",
        "baseline_maps": "verified existing maps; missing baseline maps explicitly null; no new baseline test inference"}


def evaluate(project, fit_root, root, archive_path, source, identity, report, runtime_factory=engine.Runtime):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        engine.initialize(root, identity)
        model_receipt = fit.read_phase(fit_root, "model", signature, ["model.pt", "context.json"])
        checkpoint_sha, method = report["checkpoint_sha256"], report["contract"]
        if model_receipt["files"]["model.pt"] != checkpoint_sha or p.value(fit_root, "contract", signature) != method:
            raise ValueError("Frozen candidate model/contract changed")
        first = engine.read_normals(fit_root, signature, source["cohorts"], checkpoint_sha)
        reload = engine.read_normals(fit_root, signature, source["cohorts"], checkpoint_sha, "reload")
        if p.reload_check(first, reload, method) != p.value(fit_root, "reload-check", signature):
            raise ValueError("Fit reload evidence changed")
        p.comparison.contract_replay(first, method)
        if (root/"result").exists():
            for phase in ("comparison", "warm", "stability", "location"):
                p.value(root, phase, signature)
            for g in ("test", "validation"):
                for i, photo in enumerate(source["cohorts"][g]):
                    engine.read_sample(root/g, f"photo-{i:03d}", signature, photo, checkpoint_sha)
            return p.value(root, "result", signature)
        with engine.attempt(root, signature) as monitor, zipfile.ZipFile(archive_path) as archive:
            runtime = runtime_factory(source["config"]); monitor.runtime = runtime
            runtime.load(fit_root/"model/model.pt", report["context"])
            gate = engine.normal_gate(fit_root, signature, archive, source, runtime, checkpoint_sha, method)
            # Runtime gate can differ slightly across attempts; preserve each in attempt record folder.
            if not (root/"initial-normal-gate").exists():
                p.publish(root, "initial-normal-gate", signature, gate)
            else:
                p.value(root, "initial-normal-gate", signature)
            recovery = project/"artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png"
            print("Evaluation 1/4: all55 defects and normal54/256, regardless of prior12 accuracy.", flush=True)
            candidate = dict(first)
            for group in ("test", "validation"):
                candidate[group] = engine.collect(root, group, signature, archive, source["cohorts"][group], runtime,
                    monitor, checkpoint_sha, recovery, source["previous"])
            p.comparison.validate_raw(candidate, source["cohorts"])
            result = compare(source, candidate, method)
            p.publish(root, "comparison", signature, result)
            print("Evaluation 2/4: paired warm timing on the same normal photos, no time gate.", flush=True)
            if not (root/"warm").exists():
                timing = {"candidate": warm(runtime, archive, source["cohorts"]["calibration"], first["calibration"], method, monitor)}
                runtime.release()
                runtime.load(project/"artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt", source["records"]["fit.json"]["checkpoint_context"])
                timing["baseline"] = warm(runtime, archive, source["cohorts"]["calibration"], source["reference"]["calibration"],
                    source["contract"]["candidate_top1_q90"], monitor)
                p.publish(root, "warm", signature, timing)
            else:
                p.value(root, "warm", signature)
            monitor.check(); runtime.release(); monitor.runtime = None
            print("Evaluation 3/4: fixed-score map/GT diagnostics, no new inference or threshold.", flush=True)
            if not (root/"location").exists():
                p.publish(root, "location", signature, location(project, fit_root, root, signature, archive,
                    source, candidate, method, checkpoint_sha, monitor.check))
            else:
                p.value(root, "location", signature)
            print("Evaluation 4/4: all600 paired normal resampling draws, no accuracy early-stop.", flush=True)
            stable = stability(root, signature, source["reference"], candidate, monitor.check)
            p.publish(root, "stability", signature, stable)
            if fit.sha256_file(fit_root/"model/model.pt") != checkpoint_sha:
                raise ValueError("Model changed during evaluation")
        summary = {"status": "patchcore_scaling_observed_comparison_completed", "counts": result["counts"],
            "gained_detections": result["gained_detections"], "lost_detections": result["lost_baseline_detections"],
            "accuracy_targets_passed": result["passed_accuracy_targets"], "stability_targets_passed": stable["passed"],
            "miss_improvement_observed": result["miss_improvement_observed"], "image_auroc": result["image_auroc"],
            "prior12": result["prior12"], "all55_defects_evaluated": True, "normal54_and256_evaluated": True,
            "warm_evaluated": True, "location_evaluated": True, "stability_draws": 600,
            "independent_performance_claim_allowed": False, "automatic_pass_allowed": False,
            "service_model_changed": False, "checkpoint_sha256": checkpoint_sha, "contract": method,
            "cumulative_seconds": sum(a["seconds"] for a in p.attempts(root))}
        p.publish(root, "result", signature, summary)
        return summary
