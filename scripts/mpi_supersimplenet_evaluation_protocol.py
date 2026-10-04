"""Frozen SuperSimpleNet final-checkpoint evaluation; native signed logits only."""
from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np

from scripts import mpi_supersimplenet_fit_protocol as p
from scripts import mpi_efficientad_evaluation_protocol as shared
from copy import deepcopy
from scripts import run_mpi_spatial_score as inputs
from scripts import mpi_normal_diversity_evaluation_protocol as comparison

fit = p.fit
NAME = "mpi-supersimplenet-evaluation-v1"
FIT_SHA = "ae4ccc9e7164e2a052efd1c3ef9be600f956f9fbab8eae7898e1f1cda73ce2e7"
ACCEPTANCE = deepcopy(shared.POLICY["acceptance"])
for key in ("minimum_tp55", "lost_baseline_detections_allowed", "maximum_fp54", "maximum_fp256", "warm_photo_p95_seconds"):
    assert ACCEPTANCE[key] == p.POLICY["future_evaluation"][key]
ACCEPTANCE["stability"]["refit"] = "region native-logit median/Q90 width and photo Q95; no map quantiles"
POLICY = {"acceptance": ACCEPTANCE,
          "resource_plan": p.POLICY["resource_plan"],
          "warmup_photos": 5, "warm_photos": 20, "warm_repeats": 3,
          "warm_selection": "first_20_fixed_calibration_photos_in_manifest_order",
          "warm_scope": "ZIP read, integrity, decode, preprocessing, inference, map transfer, fixed score; no save/UI/load",
          "free_disk_GiB": 12, "threshold_refitted_for_main_comparison": False,
          "bootstrap_after": "screen_and_full_accuracy_and_warm_gates_pass",
          "automatic_pass_allowed": False, "service_model_changed": False,
          "independent_performance_claim_allowed": False}


def value(root, phase, signature):
    fit.read_phase(root, phase, signature, ["value.json"])
    return json.loads((root / phase / "value.json").read_bytes())


def read_fit(project, handoffs, root):
    path = handoffs / (p.NAME + ".zip")
    if fit.sha256_file(path) != FIT_SHA:
        raise ValueError("Expected the reviewed 28800-update fit handoff")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != {*manifest, "handoff-manifest.json"}:
            raise ValueError("Fit ZIP inventory differs")
        records = {}
        for name, sha in manifest.items():
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Fit payload changed")
            records[name] = json.loads(raw)
    plan = records["plan.json"]
    signature = fit.digest(plan)
    if plan["policy"] != p.POLICY or value(root, "identity", signature) != plan:
        raise ValueError("Frozen training policy/identity differs")
    # Check historical dependencies without adding new evaluation files to the old identity.
    for name, sha in plan["sources_lf_sha256"].items():
        if hashlib.sha256((project / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Frozen fit implementation changed: {name}")
    p.validate_cohorts(plan["cohorts"])
    for phase, record in (("calibrated", "normal-results.json"), ("reload", "reload.json"), ("result", "report.json")):
        if value(root, phase, signature) != records[record]:
            raise ValueError(f"Reviewed {phase} differs")
    if (records["report.json"]["completed_updates"] != 28800
            or records["report.json"]["completed_epochs"] != 300
            or records["reload.json"]["photos"] != 297
            or records["reload.json"]["decision_changes"] != 0
            or records["reload.json"]["contract_refitted"]
            or not records["reload.json"]["passed"]
            or records["report.json"]["test_images_decoded"] != 0):
        raise ValueError("Fit not eligible")
    receipt = fit.read_phase(root, "step-028800", signature, ["model.pt", "reference.npy", "progress.json"])
    if receipt != records["checkpoint-receipts.json"][-1]:
        raise ValueError("Final checkpoint differs from reviewed receipt")
    if json.loads((root / "step-028800/progress.json").read_bytes()) != records["progress.json"]:
        raise ValueError("Final checkpoint progress differs")
    expected = {(r["photo_id"], r["rows_sha256"]): r for r in records["prediction-receipts.json"]}
    for group in ("development", "calibration"):
        cohort = plan["cohorts"][group]
        for i, pid in enumerate(cohort["photo_ids"]):
            actual = fit.read_phase(root / group, f"photo-{i:03d}", signature, ["predictions.npy"])
            key = pid, fit.digest(cohort["rows"][i*12:(i+1)*12])
            if actual != expected[key]:
                raise ValueError("Normal prediction receipt differs")
            raw = p.validate_predictions(np.load(root / group / f"photo-{i:03d}" / "predictions.npy", allow_pickle=False))
            if raw.shape != (12, 65537) or p.tile_scores(raw).tolist() != records["normal-results.json"]["tile_scores"][group][i]:
                raise ValueError("Normal raw logits differ from reviewed calibration")
    normal = records["normal-results.json"]
    if p.calibrate(**normal["tile_scores"]) != normal["contract"]:
        raise ValueError("Frozen normal score calibration differs")
    return records


def prepare(project, handoffs, fit_root):
    records = read_fit(project, handoffs, fit_root)
    source = inputs.prepare(project, handoffs)
    comparison.validate_raw(source["reference"], source["cohorts"])
    baseline = source["contract"]["candidate_top1_q90"]
    comparison.contract_replay(source["reference"], baseline)
    labels = [s["label"] for s in source["cohorts"]["test"]]
    if comparison.counts(comparison.matrix(source["reference"]), baseline, labels) != {
            "tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14}:
        raise ValueError("Reviewed baseline changed")
    for g in ("development", "calibration"):
        old = records["plan.json"]["cohorts"][g]
        photos = source["cohorts"][g]
        if old["photo_ids"] != [s["photo_id"] for s in photos]:
            raise ValueError("Normal cohort order differs")
        for a, b in zip(old["rows"], [r for s in photos for r in s["regions"]], strict=True):
            if any(a[k] != b[k] for k in ("member", "decoded_rgb_sha256", "region")):
                raise ValueError("Normal cohort pixels differ")
    keys = POLICY["acceptance"]["regression_keys"]
    refs = {s["key"]: s for s in source["reference"]["test"]}
    if len(keys) != 12 or any(not refs[k]["label"] or not comparison.c.scoring.score(refs[k], baseline)["alert"] for k in keys):
        raise ValueError("Frozen 12 prior detections differ")
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project / "docs/mpi-supersimplenet-evaluation-v1.md"])
    identity = {"name": NAME, "policy": POLICY, "fit_zip_sha256": FIT_SHA,
                "fit_signature": fit.digest(records["plan.json"]),
                "checkpoint_sha256": records["checkpoint-receipts.json"][-1]["files"]["model.pt"],
                "source": source["identity"], "cohorts": source["cohorts"],
                "contract": records["normal-results.json"]["contract"],
                "environment": records["plan.json"]["environment"],
                "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
                    f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in files}}
    return records, source, identity


def decision(tiles, contract):
    tiles = np.asarray(tiles, dtype=np.float64)
    if tiles.shape != (12,) or not np.isfinite(tiles).all():
        raise ValueError("Expected 12 finite signed tile scores")
    p.photo_scores(tiles, contract)  # validate centers, widths and finite inputs
    if not np.isfinite(contract["threshold"]):
        raise ValueError("Nonfinite threshold")
    scaled = (tiles - contract["centers"]) / contract["scales"]
    score = float(scaled.max())
    return {"score": score, "alert": score >= contract["threshold"],
            "winning_region": fit.REGIONS[int(scaled.argmax())]}


def sample(photo, tiles, contract):
    return {k: photo[k] for k in ("key", "photo_id", "label")} | {
        "tile_scores": np.asarray(tiles).tolist(), **decision(tiles, contract)}


def screen(samples):
    if [s["key"] for s in samples] != POLICY["acceptance"]["regression_keys"] or any(s["label"] != 1 for s in samples):
        raise ValueError("Screen inventory differs")
    lost = [s["key"] for s in samples if not s["alert"]]
    return {"passed": not lost, "retained": 12-len(lost), "lost_keys": lost,
            "performance_estimate": False, "predictions": samples}


def metrics(samples, validation):
    if len(samples) != 109 or sum(s["label"] for s in samples) != 55 or len(validation) != 256 or any(s["label"] for s in validation):
        raise ValueError("Full comparison inventory differs")
    return {"tp": sum(s["label"] == 1 and s["alert"] for s in samples),
            "fn": sum(s["label"] == 1 and not s["alert"] for s in samples),
            "fp54": sum(s["label"] == 0 and s["alert"] for s in samples),
            "tn54": sum(s["label"] == 0 and not s["alert"] for s in samples),
            "fp256": sum(s["alert"] for s in validation)}


def compare(samples, source):
    method = source["contract"]["candidate_top1_q90"]
    baseline = [{k: s[k] for k in ("key", "label")} | comparison.c.scoring.score(s, method)
                for s in source["reference"]["test"]]
    if [s["key"] for s in samples["test"]] != [s["key"] for s in baseline]:
        raise ValueError("Paired test order differs")
    lost = [a["key"] for a, b in zip(baseline, samples["test"], strict=True) if a["label"] and a["alert"] and not b["alert"]]
    gained = [a["key"] for a, b in zip(baseline, samples["test"], strict=True) if a["label"] and not a["alert"] and b["alert"]]
    counts = metrics(samples["test"], samples["validation"])
    a = POLICY["acceptance"]
    gates = {"tp_at_least_53": counts["tp"] >= a["minimum_tp55"], "no_lost_baseline_detections": not lost,
             "fp54_at_most_2": counts["fp54"] <= a["maximum_fp54"], "fp256_at_most_12": counts["fp256"] <= a["maximum_fp256"]}
    return {"counts": counts, "lost_keys": lost, "gained_keys": gained,
            "miss_improvement_observed": bool(gained), "gates": gates, "passed": all(gates.values()),
            "baseline_counts": {"tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14}}


def draws():
    s = POLICY["acceptance"]["stability"]
    for scope, seed in zip(s["scopes"], s["PCG64_seeds"], strict=True):
        rng = np.random.Generator(np.random.PCG64(seed))
        for i in range(s["repetitions_per_scope"]):
            di = rng.integers(0, 128, 128) if scope in ("development_only", "both") else np.arange(128)
            ci = rng.integers(0, 169, 169) if scope in ("calibration_only", "both") else np.arange(169)
            yield scope, i, di, ci


def summarize(trials):
    policy = POLICY["acceptance"]["stability"]
    if len(trials) != policy["repetitions_per_scope"] or any(not t["valid"] for t in trials):
        return {"passed": False, "invalid": sum(not t["valid"] for t in trials), "reason": "Missing/invalid draws never replaced"}
    q = {m: {k: np.quantile([t[m]["counts"][k] for t in trials], [.025, .5, .975], method="linear").tolist()
             for k in ("tp", "fp54", "fp256")} for m in ("baseline", "candidate")}
    width = {m: q[m]["tp"][2] - q[m]["tp"][0] for m in q}
    gates = {"tp_q025": q["candidate"]["tp"][0] >= policy["minimum_tp_q025"],
             "tp_width": width["candidate"] <= width["baseline"] + policy["maximum_tp_interval_width_over_baseline"],
             "fp54_q975": q["candidate"]["fp54"][2] <= policy["maximum_fp54_q975"],
             "fp256_q975": q["candidate"]["fp256"][2] <= policy["maximum_fp256_q975"]}
    return {"passed": all(gates.values()), "invalid": 0, "quantiles": q, "tp_width": width, "gates": gates}
