"""Frozen A/C/D comparison: all observed cohorts, never an accuracy stop gate."""
import hashlib
import json
import zipfile
from copy import deepcopy

import numpy as np

from scripts import mpi_supersimplenet_evaluation_protocol as old
from scripts import mpi_supersimplenet_scaling_protocol as scaling

fit, p = old.fit, old.p
NAME = "mpi-supersimplenet-scaling-evaluation-v1"
FIT_SHA = "ff4be1f4e7f3df369a2ae82cdaf8a9a297dfb0af23662061a59b169965726f33"
POLICY = deepcopy(old.POLICY)
POLICY.update(accuracy_stop_gate=False, elapsed_time_cap=None,
              bootstrap_after="all full scores regardless of accuracy or latency",
              free_disk_GiB=20, location_excluded_photos=["defect/0006"],
              primary_score="frozen native classifier logits; maps diagnostic only",
              map_scope="all55 defects and highest3 observed normals per normal group",
              baseline_maps="verified existing maps only; absence reported explicitly")
POLICY["resource_plan"]["comparison_seconds"] = None
POLICY["acceptance"]["warm_photo_p95_seconds"] = None
POLICY["acceptance"]["first_gate"] = "prior12_descriptive_only_never_blocks_full_comparison"
POLICY["acceptance"]["later_evaluation_compute_cap_seconds"] = None
POLICY["acceptance"]["executed_by_this_runner"] = True


def read_bundle(path):
    if fit.sha256_file(path) != FIT_SHA:
        raise ValueError("Expected reviewed completed A/C/D scaling handoff")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {"handoff-manifest.json"}:
            raise ValueError("Scaling ZIP inventory differs")
        result = {}
        for name, digest in manifest.items():
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Scaling ZIP payload differs")
            result[name] = json.loads(raw)
        return result


def read_arm(project, root, data, arm):
    suite = data["plan.json"]
    if old.value(root, "identity", fit.digest(suite)) != suite:
        raise ValueError("Local scaling suite differs")
    records = {k.removeprefix(arm+"/"): v for k, v in data.items() if k.startswith(arm+"/")}
    plan = records["plan.json"]
    if plan != scaling.arm_identity(suite, arm):
        raise ValueError("Scaling arm policy differs")
    signature, directory = fit.digest(plan), root/arm
    if old.value(directory, "identity", signature) != plan:
        raise ValueError("Local arm identity differs")
    for name, sha in plan["sources_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Historical fit code changed: {name}")
    for phase, name in (("calibrated", "normal-results.json"), ("reload", "reload.json"), ("result", "report.json")):
        if old.value(directory, phase, signature) != records[name]:
            raise ValueError("Local fit result differs")
    policy, report, reload = plan["policy"], records["report.json"], records["reload.json"]
    if (report["completed_updates"] != policy["updates"] or report["completed_epochs"] != policy["epochs"]
            or not reload["passed"] or reload["photos"] != 297 or reload["decision_changes"] != 0
            or reload["contract_refitted"] or report["test_images_decoded"] != 0):
        raise ValueError("Incomplete or changed fit")
    phase = f"step-{policy['updates']:06d}"
    receipt = fit.read_phase(directory, phase, signature, ["model.pt", "reference.npy", "progress.json"])
    if receipt != records["checkpoint-receipts.json"][-1]:
        raise ValueError("Final checkpoint differs from reviewed receipt")
    if json.loads((directory/phase/"progress.json").read_bytes()) != records["progress.json"]:
        raise ValueError("Final progress differs")
    expected = {(r["photo_id"], r["rows_sha256"]): r for r in records["prediction-receipts.json"]}
    for group in ("development", "calibration"):
        cohort = plan["cohorts"][group]
        for i, pid in enumerate(cohort["photo_ids"]):
            name = f"photo-{i:03d}"
            actual = fit.read_phase(directory/group, name, signature, ["predictions.npy"])
            if actual != expected[pid, fit.digest(cohort["rows"][i*12:(i+1)*12])]:
                raise ValueError("Normal prediction provenance differs")
            raw = p.validate_predictions(np.load(directory/group/name/"predictions.npy", allow_pickle=False))
            if raw.shape != (12, 65537) or raw[:, 0].tolist() != records["normal-results.json"]["tile_scores"][group][i]:
                raise ValueError("Normal predictions differ from frozen contract")
    normal = records["normal-results.json"]
    if p.calibrate(**normal["tile_scores"]) != normal["contract"]:
        raise ValueError("Normal contract cannot be reproduced")
    return records


def prepare(project, data):
    source = old.inputs.prepare(project, project/"artifacts/handoff")
    old.comparison.validate_raw(source["reference"], source["cohorts"])
    contract = source["contract"]["candidate_top1_q90"]
    old.comparison.contract_replay(source["reference"], contract)
    counts = old.comparison.counts(old.comparison.matrix(source["reference"]), contract,
                                   [s["label"] for s in source["cohorts"]["test"]])
    if counts != {"tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14}:
        raise ValueError("Frozen Q90 baseline differs")
    for group in ("development", "calibration"):
        expected = data["plan.json"]["original_cohorts"][group]
        actual = source["cohorts"][group]
        if expected["photo_ids"] != [s["photo_id"] for s in actual]:
            raise ValueError("Normal photo order differs")
        for a, b in zip(expected["rows"], [r for s in actual for r in s["regions"]], strict=True):
            if any(a[k] != b[k] for k in ("member", "region", "decoded_rgb_sha256")):
                raise ValueError("Normal pixels differ")
    training = set(data["plan.json"]["expanded_training"]["photo_ids"])
    if training & {s["photo_id"] for s in source["cohorts"]["validation"]}:
        raise ValueError("Expanded training overlaps validation")
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project/"scripts/run_mpi_supersimplenet_scaling_evaluation.ps1",
                    project/"docs/mpi-supersimplenet-scaling-evaluation-v1.md"])
    identity = {"name": NAME, "policy": POLICY, "fit_zip_sha256": FIT_SHA,
        "scaling_plan_sha256": fit.digest(data["plan.json"]), "source": source["identity"],
        "cohorts": source["cohorts"], "environment": data["plan.json"]["environment"],
        "arms": {a: {"fit_signature": fit.digest(data[a+"/plan.json"]),
                     "checkpoint_sha256": data[a+"/checkpoint-receipts.json"][-1]["files"]["model.pt"],
                     "contract": data[a+"/normal-results.json"]["contract"]} for a in scaling.ARMS},
        "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in paths}}
    return source, identity


def summarize(trials):
    """Keep invalid draws; summarize valid draws explicitly, never reroll."""
    result = {}
    for model in ("baseline", "candidate"):
        valid = [t[model] for t in trials if t[model]["valid"]]
        result[model] = {"valid": len(valid), "invalid": len(trials)-len(valid),
            "quantiles_conditional_on_valid_draws": {k: np.quantile(
                [t["counts"][k] for t in valid], [.025, .5, .975], method="linear").tolist()
                for k in ("tp", "fp54", "fp256")} if valid else None}
    return result
