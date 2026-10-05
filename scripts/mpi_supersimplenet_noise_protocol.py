"""Two locked noise amplitudes; common-difficulty evaluation, never model selection."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np

from scripts import mpi_supersimplenet_varying_protocol as v

fit = v.fit
NAME = "mpi-supersimplenet-noise-strength-v1"
BASELINE_ZIP_SHA = "109090f6e97d8759ceb5bc7658fa742aff46a97be7559451099614b3ea9ebc1c"
STRENGTHS = {"std015": .015, "std030": .03, "std060": .06}
ARMS = ["std030", "std060"]
POLICY = deepcopy(v.POLICY)
POLICY.update({
    "snapshots": [0, 100, 300, 500], "training_arms": ARMS,
    "evaluation_strengths": STRENGTHS, "primary_evaluation_strength": "std015",
    "synthetic": "official v2.2.0; only anomaly_generator.noise_std changes; perlin0.6, mean0",
    "fixed_control": "reviewed varying500 std0.015 model; read-only cross-strength evaluation",
    "maximum_attempted_updates": 1200, "free_disk_GiB": 16,
    "resource_plan": {"comparison_seconds": 2700, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "interpretation": "same one normal photo/eight tiles and previously observed held-out seeds; exploratory synthetic diagnosis only",
    "automatic_model_selection": False,
})
EVIDENCE = ["arrays.npz", "progress.json"]
FILES = ["model.pt"] + EVIDENCE


def prepare(project, handoffs, fit_root, fixed_root, varying_root):
    records, prior_identity, _ = v.prepare(project, handoffs, fit_root, fixed_root)
    path = handoffs / (v.NAME + ".zip")
    if fit.sha256_file(path) != BASELINE_ZIP_SHA:
        raise ValueError("Reviewed varying-synthetic ZIP required; preserve original")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if (len(z.namelist()) != len(manifest) + 1 or
                set(z.namelist()) != set(manifest) | {"handoff-manifest.json"} or
                any(hashlib.sha256(z.read(n)).hexdigest() != h for n, h in manifest.items())):
            raise ValueError("Varying ZIP inventory/checksum differs")
        plan = json.loads(z.read("plan.json"))
        if any(plan.get(k) != value for k, value in prior_identity.items()):
            raise ValueError("Varying baseline provenance differs")
        signature = fit.digest(plan)
        if v.old.e.value(varying_root, "identity", signature) != plan:
            raise ValueError("Baseline local identity differs")
        progress, arrays = v.read_phase(varying_root, "step-0500", signature, True)
        receipt = fit.read_phase(varying_root, "step-0500", signature, v.FILES)
        if receipt != json.loads(z.read("step-0500/receipt.json")):
            raise ValueError("Baseline final receipt differs from reviewed ZIP")
        with np.load(io.BytesIO(z.read("step-0000/arrays.npz")), allow_pickle=False) as a:
            original_arrays = {k: a[k] for k in a.files}
    identity = {**prior_identity, "name": NAME, "policy": POLICY,
        "baseline_zip_sha256": BASELINE_ZIP_SHA, "baseline_signature": signature,
        "baseline_checkpoint_sha256": receipt["files"]["model.pt"],
        "baseline_fingerprints": progress["fingerprints"],
        "baseline_training_targets": [r["target_sha256"] for r in progress["losses"]]}
    identity["sources_lf_sha256"] = dict(prior_identity["sources_lf_sha256"])
    for name in ("scripts/mpi_supersimplenet_noise_protocol.py", "scripts/mpi_supersimplenet_noise_runtime.py",
                 "scripts/run_mpi_supersimplenet_noise.py", "scripts/run_mpi_supersimplenet_noise.ps1",
                 "docs/mpi-supersimplenet-noise-strength-v1.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return records, identity, {"baseline": arrays, "original": original_arrays}


def arrays_for(flat, strength):
    prefix = strength + "__"
    return {k[len(prefix):]: a for k, a in flat.items() if k.startswith(prefix)}


def summarize(arrays):
    result = v.summarize(arrays)
    result["counterfactual"]["clean_scores_with_synthetic_labels_auroc"] = v.binary_auroc(
        arrays["labels"], arrays["eval_clean_scores"])
    return result


def aggregate(entries):
    result = v.aggregate(entries)
    values = [entries[str(s)]["counterfactual"]["clean_scores_with_synthetic_labels_auroc"]
              for s in v.POLICY["heldout_seeds"]]
    defined = [x for x in values if x is not None]
    result["clean_scores_with_synthetic_labels_auroc"] = {
        "mean_over_defined_seeds": float(np.mean(defined)) if defined else None,
        "defined_seeds": len(defined), "total_seeds": len(values)}
    return result


def targets_match(evaluations, reference):
    for strength in STRENGTHS:
        for seed in v.POLICY["evaluation_seeds"]:
            if evaluations[strength][str(seed)]["target_sha256"] != reference["std015"][str(seed)]["target_sha256"]:
                raise ValueError("Cross-strength/model synthetic target locations differ; no reroll")


def read_phase(root, phase, signature, checkpoint=False):
    fit.read_phase(root, phase, signature, FILES if checkpoint else EVIDENCE)
    progress = json.loads((root/phase/"progress.json").read_bytes())
    if set(progress["evaluations"]) != set(STRENGTHS):
        raise ValueError("Strength inventory differs")
    with np.load(root/phase/"arrays.npz", allow_pickle=False) as z:
        flat = {k: z[k] for k in z.files}
    expected = set()
    for strength, std in STRENGTHS.items():
        entries = progress["evaluations"][strength]
        if set(entries) != {str(s) for s in v.POLICY["evaluation_seeds"]}:
            raise ValueError("Seed inventory differs")
        for seed in v.POLICY["evaluation_seeds"]:
            arrays = v.arrays_for(arrays_for(flat, strength), seed)
            expected.update(f"{strength}__s{seed}__{k}" for k in arrays)
            saved = entries[str(seed)]
            if summarize(arrays) != {k: val for k, val in saved.items() if k != "checks"}:
                raise ValueError("Saved metrics cannot be reproduced")
            checks = saved["checks"]
            if (not checks["state_unchanged"] or checks["state_before_sha256"] != checks["state_after_sha256"]
                    or checks["noise_std"] != std):
                raise ValueError("Evaluation state/noise contract differs")
    if set(flat) != expected:
        raise ValueError("Unexpected evidence array")
    targets_match(progress["evaluations"], progress["evaluations"])
    return progress, flat


def checkpoints(root, arm, signature, identity):
    result, missing = [], False
    for step in POLICY["snapshots"]:
        phase = f"{arm}-step-{step:04d}"
        if not (root/phase).exists():
            missing = True
            continue
        if missing:
            raise ValueError("Checkpoint gap; preserve output")
        record, _ = read_phase(root, phase, signature, True)
        if record["arm"] != arm or record["step"] != step or len(record["losses"]) != step:
            raise ValueError("Arm checkpoint progress differs")
        for i, row in enumerate(record["losses"]):
            if (row["step"] != i+1 or row["seed"] != v.training_seed(i)
                    or row["noise_std"] != STRENGTHS[arm] or not np.isfinite(row["loss"])
                    or row["target_sha256"] != identity["baseline_training_targets"][i]):
                raise ValueError("Training schedule/noise/targets differ")
        if result and record["losses"][:result[-1]["step"]] != result[-1]["losses"]:
            raise ValueError("Training history differs")
        result.append(record)
    return result
