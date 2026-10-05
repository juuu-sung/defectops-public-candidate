"""Preregistered changing synthetic tasks, fixed held-out seeds and clean controls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np

from scripts import mpi_supersimplenet_learnability_protocol as old
from defectops.evaluation_metrics import binary_auroc

fit, q = old.fit, old.q
NAME = "mpi-supersimplenet-varying-synthetic-v1"
FIXED_ZIP_SHA = "cd4c0dad237ccddc28717a9293def9f96eb2b79d36335e89b2610ebba176a5a2"
POLICY = deepcopy(old.POLICY)
POLICY.pop("seed")
POLICY.update({
    "training_seed_start": 10000,
    "evaluation_seeds": [42, 900001, 900002, 900003, 900004],
    "heldout_seeds": [900001, 900002, 900003, 900004],
    "reference_seed": 42,
    "synthetic": "official v2.2.0; noise_std0.015; perlin0.6; seed=10000+zero_based_update",
    "fixed_control": "previous fixed-seed500 model, read-only evaluation on identical evaluation seeds",
    "interpretation": "same training images, held-out synthetic draws only; not unseen-image or real-defect accuracy",
})
EVIDENCE_FILES = ["arrays.npz", "progress.json"]
FILES = ["model.pt"] + EVIDENCE_FILES


def training_seed(index):
    if not isinstance(index, int) or not 0 <= index < POLICY["updates"]:
        raise ValueError("Update index outside locked schedule")
    seed = POLICY["training_seed_start"] + index
    if seed in POLICY["evaluation_seeds"]:
        raise ValueError("Training and evaluation seeds overlap")
    return seed


def arrays_for(flat, seed):
    prefix = f"s{seed}__"
    return {k[len(prefix):]:v for k,v in flat.items() if k.startswith(prefix)}


def summarize(arrays):
    metrics = q.summarize_arrays(arrays)
    masks, labels = arrays["masks"], arrays["labels"]
    mixed, clean = arrays["eval_mixed_maps"], arrays["eval_clean_maps"]
    difference = mixed-clean
    paired = (arrays["eval_mixed_scores"]-arrays["eval_clean_scores"])[labels]
    counterfactual = {
        "clean_map_positive_fraction_at_zero_logit": float(np.mean(clean > 0)),
        "clean_score_positive_fraction_at_zero_logit": float(np.mean(arrays["eval_clean_scores"] > 0)),
        "clean_at_synthetic_locations_auroc": binary_auroc(masks, clean),
        "mean_map_change_at_synthetic_locations": float(difference[masks].mean()) if masks.any() else None,
        "clean_mean_at_synthetic_locations": float(clean[masks].mean()) if masks.any() else None,
        "clean_positive_fraction_at_synthetic_locations": float(np.mean(clean[masks] > 0)) if masks.any() else None,
        "paired_positive_score_rise_fraction": float(np.mean(paired > 0)) if len(paired) else None,
        "same_source_classification_auroc": binary_auroc(
            np.r_[np.ones(labels.sum()),np.zeros(labels.sum())],
            np.r_[arrays["eval_mixed_scores"][labels],arrays["eval_clean_scores"][labels]]) if labels.any() else None,
        "same_position_pixel_auroc": binary_auroc(
            np.r_[np.ones(masks.sum()),np.zeros(masks.sum())],np.r_[mixed[masks],clean[masks]]) if masks.any() else None,
        "note": "Clean-location scores are counterfactual diagnostics, not normal-image accuracy. Zero is not a service threshold.",
    }
    target = hashlib.sha256(masks.tobytes()+labels.tobytes()).hexdigest()
    return {"metrics":metrics,"counterfactual":counterfactual,"target_sha256":target}


def aggregate(evaluations):
    selected = [evaluations[str(s)] for s in POLICY["heldout_seeds"]]
    paths = {
        "eval_classification_auroc": ("metrics","modes","eval_mixed","classification_auroc"),
        "eval_pixel_auroc": ("metrics","modes","eval_mixed","pixel_auroc"),
        "train_classification_auroc": ("metrics","modes","train_mixed","classification_auroc"),
        "train_pixel_auroc": ("metrics","modes","train_mixed","pixel_auroc"),
        "paired_score_delta": ("metrics","eval_synthetic_delta_mean"),
        **{k:("counterfactual",k) for k in ("clean_at_synthetic_locations_auroc",
            "mean_map_change_at_synthetic_locations","same_source_classification_auroc",
            "same_position_pixel_auroc","clean_map_positive_fraction_at_zero_logit")},
    }
    result = {}
    for name,path in paths.items():
        values = []
        for entry in selected:
            value = entry
            for key in path: value = value[key]
            if value is not None: values.append(value)
        result[name] = {"mean_over_defined_seeds":float(np.mean(values)) if values else None,
                        "defined_seeds":len(values),"total_seeds":len(selected)}
    return result


def prepare(project, handoffs, fit_root, fixed_root):
    records, original_identity = old.prepare(project,handoffs,fit_root)
    path = handoffs/(old.NAME+".zip")
    if fit.sha256_file(path) != FIXED_ZIP_SHA:
        raise ValueError("Expected reviewed fixed-task handoff ZIP; preserve it")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(manifest)+1 or set(z.namelist()) != set(manifest)|{"handoff-manifest.json"}:
            raise ValueError("Fixed-task ZIP inventory differs")
        if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in manifest.items()):
            raise ValueError("Fixed-task ZIP payload differs")
        previous = json.loads(z.read("plan.json"))
        for key in ("policy","job","fit_signature","checkpoint_sha256","original_contract_sha256","sources_lf_sha256"):
            if previous[key] != original_identity[key]: raise ValueError(f"Fixed control provenance differs: {key}")
        fixed_signature = fit.digest(previous)
        if old.e.value(fixed_root,"identity",fixed_signature) != previous:
            raise ValueError("Fixed control local identity differs")
        receipt = fit.read_phase(fixed_root,"step-0500",fixed_signature,old.FILES)
        if receipt != json.loads(z.read("step-0500/receipt.json")):
            raise ValueError("Fixed control checkpoint differs from reviewed receipt")
        references = {}
        for step in (0,500):
            with np.load(io.BytesIO(z.read(f"step-{step:04d}/arrays.npz")),allow_pickle=False) as a:
                references[step] = {k:a[k] for k in a.files}
        fixed_progress = json.loads(z.read("step-0500/progress.json"))
    identity = {**original_identity,"name":NAME,"policy":POLICY,
        "fixed_control_zip_sha256":FIXED_ZIP_SHA,"fixed_control_signature":fixed_signature,
        "fixed_control_checkpoint_sha256":receipt["files"]["model.pt"],
        "fixed_control_fingerprints":fixed_progress["fingerprints"]}
    identity["sources_lf_sha256"] = dict(original_identity["sources_lf_sha256"])
    for name in ("scripts/mpi_supersimplenet_varying_protocol.py","scripts/mpi_supersimplenet_varying_runtime.py",
                 "scripts/run_mpi_supersimplenet_varying.py","scripts/run_mpi_supersimplenet_varying.ps1",
                 "docs/mpi-supersimplenet-varying-synthetic-v1.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256((project/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    return records,identity,references


def read_phase(root, phase, signature, checkpoint=False):
    fit.read_phase(root,phase,signature,FILES if checkpoint else EVIDENCE_FILES)
    progress = json.loads((root/phase/"progress.json").read_bytes())
    if set(progress["evaluations"]) != {str(s) for s in POLICY["evaluation_seeds"]}:
        raise ValueError("Evaluation seed inventory differs")
    with np.load(root/phase/"arrays.npz",allow_pickle=False) as z:
        flat = {k:z[k] for k in z.files}
    expected = set()
    for seed in POLICY["evaluation_seeds"]:
        arrays = arrays_for(flat,seed)
        expected.update(f"s{seed}__{key}" for key in arrays)
        saved = progress["evaluations"][str(seed)]
        if summarize(arrays) != {k:v for k,v in saved.items() if k != "checks"}:
            raise ValueError("Evaluation metrics cannot be reproduced")
        checks = saved["checks"]
        if not checks["state_unchanged"] or checks["state_before_sha256"] != checks["state_after_sha256"]:
            raise ValueError("Evaluation changed model state")
    if set(flat) != expected: raise ValueError("Unexpected array payload")
    if checkpoint:
        step = progress["step"]
        if phase != f"step-{step:04d}" or step not in POLICY["snapshots"] or len(progress["losses"]) != step:
            raise ValueError("Checkpoint progress differs")
        for i,row in enumerate(progress["losses"]):
            if row["step"] != i+1 or row["seed"] != training_seed(i) or not np.isfinite(row["loss"]):
                raise ValueError("Training schedule or loss differs")
    return progress,flat


def read_checkpoints(root, signature):
    result = []; missing = False
    for step in POLICY["snapshots"]:
        phase = f"step-{step:04d}"
        if not (root/phase).exists(): missing=True; continue
        if missing: raise ValueError("Checkpoint gap; preserve outputs")
        progress,_ = read_phase(root,phase,signature,True)
        if result and (progress["losses"][:result[-1]["step"]] != result[-1]["losses"]
                       or any(progress["evaluations"][s]["target_sha256"] != result[0]["evaluations"][s]["target_sha256"]
                              for s in progress["evaluations"])):
            raise ValueError("Training history or evaluation targets changed")
        result.append(progress)
    return result
