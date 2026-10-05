"""One initialization intervention, with the reviewed varying015500 control frozen."""
from copy import deepcopy
import hashlib

from scripts import mpi_supersimplenet_noise_protocol as n

v, fit = n.v, n.fit
NAME = "mpi-supersimplenet-head-reset-v1"
POLICY = deepcopy(v.POLICY)
POLICY.update({
    "initialization": "original formal fit; replace segdec only with official constructor on CPU; fresh AdamW",
    "head_initialization_seed": 20261006,
    "fixed_control": "reviewed varying015500 checkpoint, read-only five-seed replay",
    "batchnorm_initialization": "fresh segdec affine parameters and running buffers; backbone/adaptor unchanged",
    "automatic_model_selection": False,
})
EVIDENCE_FILES, FILES = v.EVIDENCE_FILES, v.FILES


def prepare(project, handoffs, fit_root, fixed_root, varying_root):
    records, previous, references = n.prepare(project, handoffs, fit_root, fixed_root, varying_root)
    identity = {**previous, "name": NAME, "policy": POLICY}
    identity["sources_lf_sha256"] = dict(previous["sources_lf_sha256"])
    for name in ("scripts/mpi_supersimplenet_head_reset_protocol.py",
                 "scripts/mpi_supersimplenet_head_reset_runtime.py",
                 "scripts/run_mpi_supersimplenet_head_reset.py",
                 "scripts/run_mpi_supersimplenet_head_reset.ps1",
                 "docs/mpi-supersimplenet-head-reset-v1.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256(
            (project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if len(identity["baseline_training_targets"]) != POLICY["updates"]:
        raise ValueError("Baseline must contain all 500 locked training targets")
    return records, identity, references


def validate_initialization(record, original):
    if record["seed"] != POLICY["head_initialization_seed"] or record["before"] != original:
        raise ValueError("Head initialization provenance differs")
    before, after = record["before"], record["after"]
    if set(before) != set(after):
        raise ValueError("Initialization state inventory differs")
    if any(after[k] != before[k] for k in ("backbone", "adaptor")):
        raise ValueError("Head reset changed backbone or adaptor")
    if any(after[k] == before[k] for k in ("segmentation", "classification")):
        raise ValueError("Both detection heads must be newly initialized")
    if record["batchnorm_modules"] != 2 or not record["batchnorm_fresh"] or not record["optimizer_empty"]:
        raise ValueError("Head reset retained BatchNorm or optimizer state")


def validate_progress(progress, identity, initialization):
    step = progress["step"]
    if step not in POLICY["snapshots"] or len(progress["losses"]) != step:
        raise ValueError("Unexpected reset checkpoint step")
    if progress["fingerprints"]["backbone"] != initialization["before"]["backbone"]:
        raise ValueError("Frozen backbone changed")
    if step == 0 and progress["fingerprints"] != initialization["after"]:
        raise ValueError("Step zero differs from recorded fresh head")
    for i, row in enumerate(progress["losses"]):
        if row["target_sha256"] != identity["baseline_training_targets"][i]:
            raise ValueError("Training synthetic targets differ from baseline; no reroll")


def aggregate(evaluations, flat):
    # Recompute the extended clean-label counterfactual from raw arrays.
    entries = {str(s): n.summarize(v.arrays_for(flat, s)) for s in POLICY["evaluation_seeds"]}
    if any(entries[str(s)]["target_sha256"] != evaluations[str(s)]["target_sha256"]
           for s in POLICY["evaluation_seeds"]):
        raise ValueError("Summary targets differ")
    return n.aggregate(entries)
