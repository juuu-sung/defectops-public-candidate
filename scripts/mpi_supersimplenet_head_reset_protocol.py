"""One initialization intervention, with the reviewed varying015500 control frozen."""
from copy import deepcopy
import hashlib
import json
import math

from scripts import mpi_supersimplenet_noise_protocol as n
from scripts.mpi_supersimplenet_matched_eval_runtime import PUBLIC_VALIDATION, validate_public_differences

v, fit = n.v, n.fit
NAME = "mpi-supersimplenet-head-reset-v2"
PREVIOUS_NAME = "mpi-supersimplenet-head-reset-v1"
PREVIOUS_ERROR = "ValueError: Separated eval path differs from official public forward"
V1_SOURCES = {
    "scripts/mpi_supersimplenet_head_reset_protocol.py": "b9813382dfe830293bcae360efbc0e89698b73bd47584ae2358920dd2febab72",
    "scripts/mpi_supersimplenet_head_reset_runtime.py": "af84756dcc2d4010cdaefe407f402f5e21c83f167e4acb99b6dbd36ad2178fec",
    "scripts/run_mpi_supersimplenet_head_reset.py": "467ef55f6a897083560e8a162609a916dd59d9aa5ea7d6781954b16d0bb807ac",
    "scripts/run_mpi_supersimplenet_head_reset.ps1": "39d3234860a9e693f097c095f95627c711ff3fc7ca0c7840cf88af931e757b17",
    "docs/mpi-supersimplenet-head-reset-v1.md": "baed43d6640c12c04c7caba1b1e9d7556ae5b72545f6f4a305fc959cc4726159",
}
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
    expected_v1 = {**previous, "name": PREVIOUS_NAME, "policy": POLICY,
                   "sources_lf_sha256": {**previous["sources_lf_sha256"], **V1_SOURCES}}
    prior = read_previous_failure(project/"artifacts/analyses"/PREVIOUS_NAME, expected_v1,
                                  records["report.json"]["final_fingerprints"], references["baseline"])
    identity = {**previous, "name": NAME, "policy": POLICY,
                "public_eval_validation": PUBLIC_VALIDATION, "previous_failure": prior}
    identity["sources_lf_sha256"] = dict(previous["sources_lf_sha256"])
    for name in ("scripts/mpi_supersimplenet_head_reset_protocol.py",
                 "scripts/mpi_supersimplenet_head_reset_runtime.py",
                 "scripts/run_mpi_supersimplenet_head_reset.py",
                 "scripts/run_mpi_supersimplenet_head_reset.ps1",
                 "scripts/mpi_supersimplenet_matched_eval_runtime.py",
                 "docs/mpi-supersimplenet-head-reset-v2.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256(
            (project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if len(identity["baseline_training_targets"]) != POLICY["updates"]:
        raise ValueError("Baseline must contain all 500 locked training targets")
    return records, identity, references


def read_previous_failure(root, expected, original, reference):
    """Accept only the reviewed zero-update failure; never waive arbitrary failures."""
    from scripts.run_mpi_supersimplenet_varying import compare_arrays
    previous = json.loads((root/"identity/value.json").read_bytes())
    if {k: val for k, val in previous.items() if k != "runtime_environment"} != expected:
        raise ValueError("Previous failed run provenance differs")
    signature = fit.digest(previous)
    v.old.e.value(root, "identity", signature)
    initialization = v.old.e.value(root, "initialization", signature)
    validate_initialization(initialization, original)
    delta = initialization["original_replay_max_abs_difference"]
    if not math.isfinite(delta) or not 0 <= delta <= POLICY["reload_tolerance"]:
        raise ValueError("Previous original replay did not pass")
    control, arrays = v.read_phase(root, "baseline-control", signature)
    if control["fingerprints"] != expected["baseline_fingerprints"] or control["optimizer_updates"] != 0:
        raise ValueError("Previous read-only baseline differs")
    compare_arrays(arrays, reference)
    if any(root.glob("step-*")) or any((root/n).exists() for n in ("result", "bundle")):
        raise ValueError("Previous run progressed beyond reviewed zero-update failure")
    paths = sorted(root.glob("attempt-*"))
    if len(paths) != 1:
        raise ValueError("Expected one preserved failed attempt; do not delete or rerun")
    attempt = json.loads((paths[0]/"finished.json").read_bytes())
    started = json.loads((paths[0]/"started.json").read_bytes())
    if (started["identity"] != signature or attempt["status"] != "failed"
            or attempt["attempted_updates"] != 0 or attempt["error"] != PREVIOUS_ERROR
            or not math.isfinite(attempt["seconds"]) or attempt["seconds"] <= 0):
        raise ValueError("Not the reviewed pre-training public-path failure")
    files = [f"{phase}/{name}" for phase in ("identity", "initialization") for name in ("value.json", "receipt.json")]
    files += [f"baseline-control/{n}" for n in EVIDENCE_FILES + ["receipt.json"]]
    files += [f"{paths[0].name}/{n}" for n in ("started.json", "finished.json")]
    return {"name": PREVIOUS_NAME, "identity_sha256": signature,
            "seconds": attempt["seconds"], "attempted_updates": 0, "error": PREVIOUS_ERROR,
            "initialization": initialization,
            "files_sha256": {name: fit.sha256_file(root/name) for name in files}}


def check_previous_files(root, previous):
    if any(fit.sha256_file(root/name) != sha for name, sha in previous["files_sha256"].items()):
        raise ValueError("Preserved v1 evidence changed")


def validate_evaluations(evaluations):
    for seed in POLICY["evaluation_seeds"]:
        checks = evaluations[str(seed)]["checks"]
        d = checks["public_batch_comparison"]
        validate_public_differences(d)
        if (d["seed"] != seed or checks["public_eval_map_max_abs_difference"] != d["same_batch_map_max_abs"]
                or checks["public_eval_score_max_abs_difference"] != d["same_batch_score_max_abs"]):
            raise ValueError("Stored matched public-path evidence differs")


def validate_same_initialization(current, previous):
    for key in ("seed", "before", "after", "batchnorm_modules", "batchnorm_fresh", "optimizer_empty"):
        if current[key] != previous[key]:
            raise ValueError("Fresh head differs from preserved v1 initialization; no reroll")


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
    validate_evaluations(progress["evaluations"])
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
