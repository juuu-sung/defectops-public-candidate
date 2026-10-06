"""Frozen step-zero controls; elapsed time is recorded, never a stop condition."""
import hashlib
import json

import numpy as np

from scripts import mpi_supersimplenet_parity_protocol as parity

fit = parity.fit
NAME = "mpi-supersimplenet-update-audit-v1"
PRIOR_SHA = "c9f344f58def1df7eaaefecda93b1aa8b36fd6bf443a788fb89a08f2b9fd2518"
ARMS = ("official_a", "official_b", "reference_logits", "reference_author")
PAIRS = {"repeat": ("official_a", "official_b"),
         "same_loss": ("official_a", "reference_logits"),
         "loss_only": ("reference_logits", "reference_author"),
         "author_loss": ("official_a", "reference_author")}
POLICY = {
    "checkpoint_step": 0, "seeds": [42, 900001], "batch_size": 8,
    "arms": list(ARMS), "pairs": PAIRS,
    "selection": "same first eight formal training tiles; no new photos or test",
    "updates_per_arm": 1, "saved_updated_models": 0,
    "maximum_attempted_jobs": 3, "free_disk_GiB": 2,
    "tolerances": parity.POLICY["tolerances"],
    "diagnostic_values_per_parameter": 32,
    "post_update": "raw head maps/scores, clean and same synthetic inputs; eval then train; no public smoothed map",
    "resource_plan": {"comparison_seconds": None, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "test_images_decoded": 0, "test_masks_decoded": 0,
    "threshold_changed": False, "service_model_changed": False,
    "automatic_pass_allowed": False, "later_phases_automatically_started": False,
}


def jobs():
    return [{"name": f"step-000000-seed-{seed}", "step": 0, "seed": seed}
            for seed in POLICY["seeds"]]


def selected_differences(ga, gb, ua, ub):
    """Deterministic union of largest gradient/update errors, with paired raw values."""
    arrays = [np.asarray(x).reshape(-1) for x in (ga, gb, ua, ub)]
    if not arrays[0].size or len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid paired gradient/update evidence")
    count = min(POLICY["diagnostic_values_per_parameter"], arrays[0].size)
    selected = set()
    for a, b in ((arrays[0], arrays[1]), (arrays[2], arrays[3])):
        error = np.abs(a.astype(float)-b.astype(float))
        # Threshold + ascending index tie break, without sorting millions of values.
        cutoff = np.partition(error, error.size-count)[error.size-count]
        ids = np.flatnonzero(error > cutoff).tolist()
        ids += np.flatnonzero(error == cutoff)[:count-len(ids)].tolist()
        selected.update(ids)
    return [{"flat_index": i, "gradient_a": float(arrays[0][i]), "gradient_b": float(arrays[1][i]),
             "update_a": float(arrays[2][i]), "update_b": float(arrays[3][i])} for i in sorted(selected)]


def output_checks(arrays):
    result = {}
    for pair, (a, b) in PAIRS.items():
        result[pair] = {suffix: parity.compare(arrays[a+"_"+suffix], arrays[b+"_"+suffix], "outputs")
                       for suffix in suffixes()}
    return result


def suffixes():
    return ["pre_train_synthetic_"+k for k in ("maps", "scores")] + [
        f"{stage}_{mode}_{kind}_{output}" for stage, modes in (("pre", ("eval",)), ("post", ("eval", "train")))
        for mode in modes for kind in ("clean", "synthetic") for output in ("maps", "scores")]


def verify_arrays(result, arrays):
    expected = {arm+"_"+s for arm in ARMS for s in suffixes()} | {"masks", "labels"}
    if set(arrays) != expected:
        raise ValueError("Unexpected raw output inventory")
    for name in expected-{"masks", "labels"}:
        shape = (16, 64, 64) if name.endswith("_maps") else (16,)
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid output array")
    if arrays["masks"].shape != (16,64,64) or arrays["labels"].shape != (16,):
        raise ValueError("Invalid targets")
    if (arrays["masks"].dtype != bool or arrays["labels"].dtype != bool
            or not np.array_equal(arrays["labels"], arrays["masks"].reshape(16,-1).any(1))):
        raise ValueError("Target mismatch")
    if output_checks(arrays) != result["output_checks"]:
        raise ValueError("Raw output metrics differ")
    for pair in PAIRS:
        checks = list(result["output_checks"][pair].values())
        for kind in ("loss", "gradients", "updates", "buffers"):
            checks += list(result["parameter_checks"][pair][kind].values())
        if result["pair_passed"][pair] != all(c["passed"] for c in checks):
            raise ValueError("Pair verdict differs")


def prepare(project, handoffs, fit_root):
    prior = handoffs / (parity.NAME+".zip")
    if fit.sha256_file(prior) != PRIOR_SHA:
        raise ValueError("Reviewed parity ZIP differs or is missing; preserve it")
    identity = parity.prepare(project, handoffs, fit_root)
    identity.update(name=NAME, policy=POLICY, jobs=jobs(), prior_parity_zip_sha256=PRIOR_SHA)
    for name in ("scripts/run_mpi_supersimplenet_update_audit.ps1", "docs/mpi-supersimplenet-update-audit-v1.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    # Roundtrip makes tuple pairs equal to JSON-decoded plans when resuming.
    return json.loads(fit.canonical(identity))
