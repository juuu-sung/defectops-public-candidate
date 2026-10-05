"""Bounded implementation comparison; no model selection or fit continuation."""
import hashlib
import json

import numpy as np

from scripts import mpi_supersimplenet_evaluation_protocol as e

fit = e.fit
NAME = "mpi-supersimplenet-implementation-parity-v1"
AUTHOR_COMMIT = "9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301"
POLICY = {
    "steps": [0, 28800], "seeds": [42, 900001], "batch_size": 8,
    "selection": "first8 training rows of frozen formal fit; algebra check, not generalization",
    "optimizer": "one disposable AdamW update per path from identical saved moments and learning rates",
    "synthetic": "one official generator draw on zero features; same additive tensor and targets on both paths",
    "shared_boundary": "frozen backbone; independent adaptor/head/loss; official feature interpolation for parity",
    "author_feature_interpolation": "align_corners=True measured separately, never silently substituted",
    "tolerances": {"outputs": [1e-4, 1e-4], "loss": [1e-5, 1e-4],
                   "gradients": [1e-5, 1e-3], "updates": [1e-6, 1e-3], "buffers": [1e-5, 1e-4]},
    "resource_plan": {"comparison_seconds": 900, "rss_GiB": 20, "selection_and_inference_reserved_GiB": 6},
    "maximum_attempted_jobs": 5, "free_disk_GiB": 2,
    "test_images_decoded": 0, "test_masks_decoded": 0, "threshold_changed": False,
    "service_model_changed": False, "automatic_pass_allowed": False,
    "later_phases_automatically_started": False,
}


def compare(a, b, kind):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
        raise ValueError("Comparison shape/type mismatch")
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Empty/nonfinite comparison")
    atol, rtol = POLICY["tolerances"][kind]
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = x - y
    bad = int(np.count_nonzero(np.abs(delta) > atol + rtol * np.abs(y)))
    norm = float(np.linalg.norm(y.reshape(-1)))
    return {"passed": bad == 0, "shape": list(a.shape), "values": int(a.size),
            "outside_tolerance": bad, "max_abs": float(np.abs(delta).max()),
            "rms_difference": float(np.sqrt(np.mean(delta**2))),
            "relative_l2": float(np.linalg.norm(delta.reshape(-1))/norm) if norm else None,
            "atol": atol, "rtol": rtol}


def jobs():
    return [{"name": f"step-{step:06d}-seed-{seed}", "step": step, "seed": seed}
            for step in POLICY["steps"] for seed in POLICY["seeds"]]


def prepare(project, handoffs, fit_root):
    records = e.read_fit(project, handoffs, fit_root)
    plan = records["plan.json"]
    signature = fit.digest(plan)
    checkpoints = {}
    for step in POLICY["steps"]:
        phase = f"step-{step:06d}"
        receipt = fit.read_phase(fit_root, phase, signature, ["model.pt", "reference.npy", "progress.json"])
        expected = next(r for r in records["checkpoint-receipts.json"] if r["phase"] == phase)
        if receipt != expected:
            raise ValueError("Checkpoint does not match reviewed fit")
        progress = json.loads((fit_root/phase/"progress.json").read_bytes())
        checkpoints[str(step)] = {"sha256": receipt["files"]["model.pt"], "fingerprints": progress["fingerprints"]}
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    paths += [project/"scripts/run_mpi_supersimplenet_parity.ps1", project/"docs/mpi-supersimplenet-implementation-parity-v1.md"]
    return {"name": NAME, "policy": POLICY, "author_commit": AUTHOR_COMMIT,
            "fit_signature": signature, "fit_zip_sha256": e.FIT_SHA,
            "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256, "checkpoints": checkpoints,
            "contract_sha256": fit.digest(records["normal-results.json"]["contract"]),
            "rows": plan["cohorts"]["training"]["rows"][:8], "jobs": jobs(),
            "environment": plan["environment"], "sources_lf_sha256": {
                f.relative_to(project).as_posix(): hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for f in paths}}
