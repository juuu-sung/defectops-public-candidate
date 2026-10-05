"""A single fixed synthetic task; memorization is not defect generalization."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts import mpi_supersimplenet_synthetic_protocol as q

fit, e = q.fit, q.e
NAME = "mpi-supersimplenet-learnability-v1"
POLICY = {
    "updates": 500, "snapshots": [0, 50, 100, 200, 300, 400, 500],
    "selection": "training-batch00-seed42 from frozen synthetic-check ordering",
    "input_tiles": 8, "effective_batch": 16, "seed": 42,
    "initialization": "copy of final 300-epoch fit; new AdamW optimizer",
    "trainable": ["adaptor", "segdec"], "backbone_frozen": True,
    "adaptor_lr": 1e-4, "adaptor_weight_decay": .01,
    "segdec_lr": 2e-4, "segdec_weight_decay": 1e-5,
    "scheduler": None, "amp": False,
    "synthetic": "official v2.2.0; noise_std0.015; perlin0.6; RNG reset before every forward",
    "noise_location": "after current adaptor; no rescaling or reroll",
    "reload_tolerance": 1e-4,
    "resource_plan": {"comparison_seconds": 1800, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "maximum_attempted_updates": 650, "free_disk_GiB": 8,
    "test_images_decoded": 0, "test_masks_decoded": 0,
    "development_or_calibration_weight_updates": 0,
    "threshold_changed": False, "service_model_changed": False,
    "automatic_pass_allowed": False,
    "interpretation": "fixed-task learnability only; no automatic pass, model selection or expansion",
}
FILES = ["model.pt", "arrays.npz", "progress.json"]


def prepare(project, handoffs, fit_root):
    records = e.read_fit(project, handoffs, fit_root)
    plan = records["plan.json"]
    job = q.jobs(plan)[0]
    if job["name"] != "training-batch00-seed42" or len({r["photo_id"] for r in job["rows"]}) != 1:
        raise ValueError("Expected first eight regions of one fixed training normal photo")
    paths = ["scripts/mpi_supersimplenet_learnability_protocol.py",
             "scripts/mpi_supersimplenet_learnability_runtime.py",
             "scripts/run_mpi_supersimplenet_learnability.py",
             "scripts/run_mpi_supersimplenet_learnability.ps1",
             "docs/mpi-supersimplenet-learnability-v1.md",
             "scripts/mpi_supersimplenet_synthetic_runtime.py",
             "scripts/mpi_supersimplenet_synthetic_protocol.py",
             "scripts/mpi_supersimplenet_evaluation_runtime.py",
             "scripts/mpi_supersimplenet_evaluation_protocol.py",
             "scripts/run_mpi_supersimplenet_evaluation.py",
             "scripts/run_mpi_efficientad_probe.py"]
    identity = {"name": NAME, "policy": POLICY, "job": job,
        "fit_zip_sha256": e.FIT_SHA, "fit_signature": fit.digest(plan),
        "checkpoint_sha256": records["checkpoint-receipts.json"][-1]["files"]["model.pt"],
        "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256, "environment": plan["environment"],
        "original_contract_sha256": fit.digest(records["normal-results.json"]["contract"]),
        "sources_lf_sha256": {n: hashlib.sha256((project/n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in paths}}
    return records, identity


def target_digest(arrays):
    q.summarize_arrays(arrays)
    if not arrays["labels"].any() or arrays["labels"].all():
        raise ValueError("Fixed task requires both classes; stop without changing seed")
    return hashlib.sha256(arrays["masks"].tobytes()+arrays["labels"].tobytes()).hexdigest()


def read_checkpoints(root, signature):
    results = []
    missing = False
    for step in POLICY["snapshots"]:
        phase = f"step-{step:04d}"
        if not (root/phase).exists():
            missing = True
            continue
        if missing:
            raise ValueError("Checkpoint gap; preserve outputs")
        fit.read_phase(root, phase, signature, FILES)
        progress = json.loads((root/phase/"progress.json").read_bytes())
        with np.load(root/phase/"arrays.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if (progress["step"] != step or len(progress["losses"]) != step
                or [r["step"] for r in progress["losses"]] != list(range(1, step+1))
                or progress["metrics"] != q.summarize_arrays(arrays)
                or progress["target_sha256"] != target_digest(arrays)
                or (results and (progress["losses"][:results[-1]["step"]] != results[-1]["losses"]
                    or progress["target_sha256"] != results[0]["target_sha256"]))):
            raise ValueError("Checkpoint trajectory or fixed targets changed")
        results.append(progress)
    return results
