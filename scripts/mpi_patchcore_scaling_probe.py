"""Frozen 256-photo design and synthetic-only coreset feasibility helpers."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from scripts import mpi_supersimplenet_scaling_evaluation_protocol as evidence

fit = evidence.fit
NAME = "mpi-patchcore-scaling-probe-v1"
POLICY = {
    "purpose": "synthetic_coreset_arithmetic_and_resource_probe_not_model_fit",
    "training_photos_planned": 256, "regions": 12, "patches_per_tile": 1024,
    "feature_dimension": 768, "representatives_planned": 78643,
    "projection_eps": .9, "seed": 42, "chunk_rows": 65536,
    "parity_rows": 2048, "parity_selections": 64,
    "parity_chunks": [127, 512], "parity_seeds": [42, 43],
    "projection_absolute_tolerance": 1e-4, "distance_absolute_tolerance": 1e-5,
    "selection_indices_must_match": True,
    "full_size_warmup_selections": 2, "full_size_measured_selections": 100,
    "time_limit_seconds": None, "free_gpu_GiB": 6, "free_disk_GiB": 2,
    "resource_plan": {"comparison_seconds": None, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "model_training_performed": False, "images_decoded": 0,
    "test_scores_computed": 0, "automatic_pass_allowed": False,
    "service_model_changed": False,
}


def budget(photos=256):
    if photos not in (64, 256):
        raise ValueError("Only the fixed 64/256 designs are registered")
    n, d = photos*12*1024, POLICY["feature_dimension"]
    eps = POLICY["projection_eps"]
    projected = int(4*math.log(n)/(eps**2/2-eps**3/3))
    return {"photos": photos, "candidates": n, "feature_dimension": d,
            "projected_dimension": projected, "raw_float32_bytes": n*d*4,
            "projected_float32_bytes": n*projected*4,
            "representatives": POLICY["representatives_planned"],
            "bank_float32_bytes": POLICY["representatives_planned"]*d*4,
            "minimum_distance_float32_bytes": n*4,
            "one_chunk_raw_float32_bytes": POLICY["chunk_rows"]*d*4,
            "one_chunk_difference_float32_bytes": POLICY["chunk_rows"]*projected*4,
            "not_a_peak_memory_prediction": True}


def slices(count, chunk):
    if count < 1 or chunk < 1:
        raise ValueError("Positive count and chunk required")
    return [(start, min(start+chunk, count)) for start in range(0, count, chunk)]


def validate_indices(a, b):
    if a != b:
        raise ValueError("Chunked selection differs from installed official reference; do not relax tolerance")


def prepare(project: Path, source: Path):
    records = evidence.read_bundle(source)
    suite = records["plan.json"]
    base, expanded = suite["original_cohorts"]["training"], suite["expanded_training"]
    s = evidence.scaling
    s.validate_group(base, 64)
    selected_ids, selected_rgb = s.validate_group(expanded, 256)
    if expanded["photo_ids"][:64] != base["photo_ids"] or expanded["rows"][:768] != base["rows"]:
        raise ValueError("Original 64 are not preserved as the prefix")
    for group, count in (("development", 128), ("calibration", 169)):
        ids, rgb = s.validate_group(suite["original_cohorts"][group], count)
        if selected_ids & ids or selected_rgb & rgb:
            raise ValueError("Expanded cohort overlaps normal development/calibration")
    for name, sha in suite["sources_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Historical fit dependency changed: {name}")
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project/"scripts/run_mpi_patchcore_scaling_probe.ps1",
                    project/"docs/mpi-patchcore-scaling-probe-v1.md"])
    return {"name": NAME, "policy": POLICY, "scaling_fit_zip_sha256": evidence.FIT_SHA,
            "scaling_plan_sha256": fit.digest(suite),
            "expanded_training": expanded, "expanded_training_sha256": fit.digest(expanded),
            "normal_cohorts_sha256": {k: fit.digest(suite["original_cohorts"][k])
                                     for k in ("development", "calibration")},
            "budgets": {str(n): budget(n) for n in (64, 256)},
            "sources_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths},
            "cohort_note": "Reuses reviewed nested 256 including 64; validation/test RGB exclusion is inherited from the pinned fit bundle. No image opened here.",
            "future_fit_status": "not_implemented_not_started",
            "scope": "Synthetic features only; no checkpoint, image, threshold or service access"}
