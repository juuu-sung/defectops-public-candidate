"""One fixed 256-photo PatchCore experiment; disk-backed exact candidate coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts import mpi_patchcore_scaling_probe as probe
from scripts import mpi_normal_diversity_evaluation_protocol as comparison
from scripts import mpi_supersimplenet_evaluation_protocol as evaluation
from scripts.run_mpi_efficientad_probe import publish, attempts

fit = probe.fit
NAME = "mpi-patchcore-scaling-v1"
PROBE_SHA = "a4665ce927610f63641169be6a3e3749502a4777fa9244df89688dfc87bb1a89"
POLICY = {
    "training_photos": 256, "regions": 12, "patches_per_tile": 1024,
    "feature_dimension": 768, "candidates": 3145728, "representatives": 78643,
    "seed": 42, "projection_eps": .9, "projected_dimension": 369,
    "cuda_matmul_tf32": False, "cudnn_tf32": True, "cudnn_deterministic": True,
    "chunk_rows": 65536, "selection_checkpoint_interval": 1000,
    "initial_free_disk_GiB": 32, "free_gpu_GiB": 6,
    "resource_plan": {"comparison_seconds": None, "rss_GiB": 20,
                      "selection_and_inference_reserved_GiB": 6},
    "time_limit_seconds": None, "reload_absolute_tolerance": 1e-4,
    "score": "top1_percent_mean_then_region_median_Q90_width_then_photo_max",
    "normal_development": 128, "normal_calibration": 169,
    "threshold": "normal calibration photo Q95 higher; >=; never tuned on test",
    "regression12": "descriptive_only_no_early_stop",
    "location_excluded": ["defect/0006"],
    "warmup_photos": 5, "warm_photos": 20, "warm_repeats": 3,
    "stability": {**evaluation.POLICY["acceptance"]["stability"],
                  "refit": "region top1-map mean median/Q90 width and photo Q95; no pixel thresholds"},
    "accuracy_targets": {"preserve_all_53": True, "fp54_max": 2, "fp256_max": 12,
                         "miss_improvement_requires_new_detection": True},
    "resume": "atomic photo/chunk/1000-selection checkpoints; Ctrl+C only; failed or unaccounted attempts require review",
    "full_candidate_policy": "no random prefilter, no automatic shrink, no feature normalization",
    "projection_caveat": "Official eps=.9 changes JL dimension 335 to 369 as N grows. Chunked real-data projection may differ numerically from monolithic GEMM; not claimed bitwise identical full coreset.",
    "automatic_pass_allowed": False, "service_model_changed": False,
    "interpretation": "observed exploratory regression; not independent generalization",
}


def value(root, phase, signature):
    fit.read_phase(root, phase, signature, ["value.json"])
    return json.loads((root/phase/"value.json").read_bytes())


def checked_probe(project, path):
    if fit.sha256_file(path) != PROBE_SHA:
        raise ValueError("Expected the reviewed successful synthetic probe ZIP")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {"handoff-manifest.json"}:
            raise ValueError("Probe inventory differs")
        data = {}
        for name, sha in manifest.items():
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Probe payload differs")
            data[name] = json.loads(raw)
    plan = data["plan.json"]
    if (plan["policy"] != probe.POLICY or not data["parity.json"]["passed"]
            or not data["scale.json"]["passed"] or data["report.json"]["identity"] != fit.digest(plan)):
        raise ValueError("Probe did not pass its frozen recipe")
    for name, sha in plan["sources_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Reviewed probe dependency changed: {name}")
    return data


def prepare(project):
    handoffs = project/"artifacts/handoff"
    probe_data = checked_probe(project, handoffs/(probe.NAME+".zip"))
    data = probe.evidence.read_bundle(handoffs/"mpi-supersimplenet-scaling-v1.zip")
    # This validates the frozen normal/observed-test cohorts and baseline Q90 scores.
    source, _ = probe.evidence.prepare(project, data)
    checked = probe.prepare(project, handoffs/"mpi-supersimplenet-scaling-v1.zip")
    group = data["plan.json"]["expanded_training"]
    if (checked["expanded_training"] != group or group != probe_data["plan.json"]["expanded_training"]):
        raise ValueError("Reviewed nested 256-photo selection changed")
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project/"scripts/run_mpi_patchcore_scaling.ps1",
                    project/"docs/mpi-patchcore-scaling-v1.md"])
    identity = {"name": NAME, "policy": POLICY, "probe_sha256": PROBE_SHA,
        "scaling_fit_sha256": probe.evidence.FIT_SHA, "training": group,
        "source": source["identity"], "cohorts": source["cohorts"],
        "config": source["config"], "environment": probe_data["environment.json"],
        "baseline_checkpoint_sha256": evaluation.inputs.ev.CHECKPOINT_SHA256,
        "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
        "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in paths}}
    return source, identity


def array(path, shape, dtype=np.float32, finite=True):
    result = np.load(path, allow_pickle=False, mmap_mode="r")
    if result.shape != tuple(shape) or result.dtype != np.dtype(dtype):
        raise ValueError(f"Array shape/dtype differs: {path}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"Nonfinite array: {path}")
    return result


def raw_block(paths, start, end, rows_per_photo=12288, dimensions=768):
    """Bounded CPU copy in the exact photo, region, raster-stride order."""
    if not 0 <= start < end <= len(paths)*rows_per_photo:
        raise ValueError("Raw candidate interval out of bounds")
    result = np.empty((end-start, dimensions), dtype=np.float32)
    cursor = start
    while cursor < end:
        photo, offset = divmod(cursor, rows_per_photo)
        count = min(end-cursor, rows_per_photo-offset)
        raw = array(paths[photo], (rows_per_photo, dimensions))
        result[cursor-start:cursor-start+count] = raw[offset:offset+count]
        del raw
        cursor += count
    return result


def validate_selection(progress, minimum, n, maximum):
    ids = progress["selected"]
    if (progress["step"] != len(ids) or not 0 <= len(ids) <= maximum
            or len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < n for i in ids)
            or type(progress["pending_center"]) is not int or not 0 <= progress["pending_center"] < n
            or (ids and progress["pending_center"] != ids[-1])):
        raise ValueError("Selection checkpoint index/state mismatch")
    if minimum.shape != (n, 1) or minimum.dtype != np.float32:
        raise ValueError("Minimum-distance shape/dtype mismatch")
    if ids:
        if not np.isfinite(minimum).all() or np.any(minimum < 0) or np.any(minimum[ids] != 0):
            raise ValueError("Invalid minimum-distance state")
    elif not np.isposinf(minimum).all():
        raise ValueError("Initial distances must all be positive infinity")


def checkpoint_steps(root, signature, n=3145728, maximum=78643, interval=1000):
    expected = [0, *range(interval, maximum, interval), maximum]
    names = {f"step-{i:06d}" for i in expected}
    if any(x.name not in names for x in root.glob("step-*")):
        raise ValueError("Unexpected selection checkpoint")
    steps, previous, prior_indices = [], None, []
    for step in expected:
        name = f"step-{step:06d}"
        if not (root/name).exists():
            if any((root/f"step-{later:06d}").exists() for later in expected if later > step):
                raise ValueError("Selection checkpoint gap")
            break
        receipt = fit.read_phase(root, name, signature, ["minimum.npy", "progress.json"])
        progress = json.loads((root/name/"progress.json").read_bytes())
        minimum = array(root/name/"minimum.npy", (n, 1), finite=False)
        validate_selection(progress, minimum, n, maximum)
        if progress["step"] != step or progress["previous_receipt"] != previous:
            raise ValueError("Selection checkpoint chain differs")
        if progress["selected"][:len(prior_indices)] != prior_indices:
            raise ValueError("Selection checkpoint prefix changed")
        prior_indices = progress["selected"]
        previous = receipt["receipt_sha256"]
        steps.append((name, progress, receipt))
    return steps


def reload_check(first, second, method):
    maximum, flips = 0., 0
    for group in ("development", "calibration"):
        if len(first[group]) != len(second[group]):
            raise ValueError("Reload cohort length differs")
        for a, b in zip(first[group], second[group], strict=True):
            if [a[k] for k in ("key", "photo_id", "label")] != [b[k] for k in ("key", "photo_id", "label")]:
                raise ValueError("Reload identity differs")
            for x, y in zip(a["regions"], b["regions"], strict=True):
                if (x["region"], x["member"]) != (y["region"], y["member"]):
                    raise ValueError("Reload region order differs")
                if not all(np.isfinite(v[k]) for v in (x, y) for k in ("top1", "native_raw")):
                    raise ValueError("Nonfinite reload score")
                maximum = max(maximum, *(abs(x[k]-y[k]) for k in ("top1", "native_raw")))
            sa, sb = (comparison.c.scoring.score(s, method) for s in (a, b))
            maximum = max(maximum, abs(sa["score"]-sb["score"]))
            flips += sa["alert"] != sb["alert"]
    if not np.isfinite(maximum) or maximum > POLICY["reload_absolute_tolerance"] or flips:
        raise ValueError("Reload score or decision mismatch; do not recalibrate")
    return {"passed": True, "maximum_absolute_difference": maximum, "decision_changes": flips,
            "photos": sum(len(first[g]) for g in first), "contract_refitted": False}
