"""Two preselected checkpoints, independently calibrated; no checkpoint search."""
from __future__ import annotations
import copy
import hashlib
import json
import zipfile
from scripts import mpi_efficientad_evaluation_protocol as e
from scripts import mpi_efficientad_trajectory as t

fit, p = e.fit, e.p
NAME = "mpi-efficientad-checkpoint-evaluation-v1"
STEPS = (10000, 30000)
TRAJECTORY_SHA = "de7246a155a85be264f226fbef2b11a86cfda2b4a13a5a67e0eb67c832cfe21c"
POLICY = {**copy.deepcopy(e.POLICY), "steps": list(STEPS),
          "threshold_refitted_for_main_comparison": True,
          "normal_formula": {k: copy.deepcopy(p.POLICY[k]) for k in (
              "map_quantiles", "map_quantile_method", "map_formula", "negative_normalized_map_policy",
              "tile_score", "region_scale", "photo_score", "threshold", "alert_operator", "minimum_width_exclusive")},
          "calibration": "same_development128_and_calibration169_separately_per_checkpoint",
          "replay": "first_8_fixed_calibration_photos_after_model_reload_each_attempt",
          "failure": "screen_failure_skips_full_for_that_checkpoint_not_the_other",
          "stopping": "only_two_steps_no_more_checkpoint_search_or_automatic_selection",
          "free_disk_GiB": 30, "total_comparison_seconds": 28800,
          "training_performed": False}


def reviewed_trajectory(path, records, checkpoints):
    if fit.sha256_file(path) != TRAJECTORY_SHA:
        raise ValueError("Expected the reviewed trajectory ZIP")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != {*manifest, "handoff-manifest.json"}:
            raise ValueError("Trajectory inventory differs")
        for name, sha in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Trajectory payload differs")
        plan = json.loads(z.read("plan.json"))
        report = json.loads(z.read("report.json"))
    if (plan["policy"] != t.POLICY or plan["checkpoints"] != checkpoints or
            plan["fit_identity"] != fit.digest(records["plan.json"]) or
            report["status"] != "checkpoint_trajectory_completed_exploratory"):
        raise ValueError("Trajectory provenance differs")
    return plan


def prepare(project, handoffs, fit_root):
    records, source, prior = e.prepare(project, handoffs, fit_root)
    checkpoints = t.checkpoints(fit_root, records)
    trajectory = reviewed_trajectory(handoffs / (t.NAME + ".zip"), records, checkpoints)
    for name, sha in trajectory["sources_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Historical trajectory source changed: {name}")
    files = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project / "scripts/run_mpi_efficientad_checkpoint_evaluation.ps1",
                    project / "docs/mpi-efficientad-checkpoint-evaluation-v1.md"])
    identity = {"name": NAME, "policy": POLICY, "fit_zip_sha256": e.FIT_SHA,
                "trajectory_zip_sha256": TRAJECTORY_SHA, "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
                "fit_signature": prior["fit_signature"], "environment": prior["environment"],
                "source": source["identity"], "cohorts": source["cohorts"],
                "checkpoints": {str(s): checkpoints[str(s)] for s in STEPS},
                "sources_lf_sha256": {f.relative_to(project).as_posix(): hashlib.sha256(
                    f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in files}}
    return source, identity
