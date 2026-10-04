"""Frozen, descriptive checkpoint comparison; never calibrates or selects a model."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scripts import mpi_efficientad_map_diagnostic as q

fit, p = q.fit, q.p
NAME = "mpi-efficientad-trajectory-v1"
STEPS = (70000, 10000, 30000)  # Final replay must pass before examining earlier models.
CHECKPOINT_FILES = ["model.pt", "reference.npy", "progress.json"]
POLICY = {"execution_steps": list(STEPS), "display_steps": [10000, 30000, 70000],
          "cases": 20, "training_atlas_photos": 64, "regions_per_photo": 12,
          "selection": "same_12_observed_defects_and_8_normals_as_map_diagnostics",
          "earlier_checkpoint_thresholds": None, "automatic_checkpoint_selection": False,
          "raw_channels": ["ST", "AE"], "location_excluded_photos": ["defect/0006"],
          "display": "raw_zero_to_final_normal_q995_per_channel_shared_across_all_cases_and_steps",
          "replay_absolute_tolerance": 1e-4, "training_performed": False,
          "threshold_changed": False, "automatic_pass_allowed": False,
          "independent_performance_claim_allowed": False,
          "resource_plan": {"comparison_seconds": 3600, "rss_GiB": 8,
                            "selection_and_inference_reserved_GiB": 6}, "free_disk_GiB": 5}


def checkpoints(root, records):
    signature = fit.digest(records["plan.json"])
    expected = {r["phase"]: r for r in records["checkpoint-receipts.json"]}
    result = {}
    for step in STEPS:
        phase = f"step-{step:06d}"
        receipt = fit.read_phase(root, phase, signature, CHECKPOINT_FILES)
        if receipt != expected[phase]:
            raise ValueError(f"Historical checkpoint receipt changed: {phase}")
        progress = json.loads((root / phase / "progress.json").read_bytes())
        if progress["step"] != step or [r["step"] for r in progress["losses"]] != list(range(1, step+1)):
            raise ValueError("Historical checkpoint progress changed")
        result[str(step)] = {"receipt": receipt, "monitor_raw_map_means": progress["monitor_raw_map_means"],
                             "last_100_loss_component_means": np.mean(
                                 [r["loss_components"] for r in progress["losses"][-100:]], axis=0).tolist()}
    return result


def raw_summary(raw, masks, eligible):
    raw = p.validate_maps(raw)
    if raw.shape != (12, 2, 256, 256) or len(masks) != 12:
        raise ValueError("Expected exactly 12 original-sized ST/AE maps and masks")
    result = []
    for i, region in enumerate(fit.REGIONS):
        row = {"region": region}
        for channel, name in enumerate(("st", "ae")):
            stats = q.map_stats(raw[i, channel].astype(np.float64), masks[i] if eligible else None)
            # The saved-map diagnostic's signed display limits do not apply to raw maps.
            row[name] = {k: v for k, v in stats.items() if not k.startswith("display_")}
        result.append(row)
    return result


def final_replay(raw, expected, photo, contract):
    p.validate_maps(raw); p.validate_maps(expected)
    if raw.shape != expected.shape or raw.shape != (12, 2, 256, 256):
        raise ValueError("Final replay shape differs")
    delta = float(np.max(np.abs(raw.astype(np.float64)-expected)))
    a = q.e.sample(photo, p.tile_scores(raw, contract["map_quantiles"]), contract)
    b = q.e.sample(photo, p.tile_scores(expected, contract["map_quantiles"]), contract)
    score_delta = max(abs(a["score"]-b["score"]), max(abs(x-y) for x, y in zip(a["tile_scores"], b["tile_scores"], strict=True)))
    if max(delta, score_delta) > POLICY["replay_absolute_tolerance"] or a["alert"] != b["alert"]:
        raise ValueError(f"Final checkpoint replay mismatch: maps={delta}, scores={score_delta}")
    return {"passed": True, "map_maximum_absolute_difference": delta,
            "score_maximum_absolute_difference": score_delta, "decision_changed": False}


def sources(project):
    # New diagnostic dependencies are pinned separately; historical fit/eval identities stay intact.
    files = [*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
             project / "scripts/run_mpi_efficientad_trajectory.ps1",
             project / "docs/mpi-efficientad-trajectory-v1.md"]
    return {f.relative_to(project).as_posix(): hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in sorted(files)}
