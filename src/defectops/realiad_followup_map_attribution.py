"""Describe locked PCB final maps; never fit, infer, select, or tune a model."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import tempfile
import tomllib
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defectops.padim_experiment import checksum
from defectops.realiad_followup_development import candidate_image_score
from defectops.realiad_followup_final import EXPECTED_SELECTION_SHA256, _archive_member, _read_sample
from defectops.realiad_followup_final_diagnostics import (
    CATEGORY, EXPECTED_CANDIDATE, FINAL_ZIP_SHA256, GROUND_TRUTH_THRESHOLD,
    SEEDS, SHORT, _entry, build_cases, load_final_rows,
)


PRIOR_DIAGNOSTICS_SHA256 = "9b68ded4cfc1dc0a6cd67ede7da3ab4edaeab1ef313570e2d0a6fccccf7905ba"
TOP_FRACTION = 0.01
IMAGE_SIZE = 256
SCORE_TOLERANCE = 1e-5
REVIEWED_GROUPS = {"consensus_false_negative", "consensus_false_positive"}


def validate_prior_diagnostics(prior_zip: Path, cases: list[dict[str, Any]]) -> None:
    if checksum(prior_zip) != PRIOR_DIAGNOSTICS_SHA256:
        raise ValueError("Previously reviewed diagnostics ZIP SHA-256 differs")
    with zipfile.ZipFile(prior_zip) as archive:
        prior_analysis = json.loads(_entry(archive, "analysis.json"))
        prior_cases = [json.loads(line) for line in _entry(archive, "cases.jsonl").splitlines() if line.strip()]
    if (
        prior_analysis.get("status") != "analyzed_postfinal_descriptive"
        or prior_analysis.get("source_handoff_sha256") != FINAL_ZIP_SHA256
        or prior_cases != cases
    ):
        raise ValueError("Previously reviewed cohort differs from the frozen final predictions")


def validate_scoring_contract(handoff_zip: Path) -> None:
    with zipfile.ZipFile(handoff_zip) as archive:
        protocol = tomllib.loads(_entry(archive, "contracts/realiad-followup-preregistration-v1.toml").decode("utf-8"))
    candidate = [item for item in protocol["candidates"] if item["id"] == EXPECTED_CANDIDATE]
    if (
        len(candidate) != 1
        or candidate[0].get("image_score") != "mean_of_top_map_fraction"
        or candidate[0].get("top_fraction") != TOP_FRACTION
        or protocol["base_model"].get("image_size") != IMAGE_SIZE
        or tuple(protocol["base_model"].get("final_seeds", [])) != SEEDS
    ):
        raise ValueError("Frozen final scoring protocol differs")


def map_attribution(
    anomaly_map: np.ndarray, mask: np.ndarray, *, pixel_threshold: float,
    saved_image_score: float,
) -> dict[str, Any]:
    """Fixed descriptive statistics; top-1% GT overlap is bounded under cutoff ties."""
    values = np.asarray(anomaly_map, dtype=np.float32)
    gt = np.asarray(mask, dtype=np.bool_)
    if (
        values.shape != (IMAGE_SIZE, IMAGE_SIZE) or gt.shape != values.shape
        or not np.isfinite(values).all() or not math.isfinite(pixel_threshold)
        or pixel_threshold <= 0 or not math.isfinite(saved_image_score)
    ):
        raise ValueError("Saved final map, mask, or threshold is invalid")
    flat = values.reshape(-1)
    gt_flat = gt.reshape(-1)
    total = flat.size
    k = max(1, math.ceil(total * TOP_FRACTION))
    selected = np.partition(flat, total - k)[-k:]
    computed = candidate_image_score(
        0.0, values, {"image_score": "mean_of_top_map_fraction", "top_fraction": TOP_FRACTION}
    )
    difference = abs(computed - saved_image_score)
    if difference > SCORE_TOLERANCE:
        raise ValueError(f"Saved image score differs from its saved top-1% anomaly map: {difference}")

    cutoff = float(selected.min())
    above = flat > cutoff
    at_cutoff = flat == cutoff
    slots = k - int(above.sum())
    if slots <= 0 or slots > int(at_cutoff.sum()):
        raise ValueError("Invalid top-1% cutoff tie accounting")
    gt_pixels = int(gt_flat.sum())
    result: dict[str, Any] = {
        "image_score_from_saved_map": computed,
        "saved_image_score": saved_image_score,
        "absolute_score_reconstruction_difference": difference,
        "top1pct_pixel_count": k,
        "top1pct_cutoff": cutoff,
        "top1pct_cutoff_tie_count": int(at_cutoff.sum()),
        "top1pct_cutoff_tie_slots": slots,
        "gt_pixel_count": gt_pixels,
        "gt_area_fraction": gt_pixels / total,
        "map_fraction_above_pixel_threshold": float(np.mean(flat >= pixel_threshold)),
        "gt_mean_over_pixel_threshold": None,
        "non_gt_mean_over_pixel_threshold": None,
        "gt_vs_non_gt_rank_auc": None,
        "gt_fraction_above_pixel_threshold": None,
        "non_gt_fraction_above_pixel_threshold": None,
        "gt_pixels_in_top1pct_min": None,
        "gt_pixels_in_top1pct_max": None,
        "gt_recall_in_top1pct_min": None,
        "gt_recall_in_top1pct_max": None,
        "gt_score_share_of_top1pct_min": None,
        "gt_score_share_of_top1pct_max": None,
    }
    non_gt = flat[~gt_flat]
    if len(non_gt):
        result["non_gt_mean_over_pixel_threshold"] = float(np.mean(non_gt, dtype=np.float64) / pixel_threshold)
        result["non_gt_fraction_above_pixel_threshold"] = float(np.mean(non_gt >= pixel_threshold))
    if not gt_pixels or not len(non_gt):
        return result

    gt_values = flat[gt_flat]
    sorted_non_gt = np.sort(non_gt)
    lower = np.searchsorted(sorted_non_gt, gt_values, side="left")
    upper = np.searchsorted(sorted_non_gt, gt_values, side="right")
    rank_auc = float(np.sum(lower + 0.5 * (upper - lower), dtype=np.float64) / (gt_pixels * len(non_gt)))
    gt_above = int(np.count_nonzero(above & gt_flat))
    gt_tied = int(np.count_nonzero(at_cutoff & gt_flat))
    non_gt_tied = int(np.count_nonzero(at_cutoff & ~gt_flat))
    minimum = gt_above + max(0, slots - non_gt_tied)
    maximum = gt_above + min(slots, gt_tied)
    if not 0 <= minimum <= maximum <= min(k, gt_pixels):
        raise ValueError("Invalid ground-truth top-1% overlap bounds")
    strict_gt_sum = float(np.sum(flat[above & gt_flat], dtype=np.float64))
    top_sum = float(np.sum(selected, dtype=np.float64))
    result.update({
        "gt_mean_over_pixel_threshold": float(np.mean(gt_values, dtype=np.float64) / pixel_threshold),
        "gt_vs_non_gt_rank_auc": rank_auc,
        "gt_fraction_above_pixel_threshold": float(np.mean(gt_values >= pixel_threshold)),
        "gt_pixels_in_top1pct_min": minimum,
        "gt_pixels_in_top1pct_max": maximum,
        "gt_recall_in_top1pct_min": minimum / gt_pixels,
        "gt_recall_in_top1pct_max": maximum / gt_pixels,
        "gt_score_share_of_top1pct_min": (
            (strict_gt_sum + cutoff * (minimum - gt_above)) / top_sum if top_sum > 0 else None
        ),
        "gt_score_share_of_top1pct_max": (
            (strict_gt_sum + cutoff * (maximum - gt_above)) / top_sum if top_sum > 0 else None
        ),
    })
    return result


def _original_mask(source: zipfile.ZipFile, record: dict[str, Any]) -> np.ndarray:
    if record["label"] == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.bool_)
    raw = _archive_member(source, record["source_mask_member"], record["mask_sha256"])
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.size != (1024, 1024):
            raise ValueError(f"Unexpected sealed source mask: {record['source_mask_member']}")
        return np.asarray(
            image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.uint8
        ) >= GROUND_TRUTH_THRESHOLD


def _median_image_means(rows: list[dict[str, Any]], metric: str) -> float | None:
    per_image = []
    for case in rows:
        values = [case["seeds"][str(seed)]["map_attribution"].get(metric) for seed in SEEDS]
        valid = [float(value) for value in values if value is not None]
        if valid:
            per_image.append(float(np.mean(valid, dtype=np.float64)))
    return float(np.median(per_image)) if per_image else None


def summarize_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    metrics = [
        "gt_area_fraction", "gt_vs_non_gt_rank_auc", "gt_fraction_above_pixel_threshold",
        "non_gt_fraction_above_pixel_threshold", "gt_recall_in_top1pct_min",
        "gt_recall_in_top1pct_max", "gt_score_share_of_top1pct_min",
        "gt_score_share_of_top1pct_max", "map_fraction_above_pixel_threshold",
    ]
    return {
        group: {
            "image_count": len(group_rows),
            "median_of_image_means": {
                metric: _median_image_means(group_rows, metric) for metric in metrics
            },
        }
        for group, group_rows in sorted(by_group.items())
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, bundles, records = load_final_rows(args.handoff)
    cases = build_cases(records, bundles)
    validate_prior_diagnostics(args.prior_diagnostics, cases)
    validate_scoring_contract(args.handoff)
    if checksum(args.source_zip) != manifest["source_archive_sha256"]:
        raise ValueError("PCB source archive SHA-256 differs from the sealed manifest")
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"Existing map attribution output was not overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    maximum_score_difference = 0.0
    empty_anomaly_masks = 0
    with zipfile.ZipFile(args.source_zip) as source:
        for case, record in zip(cases, records, strict=True):
            expected_mask = _original_mask(source, record)
            if record["label"] == 1 and not expected_mask.any():
                empty_anomaly_masks += 1
            annotated = dict(case)
            annotated["seeds"] = {}
            for seed in SEEDS:
                evaluation = bundles[seed]["evaluation"]
                sample_path = args.samples_root / f"{SHORT}-seed{seed}" / "samples" / f"sample-{case['index']:04d}.npz"
                if checksum(sample_path) != evaluation["sample_artifact_sha256"][case["index"]]:
                    raise ValueError(f"Saved final map file checksum differs: {sample_path}")
                expected = {
                    "selection_report_sha256": EXPECTED_SELECTION_SHA256,
                    "final_manifest_sha256": evaluation["final_manifest_sha256"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "threshold": evaluation["threshold"], "category": CATEGORY, "seed": seed,
                    "source_image_member": record["source_image_member"],
                    "image_sha256": record["image_sha256"], "label": record["label"],
                    "defect_type": record["defect_type"],
                }
                metadata, anomaly_map, saved_mask = _read_sample(sample_path, expected, IMAGE_SIZE)
                if (
                    metadata["score"] != case["seeds"][str(seed)]["score"]
                    or not np.array_equal(saved_mask, expected_mask)
                ):
                    raise ValueError(f"Saved map, mask, or score differs from the final manifest: {sample_path}")
                measure = map_attribution(
                    anomaly_map, saved_mask,
                    pixel_threshold=evaluation["pixel_threshold"], saved_image_score=metadata["score"],
                )
                maximum_score_difference = max(
                    maximum_score_difference, measure["absolute_score_reconstruction_difference"]
                )
                annotated["seeds"][str(seed)] = {
                    **case["seeds"][str(seed)], "pixel_threshold": evaluation["pixel_threshold"],
                    "map_attribution": measure,
                }
            rows.append(annotated)
            if (case["index"] + 1) % 50 == 0 or case["index"] + 1 == len(cases):
                print(f"verified saved maps {case['index'] + 1}/{len(cases)}", flush=True)
    analysis = {
        "schema_version": 1,
        "status": "analyzed_postfinal_map_attribution_descriptive",
        "source_final_handoff_sha256": FINAL_ZIP_SHA256,
        "source_prior_diagnostics_sha256": PRIOR_DIAGNOSTICS_SHA256,
        "final_manifest_sha256": manifest["manifest_sha256"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "category": CATEGORY, "candidate_id": EXPECTED_CANDIDATE,
        "seeds": list(SEEDS), "image_count": len(rows), "map_count": len(rows) * len(SEEDS),
        "top_fraction": TOP_FRACTION, "top1pct_pixel_count": math.ceil(IMAGE_SIZE**2 * TOP_FRACTION),
        "empty_anomaly_mask_count": empty_anomaly_masks,
        "maximum_absolute_score_reconstruction_difference": maximum_score_difference,
        "by_consensus_group": summarize_attribution(rows),
        "model_changed": False, "thresholds_changed": False, "automatic_pass_allowed": False,
        "interpretation_limits": [
            "This reuses observed final maps for descriptive analysis; it is not a new final evaluation.",
            "Per-image GT-versus-non-GT rank AUC describes pixel ranking, not image-level PASS safety.",
            "Top-1% ground-truth overlap and score share are bounded when scores tie at the cutoff.",
            "Three seeds from one image are correlated; group medians use one mean per image.",
            "Do not tune model, threshold, or candidate on these final images and report a confirmatory result.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (stage / "cases.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (stage / "reviewed_cases.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in rows if row["group"] in REVIEWED_GROUPS
            ),
            encoding="utf-8",
        )
        os.replace(stage, output)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--prior-diagnostics", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
