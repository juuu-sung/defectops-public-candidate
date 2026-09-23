"""Post-final exploratory PCB score probe on the *development* split only.

The frozen seed-42 checkpoint is reused. No final image is read and no candidate is
selected for deployment or confirmation. Per-image maps are cached for safe resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defectops.padim_experiment import checksum, higher_quantile, records_for_split
from defectops.realiad_followup_development import (
    _candidate_metrics,
    _safe_data_path,
    candidate_image_score,
)
from defectops.realiad_followup_final import (
    EXPECTED_CANDIDATE,
    _gpu_modules,
    _model,
    _predict,
    _read_sample,
    _write_sample,
    validate_gate,
)
from defectops.realiad_followup_map_attribution import map_attribution


FRACTIONS = {
    "top_0_1pct": 0.001,
    "top_0_25pct": 0.0025,
    "top_0_5pct": 0.005,
    "locked_top_1pct": 0.01,
}
SCORE_TOLERANCE = 1e-5


def score_map(anomaly_map: np.ndarray) -> dict[str, float]:
    """Ground-truth-free image scores fixed before rerunning development."""
    return {
        name: candidate_image_score(
            0.0,
            anomaly_map,
            {"image_score": "mean_of_top_map_fraction", "top_fraction": fraction},
        )
        for name, fraction in FRACTIONS.items()
    }


def _reference_rows(gate: dict[str, Any], split: str) -> list[dict[str, Any]]:
    selected = gate["dev_result"]["candidates"][EXPECTED_CANDIDATE]
    key = "calibration_scores" if split == "calibration" else "predictions"
    path = gate["paths"]["development"] / selected["artifacts"][key]
    if checksum(path) != selected["artifacts"][f"{key}_sha256"]:
        raise ValueError(f"Original development {key} checksum differs")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _mask(record: dict[str, Any], data_root: Path, size: int) -> np.ndarray:
    if record["label"] == 0:
        return np.zeros((size, size), dtype=np.bool_)
    path = _safe_data_path(data_root, record["mask_path"])
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Development mask is not PNG: {path}")
        resized = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(resized, dtype=np.uint8) >= 128


def _score_replay(
    records: list[dict[str, Any]], maps: list[np.ndarray], reference: list[dict[str, Any]],
    *, threshold: float | None = None,
) -> float:
    if len(records) != len(maps) or len(records) != len(reference):
        raise ValueError("Original development sample count differs")
    differences = []
    for record, anomaly_map, row in zip(records, maps, reference, strict=True):
        replayed = score_map(anomaly_map)["locked_top_1pct"]
        if row.get("image_path") != record["image_path"] or not math.isfinite(row.get("score", float("nan"))):
            raise ValueError("Original development sample order or score differs")
        differences.append(abs(replayed - row["score"]))
        if threshold is not None:
            expected_decision = "REVIEW_REQUIRED" if replayed >= threshold else "PASS"
            if row.get("decision") != expected_decision or abs(row.get("threshold", float("nan")) - threshold) > SCORE_TOLERANCE:
                raise ValueError("Original development decision differs from replay")
    maximum = max(differences, default=0.0)
    if maximum > SCORE_TOLERANCE:
        raise ValueError(f"Frozen development score replay differs: {maximum}")
    return maximum


def _summarize_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"missed": [], "reviewed": []}
    for row in rows:
        if row["label"]:
            groups["reviewed" if row["baseline_decision"] == "REVIEW_REQUIRED" else "missed"].append(row)
    result = {}
    for name, members in groups.items():
        shares = [
            row["gt_score_share_of_top1pct_min"] for row in members
            if row["gt_score_share_of_top1pct_min"] is not None
        ]
        result[name] = {
            "count": len(members),
            "median_gt_pixels": float(np.median([row["gt_pixel_count"] for row in members])) if members else None,
            "median_gt_top1pct_score_share": float(np.median(shares)) if shares else None,
            "gt_zero_top1pct_count": sum(row["gt_pixels_in_top1pct_max"] == 0 for row in members),
        }
    return result


def human_summary(analysis: dict[str, Any]) -> str:
    """Short, shareable explanation; never designate a post-final winner."""
    lines = [
        "# PCB 개발 데이터 점수 방식 탐색 결과",
        "",
        "이미 관찰한 개발 데이터에서 **기존 모델은 그대로 두고**, 이미지 점수를 모으는 비율만 바꿔 비교했습니다.",
        "기존 최종 시험 이미지는 사용하지 않았으며, 아래는 새 최종 성능이 아닙니다.",
        "",
        "| 점수에 사용한 맵 면적 | 놓친 불량 | 정상 검토 요청 | 기존 미탐을 새로 잡음 | 기존 검출을 새로 놓침 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in analysis["candidates"].items():
        counts = result["metrics"]["fixed_threshold"]["counts"]
        label = "기존 1%" if name == "locked_top_1pct" else f"상위 {result['top_fraction'] * 100:g}%"
        lines.append(
            f"| {label} | {counts['fn']}/{counts['fn'] + counts['tp']} | "
            f"{counts['fp']}/{counts['fp'] + counts['tn']} | "
            f"{result['previous_misses_rescued']} | {result['previous_hits_newly_missed']} |"
        )
    missed = analysis["baseline_map_attribution"]["missed"]
    lines += [
        "",
        f"기존 개발 데이터 미탐 {missed['count']}장 중 정답 픽셀이 상위 1%에 하나도 없는 사례는 "
        f"{missed['gt_zero_top1pct_count']}장입니다. 점수를 모으는 방식만 바꿔서는 이런 국소 신호 문제를 해결하기 어렵습니다.",
        "",
        "**읽는 법:** 놓친 불량 수는 작을수록 좋지만, 정상 검토 요청과 새로 놓친 불량이 "
        "함께 늘지 않았는지도 보셔야 합니다.",
        "이 비교는 최종 결과를 본 뒤 설계한 탐색 실험입니다. 여기서 가장 좋아 보이는 방식도 "
        "새 미관측 시험 전에는 채택하거나 자동 PASS에 사용하지 않습니다.",
        "",
    ]
    return "\n".join(lines)


def compare_scores(
    calibration_maps: list[np.ndarray], development_maps: list[np.ndarray],
    calibration_records: list[dict[str, Any]], development_records: list[dict[str, Any]],
    original_calibration: list[dict[str, Any]], original_predictions: list[dict[str, Any]],
    *, locked_threshold: float, pixel_threshold: float,
    development_masks: list[np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the old score, then compare four predeclared aggregations on development."""
    if len(development_masks) != len(development_records):
        raise ValueError("Development masks differ from the manifest")
    cal_difference = _score_replay(calibration_records, calibration_maps, original_calibration)
    dev_difference = _score_replay(
        development_records, development_maps, original_predictions, threshold=locked_threshold
    )
    cal_scores = [score_map(anomaly_map) for anomaly_map in calibration_maps]
    dev_scores = [score_map(anomaly_map) for anomaly_map in development_maps]
    thresholds = {
        name: higher_quantile([scores[name] for scores in cal_scores], 0.95)
        for name in FRACTIONS
    }
    if abs(thresholds["locked_top_1pct"] - locked_threshold) > SCORE_TOLERANCE:
        raise ValueError("Locked calibration threshold differs from replay")
    baseline_decisions = [row["decision"] for row in original_predictions]
    case_rows = []
    attribution_rows = []
    for record, anomaly_map, mask, scores, baseline in zip(
        development_records, development_maps, development_masks, dev_scores, baseline_decisions, strict=True
    ):
        decisions = {name: "REVIEW_REQUIRED" if value >= thresholds[name] else "PASS" for name, value in scores.items()}
        case_rows.append({
            "image_path": record["image_path"], "label": record["label"],
            "defect_type": record["defect_type"], "scores": scores, "decisions": decisions,
        })
        if record["label"]:
            attr = map_attribution(
                anomaly_map, mask, pixel_threshold=pixel_threshold,
                saved_image_score=scores["locked_top_1pct"],
            )
            attribution_rows.append({"label": record["label"], "baseline_decision": baseline, **attr})
    candidate_results = {}
    for name in FRACTIONS:
        rows = [
            {"score": row["scores"][name], "decision": row["decisions"][name]}
            for row in case_rows
        ]
        metrics = _candidate_metrics(development_records, rows, thresholds[name], 0.95)
        rescued = sum(
            record["label"] == 1 and old == "PASS" and new["decision"] == "REVIEW_REQUIRED"
            for record, old, new in zip(development_records, baseline_decisions, rows, strict=True)
        )
        newly_missed = sum(
            record["label"] == 1 and old == "REVIEW_REQUIRED" and new["decision"] == "PASS"
            for record, old, new in zip(development_records, baseline_decisions, rows, strict=True)
        )
        candidate_results[name] = {
            "top_fraction": FRACTIONS[name], "threshold": thresholds[name],
            "metrics": metrics, "previous_misses_rescued": rescued,
            "previous_hits_newly_missed": newly_missed,
        }
    return {
        "status": "exploratory_postfinal_development_score_probe",
        "data_scope": "realiad_pcb_development_only",
        "model_changed": False, "final_images_used": False,
        "confirmatory_result": False, "automatic_pass_allowed": False,
        "calibration_count": len(calibration_records),
        "development_count": len(development_records),
        "max_calibration_score_replay_difference": cal_difference,
        "max_development_score_replay_difference": dev_difference,
        "baseline_map_attribution": _summarize_attribution(attribution_rows),
        "candidates": candidate_results,
        "limitations": [
            "The development set and final outcomes were already observed; this is hypothesis screening only.",
            "Changing the score cannot fix a weak anomaly signal at the true defect location.",
            "No new scoring rule may be promoted or described as confirmed without a new unseen holdout.",
        ],
    }, case_rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.stem}-",
        suffix=".json", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(project_root: Path, output: Path, handoff: Path) -> dict[str, Any]:
    gate = validate_gate(project_root, "realiad_pcb", 42)
    output = output.resolve()
    handoff = handoff.resolve()
    if (
        output != (project_root / "artifacts" / "realiad-development-score-probe-v1").resolve()
        or handoff != (project_root / "artifacts" / "handoff" / "realiad-development-score-probe-v1.zip").resolve()
    ):
        raise ValueError("Exploratory output paths must match the dedicated, non-final locations")
    if handoff.exists():
        raise FileExistsError("Completed exploratory handoff was not overwritten")
    development = gate["development"]
    calibration_records = records_for_split(development, "calibration")
    development_records = records_for_split(development, "test")
    if any(row.get("protocol_split") != "development" for row in development_records):
        raise ValueError("Only the development split can be rescored")
    selected = gate["dev_result"]["candidates"][EXPECTED_CANDIDATE]
    original_calibration = _reference_rows(gate, "calibration")
    original_predictions = _reference_rows(gate, "development")
    data_root = project_root / "data"
    size = gate["protocol"]["base_model"]["image_size"]
    output.mkdir(parents=True, exist_ok=True)
    gpu = None
    model = None
    all_maps: dict[str, list[np.ndarray]] = {}
    all_masks: dict[str, list[np.ndarray]] = {}
    for split, records in (("calibration", calibration_records), ("development", development_records)):
        samples_dir = output / "samples" / split
        samples_dir.mkdir(parents=True, exist_ok=True)
        maps = []
        masks = []
        for index, record in enumerate(records):
            sample_path = samples_dir / f"sample-{index:04d}.npz"
            expected = {
                "category": "realiad_pcb", "seed": 42, "split": split,
                "development_manifest_sha256": development["manifest_sha256"],
                "checkpoint_sha256": gate["dev_result"]["layer_groups"]["layer1-layer2"]["checkpoint_sha256"],
                "image_path": record["image_path"], "image_sha256": record["image_sha256"],
                "mask_sha256": record.get("mask_sha256"), "label": record["label"],
            }
            expected_mask = _mask(record, data_root, size)
            if sample_path.exists():
                _, anomaly_map, saved_mask = _read_sample(sample_path, expected, size)
                if not np.array_equal(saved_mask, expected_mask):
                    raise ValueError(f"Cached development mask differs: {sample_path}")
            else:
                if gpu is None:
                    gpu = _gpu_modules()
                    model = _model(
                        gate["protocol"], gate["dev_checkpoint"],
                        {"category": "realiad_pcb", "seed": 42,
                         "manifest_sha256": development["manifest_sha256"],
                         "execution_config_sha256": gate["execution"]["config_sha256"]}, gpu,
                    )
                path = _safe_data_path(data_root, record["image_path"])
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.mode != "RGB":
                        raise ValueError(f"Development image is not RGB JPEG: {path}")
                    if record["label"]:
                        with Image.open(_safe_data_path(data_root, record["mask_path"])) as source_mask:
                            if source_mask.size != image.size:
                                raise ValueError(f"Development image/mask size differs: {path}")
                    _, anomaly_map = _predict(model, image, gate["protocol"], gpu)
                _write_sample(sample_path, {**expected, "score": score_map(anomaly_map)["locked_top_1pct"]}, anomaly_map, expected_mask)
            maps.append(anomaly_map)
            masks.append(expected_mask)
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(f"{split} {index + 1}/{len(records)}", flush=True)
        all_maps[split] = maps
        all_masks[split] = masks
    analysis, case_rows = compare_scores(
        all_maps["calibration"], all_maps["development"], calibration_records,
        development_records, original_calibration, original_predictions,
        locked_threshold=selected["threshold"], pixel_threshold=selected["pixel"]["threshold"],
        development_masks=all_masks["development"],
    )
    analysis.update({
        "source_development_manifest_sha256": development["manifest_sha256"],
        "source_checkpoint_sha256": gate["dev_result"]["layer_groups"]["layer1-layer2"]["checkpoint_sha256"],
        "source_original_predictions_sha256": selected["artifacts"]["predictions_sha256"],
        "source_original_calibration_sha256": selected["artifacts"]["calibration_scores_sha256"],
        "cached_map_count": len(calibration_records) + len(development_records),
    })
    cases_path = output / "cases.jsonl"
    if cases_path.exists():
        existing = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
        if existing != case_rows:
            raise ValueError("Existing exploratory cases differ and were not overwritten")
    else:
        with cases_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in case_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    analysis["cases_sha256"] = checksum(cases_path)
    analysis_path = output / "analysis.json"
    if analysis_path.exists():
        if json.loads(analysis_path.read_text(encoding="utf-8")) != analysis:
            raise ValueError("Existing exploratory analysis differs and was not overwritten")
    else:
        _write_json(analysis_path, analysis)
    summary_path = output / "summary.md"
    summary = human_summary(analysis)
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise ValueError("Existing human summary differs and was not overwritten")
    else:
        summary_path.write_text(summary, encoding="utf-8", newline="\n")
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(analysis_path, "analysis.json")
        archive.write(cases_path, "cases.jsonl")
        archive.write(summary_path, "summary.md")
    analysis["handoff_zip_sha256"] = checksum(handoff)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.project_root.resolve(), args.output, args.handoff), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 1
    return 0
