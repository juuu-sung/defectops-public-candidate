"""Analyze fixed PaDiM test errors without changing the model or thresholds."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from defectops.padim_experiment import checksum, git_state, read_json, write_json
from defectops.padim_inspection import colorize_anomaly_values, validate_inference_bundles


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    """Return stable descriptive statistics for a finite value list."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("Descriptive values must be finite")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75, method="linear")),
        "maximum": float(np.max(array)),
    }


def confusion_group(label: int, decision: str) -> str:
    """Map one fixed-threshold result to TP, FP, TN or FN."""
    if label not in (0, 1) or decision not in ("PASS", "REVIEW_REQUIRED"):
        raise ValueError("Invalid label or decision")
    predicted = decision == "REVIEW_REQUIRED"
    return ("T" if predicted == bool(label) else "F") + ("P" if predicted else "N")


def spatial_metrics(
    anomaly_map: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Compute exploratory fixed-threshold pixel overlap for one image."""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.bool_)
    if anomaly_map.shape != ground_truth.shape or anomaly_map.ndim != 2:
        raise ValueError("Anomaly map and ground truth must have the same 2D shape")
    if not np.isfinite(anomaly_map).all() or not math.isfinite(threshold):
        raise ValueError("Anomaly map and threshold must be finite")
    predicted = anomaly_map >= threshold
    intersection = int(np.count_nonzero(predicted & ground_truth))
    predicted_count = int(np.count_nonzero(predicted))
    ground_truth_count = int(np.count_nonzero(ground_truth))
    union = int(np.count_nonzero(predicted | ground_truth))
    return {
        "predicted_pixels": predicted_count,
        "ground_truth_pixels": ground_truth_count,
        "intersection_pixels": intersection,
        "iou": intersection / union if union else None,
        "dice": (
            2.0 * intersection / (predicted_count + ground_truth_count)
            if predicted_count + ground_truth_count
            else None
        ),
        "pixel_precision": intersection / predicted_count if predicted_count else None,
        "pixel_recall": intersection / ground_truth_count if ground_truth_count else None,
    }


def _safe_existing_child(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Artifact path must be a non-empty string")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / relative_path).resolve(strict=True)
    path.relative_to(resolved_root)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {line_number}: {path}")
            rows.append(row)
    return rows


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _draw_exclusive_box(draw: ImageDraw.ImageDraw, box: list[int] | None, color: str) -> None:
    if box is None:
        return
    x_min, y_min, x_max, y_max = box
    draw.rectangle(
        [x_min, y_min, max(x_min, x_max - 1), max(y_min, y_max - 1)],
        outline=color,
        width=2,
    )


def _case_panel(
    source_path: Path,
    anomaly_map: np.ndarray,
    ground_truth: np.ndarray,
    pixel_threshold: float,
    case: dict[str, Any],
) -> Image.Image:
    with Image.open(source_path) as image:
        original = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    normalized = np.clip(anomaly_map / pixel_threshold, 0.0, 1.0)
    heatmap = colorize_anomaly_values(normalized)
    original_array = np.asarray(original, dtype=np.uint8)
    alpha = (0.15 + 0.45 * normalized)[..., None]
    overlay_array = np.rint(original_array * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
    overlay = Image.fromarray(overlay_array)
    predicted_mask = anomaly_map >= pixel_threshold
    overlay_draw = ImageDraw.Draw(overlay)
    _draw_exclusive_box(overlay_draw, _bbox(predicted_mask), "red")
    _draw_exclusive_box(overlay_draw, _bbox(ground_truth), "lime")

    panel = Image.new("RGB", (512, 292), "white")
    panel.paste(original, (0, 36))
    panel.paste(overlay, (256, 36))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    title = f"{case['group']} {case['image_path']}"
    detail = f"score={case['raw_score']:.4f} margin={case['score_margin']:+.4f}"
    draw.text((4, 4), title, fill="black", font=font)
    draw.text((4, 19), detail, fill="black", font=font)
    draw.text(
        (4, 278),
        "left: input | right: fixed map | red: pred | green: GT",
        fill="black",
        font=font,
    )
    return panel


def _write_contact_sheet(
    cases: list[dict[str, Any]],
    row_indices: dict[str, int],
    maps: np.ndarray,
    masks: np.ndarray,
    pixel_threshold: float,
    data_root: Path,
    manifest_records: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    columns = 2
    panel_width, panel_height = 512, 292
    rows = max(1, math.ceil(len(cases) / columns))
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), "#dddddd")
    for case_index, case in enumerate(cases):
        image_path = case["image_path"]
        source_path = _safe_existing_child(data_root, image_path)
        expected_sha256 = manifest_records[image_path].get("image_sha256")
        if checksum(source_path) != expected_sha256:
            raise ValueError(f"Source image checksum differs from manifest: {image_path}")
        map_index = row_indices[image_path]
        panel = _case_panel(
            source_path,
            maps[map_index],
            masks[map_index],
            pixel_threshold,
            case,
        )
        x = (case_index % columns) * panel_width
        y = (case_index // columns) * panel_height
        sheet.paste(panel, (x, y))
    sheet.save(output_path, format="PNG")


def _markdown_summary(analysis: dict[str, Any]) -> str:
    counts = analysis["confusion_counts"]
    lines = [
        f"# {analysis['analysis_id']}",
        "",
        "이 문서는 고정 test 결과의 탐색적 오류 분석이다. "
        "모델과 임곗값은 변경하지 않았다.",
        "같은 test에 맞춘 변경을 새 확증 성능으로 보고하면 안 된다.",
        "",
        "## 혼동행렬",
        "",
        "| TP | FP | TN | FN |",
        "|---:|---:|---:|---:|",
        f"| {counts['tp']} | {counts['fp']} | {counts['tn']} | {counts['fn']} |",
        "",
        "## 결함 유형별 recall과 픽셀 겹침",
        "",
        "| 결함 유형 | 수 | 탐지 | Recall | 평균 IoU | 평균 pixel recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in analysis["by_defect_type"].items():
        lines.append(
            f"| {name} | {result['count']} | {result['reviewed']} | "
            f"{result['recall']:.4f} | {result['mean_iou']:.4f} | "
            f"{result['mean_pixel_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 산출물",
            "",
            "- `cases.jsonl`: 100장별 고정 판정·점수 margin·탐색적 픽셀 겹침",
            "- `false-negatives.png`: FN contact sheet",
            "- `false-positives.png`: FP contact sheet",
            "- `analysis.json`: 구조화 요약과 추적 정보",
            "",
        ]
    )
    return "\n".join(lines)


def run_error_analysis(args: argparse.Namespace) -> dict[str, Any]:
    """Validate one evaluation bundle and create an immutable exploratory analysis."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts").resolve()
    fit_root = (artifacts_root / "runs").resolve()
    evaluation_root = (artifacts_root / "evaluations").resolve()
    analysis_root = (artifacts_root / "analyses").resolve()
    data_root = args.root.resolve(strict=True)
    fit_path = args.fit_run.resolve(strict=True)
    evaluation_path = args.evaluation.resolve(strict=True)
    output_path = args.output.resolve()
    fit_path.relative_to(fit_root)
    evaluation_path.relative_to(evaluation_root)
    output_path.relative_to(analysis_root)
    if output_path.exists():
        raise FileExistsError(f"Analysis output already exists: {output_path}")

    analyzer_git = git_state(project_root)
    fit_run, _, calibration_contract, _ = validate_inference_bundles(
        fit_path, evaluation_path, project_root
    )
    evaluation = read_json(_safe_existing_child(evaluation_path, "evaluation.json"))
    manifest = read_json(_safe_existing_child(evaluation_path, "manifest.json"))
    manifest_records = {
        record["image_path"]: record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("image_path"), str)
    }
    artifacts = evaluation.get("artifacts", {})
    metrics_path = _safe_existing_child(evaluation_path, artifacts.get("metrics"))
    predictions_path = _safe_existing_child(evaluation_path, artifacts.get("predictions"))
    maps_path = _safe_existing_child(evaluation_path, artifacts.get("test_maps"))
    for path, expected in (
        (metrics_path, artifacts.get("metrics_sha256")),
        (predictions_path, artifacts.get("predictions_sha256")),
        (maps_path, artifacts.get("test_maps_sha256")),
    ):
        if checksum(path) != expected:
            raise ValueError(f"Evaluation artifact checksum differs: {path.name}")
    metrics = read_json(metrics_path)
    if metrics != evaluation.get("metrics"):
        raise ValueError("metrics.json differs from evaluation.json")
    prediction_rows = _read_jsonl(predictions_path)
    expected_paths = evaluation.get("test_sample_ids")
    if [row.get("image_path") for row in prediction_rows] != expected_paths:
        raise ValueError("Prediction rows differ from frozen test sample IDs")

    with np.load(maps_path, allow_pickle=False) as saved:
        map_paths = saved["image_paths"].tolist()
        maps = saved["anomaly_maps"].astype(np.float32, copy=True)
        masks = saved["ground_truth_masks"].astype(np.bool_, copy=True)
    if map_paths != expected_paths or maps.shape != masks.shape or maps.ndim != 3:
        raise ValueError("Saved test maps differ from frozen evaluation rows")
    if not np.isfinite(maps).all():
        raise ValueError("Saved test maps contain non-finite values")

    image_threshold = float(calibration_contract["image"]["threshold"])
    pixel_threshold = float(calibration_contract["pixel"]["threshold"])
    if pixel_threshold <= 0:
        raise ValueError("Pixel threshold must be positive for fixed-scale contact sheets")
    cases = []
    for index, row in enumerate(prediction_rows):
        label = row.get("label")
        score = float(row.get("raw_score"))
        decision = row.get("decision")
        expected_decision = "REVIEW_REQUIRED" if score >= image_threshold else "PASS"
        if decision != expected_decision or float(row.get("image_threshold")) != image_threshold:
            raise ValueError(f"Frozen decision contract mismatch: {row.get('image_path')}")
        group = confusion_group(label, decision)
        spatial = spatial_metrics(maps[index], masks[index], pixel_threshold)
        cases.append(
            {
                "image_path": row["image_path"],
                "label": label,
                "defect_type": row["defect_type"],
                "group": group,
                "decision": decision,
                "raw_score": score,
                "image_threshold": image_threshold,
                "score_margin": score - image_threshold,
                "spatial": spatial,
            }
        )

    grouped = {
        name: [case for case in cases if case["group"] == name]
        for name in ("TP", "FP", "TN", "FN")
    }
    expected_counts = metrics["image"]["fixed_threshold"]["counts"]
    counts = {name.lower(): len(grouped[name]) for name in ("TP", "FP", "TN", "FN")}
    if counts != expected_counts:
        raise ValueError("Recomputed confusion counts differ from evaluation metrics")

    by_defect_type = {}
    defect_types = sorted({case["defect_type"] for case in cases if case["label"] == 1})
    for defect_type in defect_types:
        selected = [case for case in cases if case["defect_type"] == defect_type]
        reviewed = sum(case["decision"] == "REVIEW_REQUIRED" for case in selected)
        by_defect_type[defect_type] = {
            "count": len(selected),
            "reviewed": reviewed,
            "missed": len(selected) - reviewed,
            "recall": reviewed / len(selected),
            "mean_iou": float(np.mean([case["spatial"]["iou"] for case in selected])),
            "mean_pixel_recall": float(
                np.mean([case["spatial"]["pixel_recall"] for case in selected])
            ),
        }

    output_path.mkdir(parents=True, exist_ok=False)
    row_indices = {path: index for index, path in enumerate(expected_paths)}
    false_negatives_path = output_path / "false-negatives.png"
    false_positives_path = output_path / "false-positives.png"
    _write_contact_sheet(
        grouped["FN"],
        row_indices,
        maps,
        masks,
        pixel_threshold,
        data_root,
        manifest_records,
        false_negatives_path,
    )
    _write_contact_sheet(
        grouped["FP"],
        row_indices,
        maps,
        masks,
        pixel_threshold,
        data_root,
        manifest_records,
        false_positives_path,
    )
    cases_path = output_path / "cases.jsonl"
    with cases_path.open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    analysis = {
        "schema_version": 1,
        "status": "analyzed_exploratory",
        "analysis_id": output_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-evaluation error analysis only",
        "threshold_changed": False,
        "model_changed": False,
        "eligible_as_confirmatory_evaluation": False,
        "confusion_counts": counts,
        "score_distributions": {
            "normal": describe_values([case["raw_score"] for case in cases if case["label"] == 0]),
            "anomaly": describe_values([case["raw_score"] for case in cases if case["label"] == 1]),
            "false_negative_margin": describe_values(
                [case["score_margin"] for case in grouped["FN"]]
            ),
            "false_positive_margin": describe_values(
                [case["score_margin"] for case in grouped["FP"]]
            ),
        },
        "by_defect_type": by_defect_type,
        "closest_false_negatives": sorted(
            grouped["FN"], key=lambda case: case["raw_score"], reverse=True
        )[:10],
        "closest_false_positives": sorted(
            grouped["FP"], key=lambda case: case["raw_score"]
        )[:10],
        "trace": {
            "analyzer_git": analyzer_git,
            "evaluation_id": evaluation["evaluation_id"],
            "fit_run_id": fit_run["run_id"],
            "manifest_sha256": evaluation["manifest_sha256"],
            "checkpoint_sha256": evaluation["fit"]["checkpoint_sha256"],
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
        },
        "artifacts": {
            "cases": cases_path.name,
            "cases_sha256": checksum(cases_path),
            "false_negatives": false_negatives_path.name,
            "false_negatives_sha256": checksum(false_negatives_path),
            "false_positives": false_positives_path.name,
            "false_positives_sha256": checksum(false_positives_path),
        },
        "limitations": [
            "The official test set has already been observed and this analysis is exploratory.",
            "No model, preprocessing or threshold changes are justified as "
            "confirmatory by this output.",
            "Contact-sheet maps use the frozen pixel threshold and no per-image min-max scaling.",
        ],
    }
    summary_path = output_path / "analysis.md"
    summary_path.write_text(_markdown_summary(analysis), encoding="utf-8")
    analysis["artifacts"]["summary"] = summary_path.name
    analysis["artifacts"]["summary_sha256"] = checksum(summary_path)
    write_json(output_path / "analysis.json", analysis)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset parent directory")
    parser.add_argument("--fit-run", type=Path, required=True, help="Completed fit directory")
    parser.add_argument(
        "--evaluation", type=Path, required=True, help="Completed evaluation directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "analyses" / "padim-error-analysis-v1",
        help="New directory below artifacts/analyses",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_error_analysis(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_id": result["analysis_id"],
                "confusion_counts": result["confusion_counts"],
                "by_defect_type": result["by_defect_type"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
