"""Analyze persistent and seed-unstable Real-IAD PatchCore decisions without tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from defectops.manifest import canonical
from defectops.padim_experiment import checksum, git_state, read_json, write_json
from defectops.padim_inspection import colorize_anomaly_values
from defectops.patchcore_experiment import EXPECTED_REALIAD_STATUS


SEEDS = (42, 43, 44)
CATEGORIES = {
    "realiad_transistor1": {
        "slug": "realiad-transistor1",
        "manifest_sha256": "416467a31a0c4b717adf7b67a0c684ed71eecdfc168fa9caa53c387ce82c95c9",
    },
    "realiad_plastic_nut": {
        "slug": "realiad-plastic-nut",
        "manifest_sha256": "c8eb02c9be8cfc02e9826fc30b0ea9e8a2af814616dddb20d9b5f0c41dea7f27",
    },
}
MAX_SHEET_CASES = 20


def _safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact path must be a non-empty string")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / relative).resolve(strict=True)
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


def _verify_manifest_hash(manifest: dict[str, Any], expected: str) -> None:
    payload = dict(manifest)
    stored = payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical(payload)).hexdigest()
    if stored != expected or actual != expected:
        raise ValueError("Real-IAD manifest differs from the preregistered inventory")


def _load_bundle(
    evaluation_path: Path,
    category: str,
    seed: int,
) -> dict[str, Any]:
    evaluation = read_json(_safe_child(evaluation_path, "evaluation.json"))
    metrics_path = _safe_child(evaluation_path, "metrics.json")
    metrics = read_json(metrics_path)
    fit_run = read_json(_safe_child(evaluation_path, "fit-run.json"))
    manifest = read_json(_safe_child(evaluation_path, "manifest.json"))
    calibration_path = _safe_child(evaluation_path, "calibration-contract.json")
    calibration = read_json(calibration_path)
    artifacts = evaluation.get("artifacts", {})
    fit_reference = evaluation.get("fit", {})
    fit_artifacts = fit_run.get("artifacts", {})
    predictions_path = _safe_child(evaluation_path, artifacts.get("predictions"))
    maps_path = _safe_child(evaluation_path, artifacts.get("test_maps"))

    expected_slug = CATEGORIES[category]["slug"]
    expected_id = f"patchcore-{expected_slug}-seed{seed}-eval-v1"
    checks = {
        "evaluation_id": evaluation.get("evaluation_id") == expected_id
        and evaluation_path.name == expected_id,
        "evaluation_status": evaluation.get("status") == "evaluated_preregistered"
        and evaluation.get("test_evaluated") is True,
        "comparison_status": evaluation.get("comparison_status") == EXPECTED_REALIAD_STATUS,
        "fit_status": fit_run.get("status") == "fit_calibrated_preregistered"
        and fit_run.get("test_evaluated") is False,
        "fit_reference": fit_reference.get("run_id") == fit_run.get("run_id")
        and fit_reference.get("git_commit") == fit_run.get("git", {}).get("commit")
        and fit_reference.get("model_seed") == seed
        and fit_reference.get("checkpoint_sha256") == fit_artifacts.get("checkpoint_sha256"),
        "git_trace": isinstance(fit_run.get("git", {}).get("commit"), str)
        and len(fit_run["git"]["commit"]) == 40
        and isinstance(evaluation.get("evaluator_git", {}).get("commit"), str)
        and len(evaluation["evaluator_git"]["commit"]) == 40,
        "model": fit_run.get("config", {}).get("model") == "patchcore",
        "category": fit_run.get("config", {}).get("model_seed") == seed
        and manifest.get("categories") == [category],
        "manifest_reference": evaluation.get("manifest_sha256")
        == fit_run.get("manifest_sha256")
        == CATEGORIES[category]["manifest_sha256"],
        "metrics_snapshot": metrics == evaluation.get("metrics"),
        "metrics_checksum": checksum(metrics_path) == artifacts.get("metrics_sha256"),
        "predictions_checksum": checksum(predictions_path)
        == artifacts.get("predictions_sha256"),
        "maps_checksum": checksum(maps_path) == artifacts.get("test_maps_sha256"),
        "calibration_checksum": checksum(calibration_path)
        == artifacts.get("calibration_contract_sha256"),
        "calibration_snapshot": calibration == evaluation.get("calibration_contract"),
        "reload": calibration.get("model_reload_maximum_absolute_score_difference") == 0.0,
        "failures": evaluation.get("counts", {}).get("failed") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid Real-IAD evaluation {expected_id}: {', '.join(failed)}")
    _verify_manifest_hash(manifest, CATEGORIES[category]["manifest_sha256"])

    rows = _read_jsonl(predictions_path)
    sample_ids = evaluation.get("test_sample_ids")
    if (
        not isinstance(sample_ids, list)
        or [row.get("image_path") for row in rows] != sample_ids
        or len(sample_ids) != len(set(sample_ids))
    ):
        raise ValueError(f"Prediction rows differ from frozen sample IDs: {expected_id}")
    manifest_by_path = {
        row["image_path"]: row
        for row in manifest.get("records", [])
        if row.get("split") == "test"
    }
    image_threshold = float(calibration["image"]["threshold"])
    pixel_threshold = float(calibration["pixel"]["threshold"])
    if (
        not math.isfinite(image_threshold)
        or not math.isfinite(pixel_threshold)
        or image_threshold <= 0
        or pixel_threshold <= 0
    ):
        raise ValueError(f"Non-finite threshold in {expected_id}")
    for row in rows:
        image_path = row.get("image_path")
        frozen = manifest_by_path.get(image_path)
        if frozen is None or row.get("label") != frozen.get("label") or row.get(
            "defect_type"
        ) != frozen.get("defect_type"):
            raise ValueError(f"Prediction label differs from manifest: {image_path}")
        score = float(row.get("raw_score"))
        expected_decision = "REVIEW_REQUIRED" if score >= image_threshold else "PASS"
        if (
            not math.isfinite(score)
            or row.get("decision") != expected_decision
            or float(row.get("image_threshold")) != image_threshold
        ):
            raise ValueError(f"Prediction decision differs from frozen threshold: {image_path}")
    labels = [int(row["label"]) for row in rows]
    decisions = [row["decision"] == "REVIEW_REQUIRED" for row in rows]
    counts = {
        "tp": sum(label == 1 and decision for label, decision in zip(labels, decisions)),
        "fp": sum(label == 0 and decision for label, decision in zip(labels, decisions)),
        "tn": sum(label == 0 and not decision for label, decision in zip(labels, decisions)),
        "fn": sum(label == 1 and not decision for label, decision in zip(labels, decisions)),
    }
    expected_counts = metrics.get("image", {}).get("fixed_threshold", {}).get("counts")
    evaluation_counts = evaluation.get("counts", {})
    if (
        counts != expected_counts
        or evaluation_counts.get("test") != len(rows)
        or evaluation_counts.get("normal") != labels.count(0)
        or evaluation_counts.get("anomaly") != labels.count(1)
    ):
        raise ValueError(f"Recomputed image counts differ from evaluation: {expected_id}")
    return {
        "seed": seed,
        "evaluation": evaluation,
        "fit_run": fit_run,
        "manifest": manifest,
        "manifest_by_path": manifest_by_path,
        "rows": rows,
        "sample_ids": sample_ids,
        "maps_path": maps_path,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
    }


def build_consensus_cases(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine three frozen seed decisions for one category."""
    if [bundle.get("seed") for bundle in bundles] != list(SEEDS):
        raise ValueError("Bundles must be ordered seeds 42, 43 and 44")
    expected_paths = bundles[0]["sample_ids"]
    if any(bundle["sample_ids"] != expected_paths for bundle in bundles[1:]):
        raise ValueError("Seed evaluations use different test sample order")
    cases = []
    for index, image_path in enumerate(expected_paths):
        rows = [bundle["rows"][index] for bundle in bundles]
        identity = {(row["label"], row["defect_type"]) for row in rows}
        if len(identity) != 1:
            raise ValueError(f"Seed labels differ: {image_path}")
        label, defect_type = identity.pop()
        reviewed = [row["decision"] == "REVIEW_REQUIRED" for row in rows]
        votes = sum(reviewed)
        scores = [float(row["raw_score"]) for row in rows]
        thresholds = [bundle["image_threshold"] for bundle in bundles]
        margin_ratios = [
            (score - threshold) / threshold
            for score, threshold in zip(scores, thresholds, strict=True)
        ]
        if label == 1:
            group = (
                "consensus_detected"
                if votes == 3
                else "consensus_missed" if votes == 0 else "unstable_anomaly"
            )
        else:
            group = (
                "consensus_normal"
                if votes == 0
                else "consensus_reviewed_normal" if votes == 3 else "unstable_normal"
            )
        cases.append(
            {
                "index": index,
                "image_path": image_path,
                "label": label,
                "defect_type": defect_type,
                "group": group,
                "review_votes": votes,
                "seed_decisions": {str(seed): decision for seed, decision in zip(SEEDS, reviewed)},
                "seed_raw_scores": {str(seed): score for seed, score in zip(SEEDS, scores)},
                "seed_thresholds": {
                    str(seed): threshold for seed, threshold in zip(SEEDS, thresholds)
                },
                "seed_margin_ratios": {
                    str(seed): margin for seed, margin in zip(SEEDS, margin_ratios)
                },
                "mean_margin_ratio": float(np.mean(margin_ratios)),
            }
        )
    return cases


def _balanced_select(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[case["defect_type"]].append(case)
    for rows in by_type.values():
        rows.sort(key=lambda row: (-row["mean_margin_ratio"], row["image_path"]))
    selected = []
    names = sorted(by_type)
    while names and len(selected) < limit:
        remaining = []
        for name in names:
            if by_type[name] and len(selected) < limit:
                selected.append(by_type[name].pop(0))
            if by_type[name]:
                remaining.append(name)
        names = remaining
    return selected


def select_sheet_cases(cases: list[dict[str, Any]], limit: int = MAX_SHEET_CASES) -> dict[str, list[dict[str, Any]]]:
    """Choose compact, deterministic case sheets without changing any rule."""
    consensus_fn = [case for case in cases if case["group"] == "consensus_missed"]
    unstable_anomaly = [case for case in cases if case["group"] == "unstable_anomaly"]
    reviewed_normal = [case for case in cases if case["label"] == 0 and case["review_votes"] > 0]
    reviewed_normal.sort(
        key=lambda row: (-row["review_votes"], -row["mean_margin_ratio"], row["image_path"])
    )
    return {
        "consensus-false-negatives": _balanced_select(consensus_fn, limit),
        "unstable-anomalies": _balanced_select(unstable_anomaly, limit),
        "reviewed-normal-cases": reviewed_normal[:limit],
    }


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]


def _draw_box(draw: ImageDraw.ImageDraw, box: list[int] | None, color: str) -> None:
    if box is None:
        return
    x_min, y_min, x_max, y_max = box
    draw.rectangle(
        [x_min, y_min, max(x_min, x_max - 1), max(y_min, y_max - 1)],
        outline=color,
        width=2,
    )


def _load_selected_maps(
    bundles: list[dict[str, Any]], selected_indices: set[int]
) -> dict[int, dict[str, np.ndarray]]:
    visuals: dict[int, dict[str, Any]] = {
        index: {"normalized_sum": None, "pixel_votes": None, "ground_truth": None}
        for index in selected_indices
    }
    expected_paths = bundles[0]["sample_ids"]
    expected_shape = None
    for bundle in bundles:
        with np.load(bundle["maps_path"], allow_pickle=False) as saved:
            paths = saved["image_paths"].tolist()
            maps = saved["anomaly_maps"].astype(np.float32, copy=False)
            masks = saved["ground_truth_masks"].astype(np.bool_, copy=False)
            if paths != expected_paths or maps.shape != masks.shape or maps.ndim != 3:
                raise ValueError("Saved test maps differ from frozen test order")
            if expected_shape is None:
                expected_shape = maps.shape
            elif maps.shape != expected_shape:
                raise ValueError("Seed test-map shapes differ")
            for index in selected_indices:
                anomaly_map = maps[index]
                if not np.isfinite(anomaly_map).all():
                    raise ValueError(f"Non-finite anomaly map: {expected_paths[index]}")
                normalized = anomaly_map / bundle["pixel_threshold"]
                predicted = anomaly_map >= bundle["pixel_threshold"]
                visual = visuals[index]
                visual["normalized_sum"] = (
                    normalized.astype(np.float64)
                    if visual["normalized_sum"] is None
                    else visual["normalized_sum"] + normalized
                )
                visual["pixel_votes"] = (
                    predicted.astype(np.uint8)
                    if visual["pixel_votes"] is None
                    else visual["pixel_votes"] + predicted
                )
                if visual["ground_truth"] is None:
                    visual["ground_truth"] = masks[index].copy()
                elif not np.array_equal(visual["ground_truth"], masks[index]):
                    raise ValueError(f"Ground-truth masks differ across seeds: {expected_paths[index]}")
            del maps, masks
    return {
        index: {
            "mean_normalized_map": visual["normalized_sum"] / len(bundles),
            "majority_pixel_mask": visual["pixel_votes"] >= 2,
            "ground_truth": visual["ground_truth"],
        }
        for index, visual in visuals.items()
    }


def _case_panel(
    data_root: Path,
    manifest_row: dict[str, Any],
    case: dict[str, Any],
    visual: dict[str, np.ndarray],
) -> Image.Image:
    source_path = _safe_child(data_root, case["image_path"])
    if checksum(source_path) != manifest_row.get("image_sha256"):
        raise ValueError(f"Source image checksum differs from manifest: {case['image_path']}")
    with Image.open(source_path) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError(f"Unexpected Real-IAD source image: {case['image_path']}")
        original = image.resize((256, 256), Image.Resampling.BILINEAR).copy()
    normalized = np.clip(visual["mean_normalized_map"], 0.0, 1.0)
    heatmap = colorize_anomaly_values(normalized)
    original_array = np.asarray(original, dtype=np.uint8)
    alpha = (0.15 + 0.45 * normalized)[..., None]
    overlay = Image.fromarray(
        np.rint(original_array * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)
    )
    draw_overlay = ImageDraw.Draw(overlay)
    _draw_box(draw_overlay, _bbox(visual["majority_pixel_mask"]), "red")
    _draw_box(draw_overlay, _bbox(visual["ground_truth"]), "lime")

    panel = Image.new("RGB", (512, 292), "white")
    panel.paste(original, (0, 36))
    panel.paste(overlay, (256, 36))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text(
        (4, 4),
        f"{case['group']} {case['defect_type']} {Path(case['image_path']).name}",
        fill="black",
        font=font,
    )
    draw.text(
        (4, 19),
        f"review votes={case['review_votes']}/3 mean margin={case['mean_margin_ratio']:+.3f}",
        fill="black",
        font=font,
    )
    draw.text(
        (4, 278),
        "left: input | right: mean fixed maps | red: majority pred | green: GT",
        fill="black",
        font=font,
    )
    return panel


def _write_sheet(
    cases: list[dict[str, Any]],
    visuals: dict[int, dict[str, np.ndarray]],
    data_root: Path,
    manifest_by_path: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    columns = 2
    panel_width, panel_height = 512, 292
    rows = max(1, math.ceil(len(cases) / columns))
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), "#dddddd")
    if not cases:
        ImageDraw.Draw(sheet).text((12, 12), "No cases in this group", fill="black")
    for case_index, case in enumerate(cases):
        panel = _case_panel(
            data_root,
            manifest_by_path[case["image_path"]],
            case,
            visuals[case["index"]],
        )
        sheet.paste(
            panel,
            ((case_index % columns) * panel_width, (case_index // columns) * panel_height),
        )
    sheet.save(output_path, format="PNG")


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(int)
    by_type: dict[str, dict[str, int | float]] = {}
    for case in cases:
        groups[case["group"]] += 1
    defect_types = sorted({case["defect_type"] for case in cases if case["label"] == 1})
    for defect_type in defect_types:
        selected = [case for case in cases if case["label"] == 1 and case["defect_type"] == defect_type]
        vote_total = sum(case["review_votes"] for case in selected)
        by_type[defect_type] = {
            "count": len(selected),
            "consensus_detected": sum(case["review_votes"] == 3 for case in selected),
            "consensus_missed": sum(case["review_votes"] == 0 for case in selected),
            "unstable": sum(case["review_votes"] in (1, 2) for case in selected),
            "mean_seed_recall": vote_total / (3 * len(selected)),
        }
    return {"groups": dict(sorted(groups.items())), "by_defect_type": by_type}


def _markdown_summary(analysis: dict[str, Any]) -> str:
    lines = [
        "# Real-IAD PatchCore 다중 seed 오류 진단",
        "",
        "세 seed의 이미 공개된 고정 test 판정을 비교한 탐색적 진단이다. 모델·전처리·임곗값은 변경하지 않았다.",
        "이 결과로 같은 test에 맞춰 조정한 뒤 새 확증 성능이라고 보고하면 안 된다.",
    ]
    for category, result in analysis["categories"].items():
        groups = result["groups"]
        lines.extend(
            [
                "",
                f"## {category}",
                "",
                f"- 세 seed 공통 탐지 불량: {groups.get('consensus_detected', 0)}장",
                f"- 세 seed 공통 미탐 불량: {groups.get('consensus_missed', 0)}장",
                f"- seed별 판정 불안정 불량: {groups.get('unstable_anomaly', 0)}장",
                f"- 세 seed 공통 정상 판정: {groups.get('consensus_normal', 0)}장",
                f"- 세 seed 공통 정상 오탐: {groups.get('consensus_reviewed_normal', 0)}장",
                f"- seed별 판정 불안정 정상: {groups.get('unstable_normal', 0)}장",
                "",
                "| 결함 코드 | 수 | 공통 탐지 | 공통 미탐 | 불안정 | seed 평균 recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for defect_type, row in result["by_defect_type"].items():
            lines.append(
                f"| {defect_type} | {row['count']} | {row['consensus_detected']} | "
                f"{row['consensus_missed']} | {row['unstable']} | {row['mean_seed_recall']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 시각화 선택 규칙",
            "",
            f"각 시트는 최대 {MAX_SHEET_CASES}장이다. 불량은 결함 코드별 round-robin으로 뽑고, "
            "같은 코드 안에서는 고정 임곗값에 가까운 사례를 먼저 표시한다.",
            "정상 review 사례는 review seed 수와 평균 margin 순으로 표시한다.",
            "빨간 상자는 세 pixel map 중 2개 이상이 고정 pixel 임곗값을 넘은 영역이고 초록 상자는 GT다.",
            "",
            "## 다음 판단",
            "",
            "공통 미탐은 seed 변경으로 해결되지 않는 구조적 실패 후보이고, 불안정 사례는 seed 민감도 후보다.",
            "이 진단은 원인 가설을 세우기 위한 것이며 현재 test에서 임곗값을 다시 고르는 근거로 사용하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    analyses_root = (project_root / "artifacts" / "analyses").resolve()
    evaluation_root = args.evaluations_root.resolve(strict=True)
    data_root = args.root.resolve(strict=True)
    output_path = args.output.resolve()
    output_path.relative_to(analyses_root)
    if output_path.exists():
        raise FileExistsError(f"Diagnostics output already exists: {output_path}")

    analysis: dict[str, Any] = {
        "schema_version": 1,
        "status": "analyzed_multiseed_diagnostics",
        "analysis_id": output_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_status": EXPECTED_REALIAD_STATUS,
        "purpose": "descriptive consensus and seed-instability analysis after frozen test evaluation",
        "seeds": list(SEEDS),
        "model_changed": False,
        "thresholds_changed": False,
        "test_reused_for_diagnostics": True,
        "git": git_state(project_root),
        "categories": {},
    }
    category_payloads = {}
    for category, contract in CATEGORIES.items():
        bundles = [
            _load_bundle(
                evaluation_root / f"patchcore-{contract['slug']}-seed{seed}-eval-v1",
                category,
                seed,
            )
            for seed in SEEDS
        ]
        cases = build_consensus_cases(bundles)
        category_summary = summarize_cases(cases)
        category_summary["test_count"] = len(cases)
        category_summary["manifest_sha256"] = contract["manifest_sha256"]
        category_summary["evaluation_ids"] = [
            bundle["evaluation"]["evaluation_id"] for bundle in bundles
        ]
        analysis["categories"][category] = category_summary
        category_payloads[category] = (bundles, cases)

    output_path.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    for category, (bundles, cases) in category_payloads.items():
        category_path = output_path / CATEGORIES[category]["slug"]
        category_path.mkdir()
        cases_path = category_path / "cases.jsonl"
        with cases_path.open("x", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
        selected = select_sheet_cases(cases)
        selected_indices = {
            case["index"] for group in selected.values() for case in group
        }
        visuals = _load_selected_maps(bundles, selected_indices) if selected_indices else {}
        category_artifacts = {
            "cases": {"path": str(cases_path.relative_to(output_path)), "sha256": checksum(cases_path)},
            "sheets": {},
        }
        for name, sheet_cases in selected.items():
            sheet_path = category_path / f"{name}.png"
            _write_sheet(
                sheet_cases,
                visuals,
                data_root,
                bundles[0]["manifest_by_path"],
                sheet_path,
            )
            category_artifacts["sheets"][name] = {
                "path": str(sheet_path.relative_to(output_path)),
                "sha256": checksum(sheet_path),
                "shown": len(sheet_cases),
                "available": sum(
                    case["group"] == "consensus_missed"
                    if name == "consensus-false-negatives"
                    else case["group"] == "unstable_anomaly"
                    if name == "unstable-anomalies"
                    else case["label"] == 0 and case["review_votes"] > 0
                    for case in cases
                ),
            }
        artifacts[category] = category_artifacts

    analysis["artifacts"] = artifacts
    summary_path = output_path / "summary.md"
    summary_path.write_text(_markdown_summary(analysis), encoding="utf-8", newline="\n")
    analysis["summary_sha256"] = checksum(summary_path)
    write_json(output_path / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "analysis_id": analysis["analysis_id"],
                "categories": analysis["categories"],
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return analysis


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root")
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=project_root / "artifacts" / "evaluations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "analyses" / "realiad-patchcore-multiseed-v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_diagnostics(args)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
