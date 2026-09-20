"""Validate and summarize the exploratory PaDiM/PatchCore three-seed evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODELS = ("padim", "patchcore")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_COUNTS = {"test": 100, "normal": 60, "anomaly": 40, "failed": 0}
METRIC_NAMES = (
    "image_auroc",
    "image_average_precision",
    "precision",
    "recall",
    "f1",
    "escape_rate",
    "overkill_rate",
    "review_rate",
    "pixel_auroc",
    "pixel_aupro",
    "latency_p95_ms",
    "image_threshold",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _load_evaluation(path: Path) -> dict[str, Any]:
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"Evaluation path must be a directory: {directory}")
    evaluation = _json(directory / "evaluation.json")
    metrics = _json(directory / "metrics.json")
    fit_run = _json(directory / "fit-run.json")
    if evaluation.get("schema_version") != 1 or metrics.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation or metrics schema")
    if evaluation.get("status") not in {"evaluated", "evaluated_exploratory"}:
        raise ValueError("Evaluation is not complete")
    if evaluation.get("test_evaluated") is not True or evaluation.get("counts") != EXPECTED_COUNTS:
        raise ValueError("Evaluation does not contain the frozen 100-image test contract")
    if evaluation.get("metrics") != metrics:
        raise ValueError("metrics.json differs from evaluation.json")
    artifacts = evaluation.get("artifacts", {})
    if artifacts.get("metrics") != "metrics.json" or _sha256(directory / "metrics.json") != artifacts.get(
        "metrics_sha256"
    ):
        raise ValueError("metrics.json checksum differs from evaluation.json")
    config = fit_run.get("config", {})
    model = config.get("model")
    seed = config.get("model_seed")
    if model not in EXPECTED_MODELS or seed not in EXPECTED_SEEDS:
        raise ValueError("Unexpected model or model seed in fit-run.json")
    fit = evaluation.get("fit", {})
    if fit.get("run_id") != fit_run.get("run_id") or fit.get("model_seed") != seed:
        raise ValueError("Evaluation fit reference differs from fit-run.json")
    if fit.get("checkpoint_sha256") != fit_run.get("artifacts", {}).get("checkpoint_sha256"):
        raise ValueError("Evaluation checkpoint reference differs from fit-run.json")
    checkpoint_sha256 = fit.get("checkpoint_sha256")
    manifest_sha256 = evaluation.get("manifest_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha256)
    ):
        raise ValueError("Evaluation manifest or checkpoint checksum is invalid")
    image = metrics.get("image", {})
    fixed = image.get("fixed_threshold", {})
    rates = fixed.get("rates", {})
    pixel = metrics.get("pixel", {})
    confusion = fixed.get("counts")
    if not isinstance(confusion, dict) or set(confusion) != {"tp", "fp", "tn", "fn"}:
        raise ValueError("Image confusion counts are missing")
    test_sample_ids = evaluation.get("test_sample_ids")
    if (
        not isinstance(test_sample_ids, list)
        or len(test_sample_ids) != 100
        or len(set(test_sample_ids)) != 100
        or not all(isinstance(item, str) and item for item in test_sample_ids)
    ):
        raise ValueError("Evaluation test sample IDs are invalid")
    row = {
        "model": model,
        "seed": seed,
        "evaluation_id": evaluation.get("evaluation_id"),
        "fit_run_id": fit.get("run_id"),
        "checkpoint_sha256": fit.get("checkpoint_sha256"),
        "image_auroc": _finite(image.get("auroc", {}).get("value"), "image AUROC"),
        "image_average_precision": _finite(
            image.get("average_precision", {}).get("value"), "image average precision"
        ),
        "precision": _finite(rates.get("precision"), "precision"),
        "recall": _finite(rates.get("recall"), "recall"),
        "f1": _finite(rates.get("f1"), "f1"),
        "escape_rate": _finite(rates.get("escape_rate"), "escape rate"),
        "overkill_rate": _finite(rates.get("overkill_rate"), "overkill rate"),
        "review_rate": _finite(rates.get("review_rate"), "review rate"),
        "pixel_auroc": _finite(pixel.get("auroc", {}).get("value"), "pixel AUROC"),
        "pixel_aupro": _finite(pixel.get("aupro", {}).get("value"), "pixel AUPRO"),
        "latency_p95_ms": _finite(
            evaluation.get("latency_ms", {}).get("p95_higher"), "latency p95"
        ),
        "image_threshold": _finite(fixed.get("threshold"), "image threshold"),
        "counts": confusion,
    }
    expected_confusion_total = sum(row["counts"].get(key, -1000) for key in ("tp", "fp", "tn", "fn"))
    if expected_confusion_total != 100:
        raise ValueError("Image confusion counts do not total 100")
    if not isinstance(row["evaluation_id"], str) or not row["evaluation_id"]:
        raise ValueError("Evaluation ID is missing")
    return {
        "row": row,
        "manifest_sha256": manifest_sha256,
        "test_sample_ids": test_sample_ids,
    }


def _describe(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def build_summary(evaluation_paths: list[Path]) -> dict[str, Any]:
    """Validate exactly six frozen-test evaluations and aggregate seed variability."""
    if len(evaluation_paths) != 6:
        raise ValueError("Exactly six evaluations are required: two models × seeds 42, 43 and 44")
    loaded = [_load_evaluation(path) for path in evaluation_paths]
    manifests = {item["manifest_sha256"] for item in loaded}
    if len(manifests) != 1 or not next(iter(manifests), None):
        raise ValueError("All evaluations must use one non-empty manifest checksum")
    sample_lists = {json.dumps(item["test_sample_ids"], sort_keys=True) for item in loaded}
    if len(sample_lists) != 1:
        raise ValueError("All evaluations must use the exact same ordered test sample IDs")
    rows = [item["row"] for item in loaded]
    if len({row["evaluation_id"] for row in rows}) != 6:
        raise ValueError("Evaluation IDs must be unique")
    models: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        model_rows = sorted((row for row in rows if row["model"] == model), key=lambda row: row["seed"])
        if tuple(row["seed"] for row in model_rows) != EXPECTED_SEEDS:
            raise ValueError(f"{model} must contain exactly seeds 42, 43 and 44")
        models[model] = {
            "seeds": list(EXPECTED_SEEDS),
            "runs": model_rows,
            "aggregate": {
                name: _describe([row[name] for row in model_rows]) for name in METRIC_NAMES
            },
        }
    comparison_metrics = (
        "image_auroc",
        "image_average_precision",
        "recall",
        "f1",
        "escape_rate",
        "overkill_rate",
        "pixel_auroc",
        "pixel_aupro",
        "latency_p95_ms",
    )
    deltas = {
        name: models["patchcore"]["aggregate"][name]["mean"]
        - models["padim"]["aggregate"][name]["mean"]
        for name in comparison_metrics
    }
    return {
        "schema_version": 1,
        "status": "summarized_exploratory_multiseed",
        "interpretation_status": "exploratory_after_seed42_test_results_observed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": next(iter(manifests)),
        "test_count": 100,
        "expected_seeds": list(EXPECTED_SEEDS),
        "models": models,
        "patchcore_minus_padim_mean": deltas,
        "limitations": [
            "The same MVTec test split is reused for all seeds.",
            "Seed-42 test results were observed before these repeat runs were implemented.",
            "This summary measures exploratory seed variability, not unbiased deployment performance.",
            "No hyperparameter or threshold may be changed from these test results without new untouched data.",
        ],
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Transistor 3-seed exploratory summary",
        "",
        "Status: exploratory; seed-42 test results were already observed before the repeat runs.",
        "",
        "| Metric | PaDiM mean ± sample std | PatchCore mean ± sample std | PatchCore − PaDiM |",
        "|---|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        padim = summary["models"]["padim"]["aggregate"][name]
        patchcore = summary["models"]["patchcore"]["aggregate"][name]
        delta = patchcore["mean"] - padim["mean"]
        lines.append(
            f"| {name} | {padim['mean']:.6f} ± {padim['sample_std']:.6f} | "
            f"{patchcore['mean']:.6f} ± {patchcore['sample_std']:.6f} | {delta:+.6f} |"
        )
    lines.extend(["", "## Per-seed runs", ""])
    for model in EXPECTED_MODELS:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Seed | Image AUROC | Recall | F1 | Pixel AUPRO | Latency p95 ms |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in summary["models"][model]["runs"]:
            lines.append(
                f"| {row['seed']} | {row['image_auroc']:.6f} | {row['recall']:.6f} | "
                f"{row['f1']:.6f} | {row['pixel_aupro']:.6f} | {row['latency_p95_ms']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_summary(summary: dict[str, Any], output: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    summaries_root = (project_root / "artifacts" / "summaries").resolve()
    output_path = output.resolve()
    output_path.relative_to(summaries_root)
    if output_path.exists():
        raise FileExistsError(f"Summary output already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    json_path = output_path / "summary.json"
    markdown_path = output_path / "summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return {
        "status": summary["status"],
        "output": str(output_path),
        "summary_json_sha256": _sha256(json_path),
        "summary_markdown_sha256": _sha256(markdown_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=Path,
        required=True,
        help="Evaluation directory; pass exactly six times",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = write_summary(build_summary(args.evaluation), args.output)
    except Exception as exc:  # noqa: BLE001 - concise CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
