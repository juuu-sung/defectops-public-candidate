"""Validate and summarize the frozen three-category, three-seed comparison."""

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


MODELS = ("padim", "patchcore")
SEEDS = (42, 43, 44)
CATEGORY_CONTRACTS = {
    "transistor": {
        "manifest_sha256": "9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585",
        "counts": {"test": 100, "normal": 60, "anomaly": 40, "failed": 0},
    },
    "cable": {
        "manifest_sha256": "f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70",
        "counts": {"test": 150, "normal": 58, "anomaly": 92, "failed": 0},
    },
    "metal_nut": {
        "manifest_sha256": "6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5",
        "counts": {"test": 115, "normal": 22, "anomaly": 93, "failed": 0},
    },
}
METRICS = (
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
    "evaluation_latency_p95_ms",
)
SELECTION_METRICS = ("escape_rate", "image_auroc", "overkill_rate")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def _describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _expected_status(category: str, model: str) -> str:
    if model == "padim":
        return "evaluated"
    return "evaluated_exploratory" if category == "transistor" else "evaluated_preregistered"


def _load_evaluation(path: Path) -> dict[str, Any]:
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"Evaluation path must be a directory: {directory}")
    evaluation = _json(directory / "evaluation.json")
    metrics = _json(directory / "metrics.json")
    fit_run = _json(directory / "fit-run.json")
    if evaluation.get("schema_version") != 1 or metrics.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation or metrics schema")
    if evaluation.get("metrics") != metrics:
        raise ValueError("metrics.json differs from evaluation.json")
    artifacts = evaluation.get("artifacts", {})
    if (
        artifacts.get("metrics") != "metrics.json"
        or _sha256(directory / "metrics.json") != artifacts.get("metrics_sha256")
    ):
        raise ValueError("metrics.json checksum differs from evaluation.json")

    config = fit_run.get("config", {})
    model = config.get("model")
    seed = config.get("model_seed")
    if model not in MODELS or seed not in SEEDS:
        raise ValueError("Unexpected model or model seed in fit-run.json")
    fit = evaluation.get("fit", {})
    if fit.get("run_id") != fit_run.get("run_id") or fit.get("model_seed") != seed:
        raise ValueError("Evaluation fit reference differs from fit-run.json")
    if fit.get("checkpoint_sha256") != fit_run.get("artifacts", {}).get("checkpoint_sha256"):
        raise ValueError("Evaluation checkpoint reference differs from fit-run.json")

    sample_ids = evaluation.get("test_sample_ids")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(sample_ids) != len(set(sample_ids))
        or not all(isinstance(item, str) and "/" in item for item in sample_ids)
    ):
        raise ValueError("Evaluation test sample IDs are invalid")
    prefixes = {item.split("/", 1)[0] for item in sample_ids}
    if len(prefixes) != 1:
        raise ValueError("Evaluation mixes categories")
    category = next(iter(prefixes))
    if category not in CATEGORY_CONTRACTS:
        raise ValueError(f"Unexpected category: {category}")
    contract = CATEGORY_CONTRACTS[category]
    if evaluation.get("manifest_sha256") != contract["manifest_sha256"]:
        raise ValueError(f"{category} manifest checksum differs from the registered contract")
    if evaluation.get("counts") != contract["counts"] or len(sample_ids) != contract["counts"]["test"]:
        raise ValueError(f"{category} evaluation counts differ from the registered contract")
    if evaluation.get("status") != _expected_status(category, model):
        raise ValueError("Evaluation status differs from the evidence contract")
    if evaluation.get("test_evaluated") is not True:
        raise ValueError("Evaluation is not complete")

    image = metrics.get("image", {})
    fixed = image.get("fixed_threshold", {})
    rates = fixed.get("rates", {})
    pixel = metrics.get("pixel", {})
    confusion = fixed.get("counts")
    if not isinstance(confusion, dict) or set(confusion) != {"tp", "fp", "tn", "fn"}:
        raise ValueError("Image confusion counts are missing")
    if (
        sum(confusion.values()) != contract["counts"]["test"]
        or confusion["tp"] + confusion["fn"] != contract["counts"]["anomaly"]
        or confusion["tn"] + confusion["fp"] != contract["counts"]["normal"]
    ):
        raise ValueError("Image confusion counts differ from category totals")

    row = {
        "category": category,
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
        "evaluation_latency_p95_ms": _finite(
            evaluation.get("latency_ms", {}).get("p95_higher"), "evaluation latency p95"
        ),
        "counts": confusion,
    }
    for name in METRICS[:-1]:
        if not 0.0 <= row[name] <= 1.0:
            raise ValueError(f"Metric outside [0, 1]: {name}")
    if not isinstance(row["evaluation_id"], str) or not row["evaluation_id"]:
        raise ValueError("Evaluation ID is missing")
    checkpoint = row["checkpoint_sha256"]
    if (
        not isinstance(checkpoint, str)
        or len(checkpoint) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint)
    ):
        raise ValueError("Checkpoint SHA-256 is invalid")
    return {"row": row, "sample_ids": sample_ids}


def _select_candidate(models: dict[str, Any]) -> dict[str, Any]:
    padim = models["padim"]["macro_across_categories_by_seed"]
    patchcore = models["patchcore"]["macro_across_categories_by_seed"]
    rules = (
        ("escape_rate", "lower"),
        ("image_auroc", "higher"),
        ("overkill_rate", "lower"),
    )
    for metric, direction in rules:
        left = float(padim[metric]["mean"])
        right = float(patchcore[metric]["mean"])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            if direction == "lower":
                winner = "padim" if left < right else "patchcore"
            else:
                winner = "padim" if left > right else "patchcore"
            return {
                "status": "selected_before_warm_latency_tiebreaker",
                "portfolio_candidate": winner,
                "deciding_metric": metric,
                "direction": direction,
                "automatic_pass_allowed": False,
            }
    return {
        "status": "requires_frozen_warm_latency_tiebreaker",
        "portfolio_candidate": None,
        "deciding_metric": "warm_p95_latency",
        "direction": "lower",
        "automatic_pass_allowed": False,
    }


def build_summary(evaluation_paths: list[Path]) -> dict[str, Any]:
    """Validate exactly 18 evaluations and aggregate equal-weight category macros."""
    if len(evaluation_paths) != 18:
        raise ValueError("Exactly 18 evaluations are required: 3 categories × 2 models × 3 seeds")
    loaded = [_load_evaluation(path) for path in evaluation_paths]
    rows = [item["row"] for item in loaded]
    keys = [(row["category"], row["model"], row["seed"]) for row in rows]
    expected_keys = {
        (category, model, seed)
        for category in CATEGORY_CONTRACTS
        for model in MODELS
        for seed in SEEDS
    }
    if set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("Evaluations do not contain each category/model/seed exactly once")
    if len({row["evaluation_id"] for row in rows}) != 18:
        raise ValueError("Evaluation IDs must be unique")
    for category in CATEGORY_CONTRACTS:
        sample_sets = {
            json.dumps(item["sample_ids"], separators=(",", ":"))
            for item in loaded
            if item["row"]["category"] == category
        }
        if len(sample_sets) != 1:
            raise ValueError(f"{category} evaluations use different ordered test sample IDs")

    categories: dict[str, Any] = {}
    for category in CATEGORY_CONTRACTS:
        categories[category] = {}
        for model in MODELS:
            selected = sorted(
                (row for row in rows if row["category"] == category and row["model"] == model),
                key=lambda row: row["seed"],
            )
            categories[category][model] = {
                "runs": selected,
                "aggregate_across_seeds": {
                    metric: _describe([row[metric] for row in selected]) for metric in METRICS
                },
            }

    models: dict[str, Any] = {}
    for model in MODELS:
        per_seed = []
        for seed in SEEDS:
            row = {"seed": seed}
            for metric in METRICS:
                row[metric] = statistics.fmean(
                    categories[category][model]["runs"][seed - SEEDS[0]][metric]
                    for category in CATEGORY_CONTRACTS
                )
            per_seed.append(row)
        models[model] = {
            "category_equal_weight_macro_by_seed": per_seed,
            "macro_across_categories_by_seed": {
                metric: _describe([row[metric] for row in per_seed]) for metric in METRICS
            },
        }

    deltas = {
        metric: (
            models["patchcore"]["macro_across_categories_by_seed"][metric]["mean"]
            - models["padim"]["macro_across_categories_by_seed"][metric]["mean"]
        )
        for metric in METRICS
    }
    selection = _select_candidate(models)
    return {
        "schema_version": 1,
        "status": "summarized_preregistered_three_category_mixed_evidence",
        "evidence_status": "mixed_transistor_observed_additional_categories_preregistered",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "models": models,
        "patchcore_minus_padim_macro_mean": deltas,
        "selection": selection,
        "selection_rule": {
            "primary": "lowest_macro_escape_rate_at_frozen_threshold",
            "tie_breakers": [
                "highest_macro_image_auroc",
                "lowest_macro_overkill_rate",
                "lowest_warm_p95_latency",
            ],
        },
        "limitations": [
            "Transistor test results were observed before the additional-category protocol was registered.",
            "Cable and metal_nut use preregistered frozen rules, but all categories use public MVTec test splits.",
            "Macro values give each category equal weight and then summarize the three seed-level macros.",
            "Evaluation latency includes decode and preprocessing and is not the frozen warm-latency tiebreaker.",
            "The selected model is a portfolio candidate; automatic PASS remains prohibited.",
            "Post-test tuning requires new untouched data.",
        ],
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Three-category PaDiM vs PatchCore summary",
        "",
        "Evidence: mixed; transistor was previously observed, while cable and metal_nut were preregistered.",
        "",
        "## Equal-weight category macro",
        "",
        "| Metric | PaDiM mean ± sample std | PatchCore mean ± sample std | PatchCore − PaDiM |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        padim = summary["models"]["padim"]["macro_across_categories_by_seed"][metric]
        patchcore = summary["models"]["patchcore"]["macro_across_categories_by_seed"][metric]
        lines.append(
            f"| {metric} | {padim['mean']:.6f} ± {padim['sample_std']:.6f} | "
            f"{patchcore['mean']:.6f} ± {patchcore['sample_std']:.6f} | "
            f"{summary['patchcore_minus_padim_macro_mean'][metric]:+.6f} |"
        )
    lines.extend(["", "## Category and seed results", ""])
    for category in CATEGORY_CONTRACTS:
        lines.extend([
            f"### {category}",
            "",
            "| Model | Seed | Image AUROC | Recall | Escape rate | Overkill rate | Pixel AUPRO |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model in MODELS:
            for row in summary["categories"][category][model]["runs"]:
                lines.append(
                    f"| {model} | {row['seed']} | {row['image_auroc']:.6f} | "
                    f"{row['recall']:.6f} | {row['escape_rate']:.6f} | "
                    f"{row['overkill_rate']:.6f} | {row['pixel_aupro']:.6f} |"
                )
        lines.append("")
    selection = summary["selection"]
    lines.extend([
        "## Frozen selection rule result",
        "",
        f"- Status: `{selection['status']}`",
        f"- Portfolio candidate: `{selection['portfolio_candidate']}`",
        f"- Deciding metric: `{selection['deciding_metric']}`",
        "- Automatic PASS allowed: `false`",
        "",
        "## Interpretation limits",
        "",
    ])
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
        "portfolio_candidate": summary["selection"]["portfolio_candidate"],
        "output": str(output_path),
        "summary_json_sha256": _sha256(json_path),
        "summary_markdown_sha256": _sha256(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = write_summary(build_summary(args.evaluation), args.output)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
