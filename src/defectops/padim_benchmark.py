"""Benchmark warm PaDiM inference with one model load and fixed test inputs."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from defectops.padim_experiment import checksum, read_json, write_json
from defectops.padim_inspection import load_inference_runtime, predict_validated_png


def latency_summary(values: list[float]) -> dict[str, float | int]:
    """Summarize a non-empty finite latency sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Latency values must be a non-empty finite non-negative list")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95_higher": float(np.quantile(array, 0.95, method="higher")),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _safe_existing_child(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Path must be a non-empty string")
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


def run_warm_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Load one runtime, warm it up and measure repeated batch-size-one calls."""
    if type(args.warmup) is not int or args.warmup < 1:
        raise ValueError("--warmup must be a positive integer")
    if type(args.repeats) is not int or not 1 <= args.repeats <= 5:
        raise ValueError("--repeats must be an integer from 1 to 5")
    if type(args.count) is not int or not 1 <= args.count <= 100:
        raise ValueError("--count must be an integer from 1 to 100")

    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts").resolve()
    fit_root = (artifacts_root / "runs").resolve()
    evaluation_root = (artifacts_root / "evaluations").resolve()
    benchmark_root = (artifacts_root / "benchmarks").resolve()
    data_root = args.root.resolve(strict=True)
    fit_path = args.fit_run.resolve(strict=True)
    evaluation_path = args.evaluation.resolve(strict=True)
    output_path = args.output.resolve()
    fit_path.relative_to(fit_root)
    evaluation_path.relative_to(evaluation_root)
    output_path.relative_to(benchmark_root)
    if output_path.exists():
        raise FileExistsError(f"Benchmark output already exists: {output_path}")

    evaluation = read_json(_safe_existing_child(evaluation_path, "evaluation.json"))
    manifest = read_json(_safe_existing_child(evaluation_path, "manifest.json"))
    manifest_records = {
        record["image_path"]: record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("image_path"), str)
    }
    sample_ids = evaluation.get("test_sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != 100:
        raise ValueError("Warm benchmark requires the frozen 100-image test list")
    selected_ids = sample_ids[: args.count]
    prediction_name = evaluation.get("artifacts", {}).get("predictions")
    predictions_path = _safe_existing_child(evaluation_path, prediction_name)
    expected_predictions_sha256 = evaluation.get("artifacts", {}).get("predictions_sha256")
    if checksum(predictions_path) != expected_predictions_sha256:
        raise ValueError("Evaluation predictions checksum differs")
    expected_rows = _read_jsonl(predictions_path)
    expected_scores = {row["image_path"]: float(row["raw_score"]) for row in expected_rows}
    if [row.get("image_path") for row in expected_rows] != sample_ids:
        raise ValueError("Evaluation predictions differ from frozen test sample IDs")

    runtime = load_inference_runtime(project_root, fit_path, evaluation_path)
    first_path = _safe_existing_child(data_root, selected_ids[0])
    first_expected_sha256 = manifest_records[selected_ids[0]].get("image_sha256")
    for _ in range(args.warmup):
        warmup_prediction = predict_validated_png(runtime, first_path)
        if warmup_prediction["input_sha256"] != first_expected_sha256:
            raise ValueError("Warmup input checksum differs from manifest")
    runtime.torch.cuda.reset_peak_memory_stats()

    output_path.mkdir(parents=True, exist_ok=False)
    rows = []
    maximum_score_difference = 0.0
    measured_started = time.perf_counter()
    for repeat in range(1, args.repeats + 1):
        for image_index, image_path in enumerate(selected_ids, start=1):
            source_path = _safe_existing_child(data_root, image_path)
            prediction = predict_validated_png(runtime, source_path)
            expected_sha256 = manifest_records[image_path].get("image_sha256")
            if prediction["input_sha256"] != expected_sha256:
                raise ValueError(f"Benchmark input checksum differs from manifest: {image_path}")
            expected_score = expected_scores[image_path]
            difference = abs(prediction["raw_score"] - expected_score)
            maximum_score_difference = max(maximum_score_difference, difference)
            if difference > 1e-4 + 1e-5 * abs(expected_score):
                raise RuntimeError(f"Warm score differs from evaluation: {image_path}")
            rows.append(
                {
                    "repeat": repeat,
                    "image_index": image_index,
                    "image_path": image_path,
                    "input_sha256": prediction["input_sha256"],
                    "raw_score": prediction["raw_score"],
                    "score_difference_from_evaluation": difference,
                    "latency_ms": prediction["latency_ms"],
                }
            )
            completed = (repeat - 1) * len(selected_ids) + image_index
            if completed % 10 == 0 or completed == args.repeats * len(selected_ids):
                print(f"  benchmark {completed}/{args.repeats * len(selected_ids)}")
    runtime.torch.cuda.synchronize()
    measured_wall_seconds = time.perf_counter() - measured_started

    samples_path = output_path / "samples.jsonl"
    with samples_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    latency_keys = (
        "decode_and_checksum",
        "preprocess",
        "gpu_inference_and_map_transfer",
        "sample_total",
    )
    latency = {
        key: latency_summary([row["latency_ms"][key] for row in rows])
        for key in latency_keys
    }
    benchmark = {
        "schema_version": 1,
        "status": "benchmarked_warm",
        "benchmark_id": output_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "performance measurement only; labels and thresholds are not tuned",
        "batch_size": 1,
        "warmup_invocations": args.warmup,
        "repeats": args.repeats,
        "unique_image_count": len(selected_ids),
        "measured_invocations": len(rows),
        "measured_wall_seconds": measured_wall_seconds,
        "throughput_images_per_second": len(rows) / measured_wall_seconds,
        "setup_and_model_load_ms": runtime.setup_ms,
        "latency_ms": latency,
        "maximum_absolute_score_difference_from_evaluation": maximum_score_difference,
        "gpu_memory": {
            "peak_allocated_bytes": runtime.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": runtime.torch.cuda.max_memory_reserved(),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": runtime.versions,
            "torch_cuda_runtime": runtime.torch.version.cuda,
            "gpu": runtime.torch.cuda.get_device_name(),
        },
        "trace": {
            "benchmark_git": runtime.current_git,
            "fit_run_id": runtime.fit_run["run_id"],
            "evaluation_id": evaluation["evaluation_id"],
            "manifest_sha256": evaluation["manifest_sha256"],
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "thresholds_changed": False,
            "labels_used_for_timing": False,
        },
        "artifacts": {
            "samples": samples_path.name,
            "samples_sha256": checksum(samples_path),
        },
        "limitations": [
            "This measures batch-size-one warm calls in one Python process.",
            "Decode and checksum time is reported separately from GPU inference.",
            "The benchmark reuses frozen test inputs for timing only and is not a new evaluation.",
        ],
    }
    write_json(output_path / "benchmark.json", benchmark)
    return benchmark


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset parent directory")
    parser.add_argument("--fit-run", type=Path, required=True, help="Completed fit directory")
    parser.add_argument(
        "--evaluation", type=Path, required=True, help="Completed evaluation directory"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Unmeasured warmup invocations")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Measured passes over selected inputs"
    )
    parser.add_argument("--count", type=int, default=100, help="Frozen test inputs per pass")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "benchmarks" / "padim-warm-v1",
        help="New directory below artifacts/benchmarks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_warm_benchmark(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "benchmark_id": result["benchmark_id"],
                "measured_invocations": result["measured_invocations"],
                "throughput_images_per_second": result["throughput_images_per_second"],
                "gpu_inference_latency_ms": result["latency_ms"][
                    "gpu_inference_and_map_transfer"
                ],
                "sample_total_latency_ms": result["latency_ms"]["sample_total"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
