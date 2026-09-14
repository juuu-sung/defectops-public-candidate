"""Evaluate one immutable exploratory PatchCore fit on the frozen test split."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defectops.evaluation_metrics import (
    aupro_fpr_grid,
    binary_auroc,
    binary_average_precision,
    confusion_counts,
    connected_components_4,
    rates_from_confusion,
    stratified_bootstrap_interval,
    wilson_interval,
)
from defectops.manifest import canonical, verify_manifest
from defectops.padim_evaluation import (
    _extract_prediction,
    _fixed_image_metrics,
    _read_jsonl,
    _require_fit_ancestor,
    _safe_child,
    _safe_data_path,
    _scale_bbox,
    load_evaluation_config,
)
from defectops.padim_experiment import (
    EXPECTED_PACKAGES,
    checksum,
    git_state,
    initialize_cuda_device,
    installed_versions,
    read_json,
    records_for_split,
    write_json,
)
from defectops.patchcore_experiment import (
    EXPECTED_COMPARISON_STATUS,
    load_patchcore_experiment,
)


EXPECTED_EVALUATION_PURPOSE = (
    "exploratory frozen-rule evaluation of the seed-42 PatchCore fit"
)


def load_patchcore_evaluation_config(path: Path) -> dict[str, Any]:
    """Load the shared metric contract and its PatchCore-specific purpose."""
    raw_config = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw_config.get("comparison_status") != EXPECTED_COMPARISON_STATUS:
        raise ValueError("Unexpected PatchCore evaluation comparison status")
    config = load_evaluation_config(path)
    if config["purpose"] != EXPECTED_EVALUATION_PURPOSE:
        raise ValueError("Unexpected PatchCore evaluation purpose")
    config["comparison_status"] = EXPECTED_COMPARISON_STATUS
    return config


def validate_patchcore_fit_artifact(
    fit_path: Path,
    manifest: dict[str, Any],
    train_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Validate an immutable calibration-only PatchCore fit bundle."""
    fit_run = read_json(_safe_child(fit_path, "run.json"))
    if (
        fit_run.get("status") != "fit_calibrated_exploratory"
        or fit_run.get("comparison_status") != EXPECTED_COMPARISON_STATUS
        or fit_run.get("test_evaluated") is not False
    ):
        raise ValueError("Fit bundle is not a completed exploratory calibration-only run")
    if fit_run.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Fit and evaluation manifest hashes differ")
    fit_manifest = read_json(_safe_child(fit_path, "manifest.json"))
    if canonical(fit_manifest) != canonical(manifest):
        raise ValueError("Fit manifest snapshot differs from the evaluation manifest")

    fit_seed = fit_run.get("config", {}).get("model_seed")
    fit_config = load_patchcore_experiment(
        _safe_child(fit_path, "patchcore-pilot.toml"), manifest, fit_seed
    )
    if fit_run.get("config") != fit_config:
        raise ValueError("Fit configuration snapshot differs from run.json")
    expected_counts = {
        "train": len(train_records),
        "calibration": len(calibration_records),
        "test_reserved_not_loaded_by_this_fit": len(test_records),
    }
    if fit_run.get("counts") != expected_counts:
        raise ValueError("Fit sample counts differ from the frozen manifest")
    expected_ids = {
        "train": [record["image_path"] for record in train_records],
        "calibration": [record["image_path"] for record in calibration_records],
    }
    if fit_run.get("sample_ids") != expected_ids:
        raise ValueError("Fit sample IDs differ from the frozen train/calibration lists")

    fit_commit = fit_run.get("git", {}).get("commit")
    if not isinstance(fit_commit, str) or len(fit_commit) != 40:
        raise ValueError("Fit run is missing its Git commit")
    _require_fit_ancestor(project_root, fit_commit)
    artifacts = fit_run.get("artifacts", {})
    checkpoint_path = _safe_child(fit_path, artifacts.get("checkpoint"))
    calibration_scores_path = _safe_child(
        fit_path, artifacts.get("calibration_scores")
    )
    if checksum(checkpoint_path) != artifacts.get("checkpoint_sha256"):
        raise ValueError("Model checkpoint checksum differs from run.json")
    if checksum(calibration_scores_path) != artifacts.get(
        "calibration_scores_sha256"
    ):
        raise ValueError("Calibration score checksum differs from run.json")
    threshold = fit_run.get("calibration", {}).get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("Fit run is missing a finite image threshold")
    memory_shape = fit_run.get("memory_bank", {}).get("shape")
    if (
        not isinstance(memory_shape, list)
        or len(memory_shape) != 2
        or any(type(value) is not int or value <= 0 for value in memory_shape)
    ):
        raise ValueError("Fit run is missing a valid PatchCore memory-bank shape")
    return fit_run, fit_config, checkpoint_path, calibration_scores_path


def run_patchcore_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Reload PatchCore, freeze calibration-only rules, and evaluate test."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts").resolve()
    runs_root = (artifacts_root / "runs").resolve()
    evaluations_root = (artifacts_root / "evaluations").resolve()
    data_root = args.root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    evaluation_config_path = args.evaluation_config.resolve(strict=True)
    fit_path = args.fit_run.resolve(strict=True)
    fit_path.relative_to(runs_root)
    output_path = args.output.resolve()
    output_path.relative_to(evaluations_root)
    if output_path.exists():
        raise FileExistsError(
            "Evaluation output already exists and will not be overwritten. "
            f"Choose a new --output directory: {output_path}"
        )

    manifest = read_json(manifest_path)
    verify_manifest(data_root, manifest)
    evaluation_config = load_patchcore_evaluation_config(evaluation_config_path)
    train_records = records_for_split(manifest, "train")
    calibration_records = records_for_split(manifest, "calibration")
    test_records = records_for_split(manifest, "test")
    if len(test_records) != 100 or sum(record["label"] for record in test_records) != 40:
        raise ValueError(
            "Frozen transistor test split must contain 60 normal and 40 anomalous images"
        )
    evaluator_git = git_state(project_root)
    fit_run, fit_config, checkpoint_path, calibration_scores_path = (
        validate_patchcore_fit_artifact(
            fit_path,
            manifest,
            train_records,
            calibration_records,
            test_records,
            project_root,
        )
    )

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The lab evaluation requires Python 3.12, found {platform.python_version()}"
        )
    versions = installed_versions()

    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchvision
    from anomalib.models import Patchcore
    from PIL import Image

    if (
        torch.__version__ != EXPECTED_PACKAGES["torch"]
        or torchvision.__version__ != EXPECTED_PACKAGES["torchvision"]
    ):
        raise RuntimeError(
            "Imported torch/torchvision versions differ from the validated combination"
        )
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("Evaluation requires the validated CUDA 12.8 GPU environment")

    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(
        evaluation_config_path, output_path / "patchcore-evaluation-v1.toml"
    )
    shutil.copyfile(manifest_path, output_path / "manifest.json")
    shutil.copyfile(fit_path / "run.json", output_path / "fit-run.json")

    seed = fit_config["model_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = initialize_cuda_device(torch)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("model") != fit_config
        or checkpoint.get("manifest_sha256") != manifest["manifest_sha256"]
        or checkpoint.get("git_commit") != fit_run["git"]["commit"]
        or float(checkpoint.get("threshold"))
        != float(fit_run["calibration"]["threshold"])
        or checkpoint.get("memory_bank_shape")
        != fit_run["memory_bank"]["shape"]
    ):
        raise ValueError("Checkpoint metadata differs from the validated fit run")
    model = Patchcore(
        backbone=fit_config["backbone"],
        layers=fit_config["layers"],
        pre_trained=False,
        coreset_sampling_ratio=fit_config["coreset_sampling_ratio"],
        num_neighbors=fit_config["num_neighbors"],
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or "model.memory_bank" not in state_dict:
        raise ValueError("Checkpoint is missing the PatchCore memory bank")
    saved_memory_bank = state_dict["model.memory_bank"]
    if list(saved_memory_bank.shape) != checkpoint["memory_bank_shape"]:
        raise ValueError("Checkpoint memory-bank tensor shape differs from metadata")
    model.model.memory_bank = torch.empty_like(saved_memory_bank)
    model.load_state_dict(state_dict, strict=True)
    if list(model.model.memory_bank.shape) != checkpoint["memory_bank_shape"]:
        raise RuntimeError("Reloaded PatchCore memory-bank shape differs from checkpoint")
    model.to(device)
    model._is_fitted.fill_(True)
    model.model.eval()

    image_size = fit_config["image_size"]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def load_image(record: dict[str, Any]) -> tuple[Any, tuple[int, int]]:
        image_path = _safe_data_path(data_root, record["image_path"])
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (1024, 1024):
                raise ValueError(f"Unexpected PatchCore input contract: {record['image_path']}")
            original_size = image.size
            array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        tensor = functional.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return ((tensor - mean) / std).unsqueeze(0).to(device), original_size

    def predict(record: dict[str, Any]) -> tuple[float, Any, tuple[int, int]]:
        image_batch, original_size = load_image(record)
        with torch.no_grad():
            predictions = model.model(image_batch)
        return (*_extract_prediction(predictions, image_size), original_size)

    expected_calibration = _read_jsonl(calibration_scores_path)
    if [row.get("image_path") for row in expected_calibration] != [
        record["image_path"] for record in calibration_records
    ]:
        raise ValueError("Calibration score rows differ from the frozen calibration list")
    print(f"Re-scoring {len(calibration_records)} calibration images before opening test...")
    calibration_maps = []
    maximum_rescore_difference = 0.0
    for index, (record, expected_row) in enumerate(
        zip(calibration_records, expected_calibration, strict=True), start=1
    ):
        score, anomaly_map, _ = predict(record)
        expected_score = float(expected_row["raw_score"])
        difference = abs(score - expected_score)
        maximum_rescore_difference = max(maximum_rescore_difference, difference)
        if difference > 1e-4 + 1e-5 * abs(expected_score):
            raise RuntimeError(
                f"Reloaded model score differs for {record['image_path']}: {difference}"
            )
        calibration_maps.append(anomaly_map)
        if index % 10 == 0 or index == len(calibration_records):
            print(f"  calibration {index}/{len(calibration_records)}")
    calibration_map_array = np.stack(calibration_maps).astype(np.float32, copy=False)
    pixel_threshold = float(
        np.quantile(
            calibration_map_array.reshape(-1),
            evaluation_config["pixel_calibration_quantile"],
            method="higher",
        )
    )
    image_threshold = float(fit_run["calibration"]["threshold"])
    calibration_contract = {
        "status": "exploratory_rules_frozen_before_test_images_loaded",
        "comparison_status": EXPECTED_COMPARISON_STATUS,
        "fit_run_id": fit_run["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "score_space": "raw",
        "image": {
            "threshold": image_threshold,
            "source": "fit_run_normal_calibration",
            "sample_count": len(calibration_records),
            "decision_rule": "score >= threshold",
        },
        "pixel": {
            "threshold": pixel_threshold,
            "source": "pooled_normal_calibration_anomaly_map_pixels",
            "image_count": len(calibration_records),
            "pixel_count": int(calibration_map_array.size),
            "quantile": evaluation_config["pixel_calibration_quantile"],
            "quantile_method": "higher",
            "decision_rule": "score >= threshold",
            "morphology": "none",
            "connectivity": 4,
            "minimum_region_area_pixels": 1,
        },
        "model_reload_maximum_absolute_score_difference": maximum_rescore_difference,
    }
    calibration_contract_path = output_path / "calibration-contract.json"
    write_json(calibration_contract_path, calibration_contract)

    print(f"Evaluating {len(test_records)} frozen test images...")
    test_started_at = datetime.now(timezone.utc)
    test_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    test_maps = []
    test_masks = []
    latencies_ms = []
    for index, record in enumerate(test_records, start=1):
        sample_start = time.perf_counter()
        score, anomaly_map, original_size = predict(record)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - sample_start) * 1000.0
        latencies_ms.append(latency_ms)
        if record["label"] == 1:
            mask_path = _safe_data_path(data_root, record["mask_path"])
            with Image.open(mask_path) as mask_image:
                if mask_image.size != original_size:
                    raise RuntimeError(f"Mask dimensions differ for {record['image_path']}")
                resized_mask = mask_image.convert("L").resize(
                    (image_size, image_size), resample=Image.Resampling.NEAREST
                )
                ground_truth = np.asarray(resized_mask, dtype=np.uint8) > 0
        else:
            ground_truth = np.zeros((image_size, image_size), dtype=np.bool_)
        predicted_mask = anomaly_map >= pixel_threshold
        regions = []
        for component in connected_components_4(predicted_mask):
            map_box = component["bbox_xyxy_exclusive"]
            regions.append(
                {
                    "area_map_pixels": component["area"],
                    "bbox_map_xyxy_exclusive": map_box,
                    "bbox_original_xyxy_exclusive": _scale_bbox(
                        map_box, anomaly_map.shape, original_size
                    ),
                }
            )
        decision = "REVIEW_REQUIRED" if score >= image_threshold else "PASS"
        rows.append(
            {
                "image_path": record["image_path"],
                "label": record["label"],
                "defect_type": record["defect_type"],
                "raw_score": score,
                "image_threshold": image_threshold,
                "decision": decision,
                "correct": (decision == "REVIEW_REQUIRED") == bool(record["label"]),
                "latency_ms": latency_ms,
                "original_size_wh": list(original_size),
                "map_size_hw": list(anomaly_map.shape),
                "pixel_threshold": pixel_threshold,
                "anomalous_map_pixels": int(np.count_nonzero(predicted_mask)),
                "anomalous_map_fraction": float(np.mean(predicted_mask)),
                "regions": regions,
            }
        )
        test_maps.append(anomaly_map)
        test_masks.append(ground_truth)
        if index % 10 == 0 or index == len(test_records):
            print(f"  test {index}/{len(test_records)}")
    test_elapsed_seconds = time.perf_counter() - test_start
    test_finished_at = datetime.now(timezone.utc)

    maps_array = np.stack(test_maps).astype(np.float32, copy=False)
    masks_array = np.stack(test_masks).astype(np.bool_, copy=False)
    labels = np.asarray([record["label"] for record in test_records], dtype=np.bool_)
    scores = np.asarray([row["raw_score"] for row in rows], dtype=np.float64)
    image_counts = confusion_counts(labels, scores, image_threshold)
    confidence_level = evaluation_config["confidence_level"]
    image_metrics = {
        "auroc": {
            "value": binary_auroc(labels, scores),
            "confidence_interval": stratified_bootstrap_interval(
                labels,
                scores,
                binary_auroc,
                seed=evaluation_config["bootstrap_seed"],
                replicates=evaluation_config["bootstrap_replicates"],
                confidence_level=confidence_level,
            ),
        },
        "average_precision": {
            "value": binary_average_precision(labels, scores),
            "confidence_interval": stratified_bootstrap_interval(
                labels,
                scores,
                binary_average_precision,
                seed=evaluation_config["bootstrap_seed"] + 1,
                replicates=evaluation_config["bootstrap_replicates"],
                confidence_level=confidence_level,
            ),
            "test_anomaly_prevalence": float(np.mean(labels)),
        },
        "fixed_threshold": {
            "threshold": image_threshold,
            **_fixed_image_metrics(image_counts, confidence_level),
        },
    }

    defect_type_metrics = {}
    defect_types = sorted(
        {record["defect_type"] for record in test_records if record["label"] == 1}
    )
    for defect_type in defect_types:
        indices = [
            index
            for index, record in enumerate(test_records)
            if record["defect_type"] == defect_type
        ]
        reviewed = sum(rows[index]["decision"] == "REVIEW_REQUIRED" for index in indices)
        defect_type_metrics[defect_type] = {
            "count": len(indices),
            "reviewed": reviewed,
            "recall": reviewed / len(indices),
            "confidence_interval": wilson_interval(
                reviewed, len(indices), confidence_level
            ),
        }

    pixel_labels = masks_array.reshape(-1)
    pixel_scores = maps_array.reshape(-1)
    pixel_counts = confusion_counts(pixel_labels, pixel_scores, pixel_threshold)
    pixel_metrics = {
        "auroc": {"value": binary_auroc(pixel_labels, pixel_scores)},
        "aupro": aupro_fpr_grid(
            maps_array,
            masks_array,
            fpr_limit=evaluation_config["aupro_fpr_limit"],
            fpr_points=evaluation_config["aupro_fpr_points"],
        ),
        "fixed_threshold": {
            "threshold": pixel_threshold,
            "counts": pixel_counts,
            "rates": rates_from_confusion(pixel_counts),
        },
        "uncertainty": "not estimated from pixels as independent observations",
    }
    metrics = {
        "schema_version": 1,
        "comparison_status": EXPECTED_COMPARISON_STATUS,
        "image": image_metrics,
        "image_by_defect_type": defect_type_metrics,
        "pixel": pixel_metrics,
        "confidence": {
            "level": confidence_level,
            "ranking_method": "image-stratified percentile bootstrap",
            "bootstrap_seed": evaluation_config["bootstrap_seed"],
            "bootstrap_replicates": evaluation_config["bootstrap_replicates"],
            "proportion_method": "Wilson score interval",
        },
    }
    metrics_path = output_path / "metrics.json"
    write_json(metrics_path, metrics)

    predictions_path = output_path / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    maps_path = output_path / "test-maps.npz"
    with maps_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            image_paths=np.asarray([record["image_path"] for record in test_records]),
            anomaly_maps=maps_array,
            ground_truth_masks=masks_array.astype(np.uint8),
        )

    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    evaluation = {
        "schema_version": 1,
        "status": "evaluated_exploratory",
        "comparison_status": EXPECTED_COMPARISON_STATUS,
        "test_evaluated": True,
        "purpose": evaluation_config["purpose"],
        "evaluation_id": output_path.name,
        "test_started_at": test_started_at.isoformat(),
        "test_finished_at": test_finished_at.isoformat(),
        "test_elapsed_seconds": test_elapsed_seconds,
        "evaluator_git": evaluator_git,
        "fit": {
            "run_id": fit_run["run_id"],
            "git_commit": fit_run["git"]["commit"],
            "checkpoint_sha256": checksum(checkpoint_path),
            "model_seed": fit_config["model_seed"],
            "memory_bank_shape": fit_run["memory_bank"]["shape"],
        },
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_config": evaluation_config,
        "counts": {
            "test": len(test_records),
            "normal": int(np.count_nonzero(~labels)),
            "anomaly": int(np.count_nonzero(labels)),
            "failed": 0,
        },
        "test_sample_ids": [record["image_path"] for record in test_records],
        "calibration_contract": calibration_contract,
        "latency_ms": {
            "scope": "image decode, preprocessing, GPU inference, map transfer",
            "batch_size": 1,
            "mean": float(np.mean(latency_array)),
            "median": float(np.median(latency_array)),
            "p95_higher": float(np.quantile(latency_array, 0.95, method="higher")),
            "maximum": float(np.max(latency_array)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": versions,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "metrics": metrics,
        "artifacts": {
            "calibration_contract": calibration_contract_path.name,
            "calibration_contract_sha256": checksum(calibration_contract_path),
            "metrics": metrics_path.name,
            "metrics_sha256": checksum(metrics_path),
            "predictions": predictions_path.name,
            "predictions_sha256": checksum(predictions_path),
            "test_maps": maps_path.name,
            "test_maps_sha256": checksum(maps_path),
            "evaluation_config_snapshot": "patchcore-evaluation-v1.toml",
            "manifest_snapshot": "manifest.json",
            "fit_run_snapshot": "fit-run.json",
        },
        "limitations": [
            "The PaDiM official test result was observed before PatchCore was configured.",
            "This result is exploratory and is not a confirmatory model-selection estimate.",
            "MVTec test prevalence does not represent factory prevalence.",
            "Pixel confidence intervals do not treat pixels as independent samples.",
            "Any further tuning requires new untouched data for an unbiased comparison.",
        ],
    }
    evaluation_path = output_path / "evaluation.json"
    write_json(evaluation_path, evaluation)
    print(
        json.dumps(
            {
                "status": evaluation["status"],
                "comparison_status": evaluation["comparison_status"],
                "evaluation_id": evaluation["evaluation_id"],
                "image_auroc": image_metrics["auroc"]["value"],
                "image_average_precision": image_metrics["average_precision"]["value"],
                "image_fixed_threshold": image_metrics["fixed_threshold"],
                "pixel_auroc": pixel_metrics["auroc"]["value"],
                "pixel_aupro": pixel_metrics["aupro"]["value"],
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return evaluation


def build_parser() -> argparse.ArgumentParser:
    """Build the exploratory PatchCore test-evaluation parser."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset parent directory")
    parser.add_argument(
        "--fit-run",
        type=Path,
        required=True,
        help="Completed PatchCore fit directory below artifacts/runs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "transistor-v1.json",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=project_root / "configs" / "patchcore-evaluation-v1.toml",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory below artifacts/evaluations"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the exploratory PatchCore evaluation."""
    args = build_parser().parse_args(argv)
    try:
        run_patchcore_evaluation(args)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
