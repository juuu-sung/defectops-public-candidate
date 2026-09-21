"""Run one traceable anomaly inspection from immutable validated bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from defectops.evaluation_metrics import connected_components_4
from defectops.manifest import canonical
from defectops.padim_evaluation import load_evaluation_config
from defectops.padim_experiment import (
    EXPECTED_PACKAGES,
    checksum,
    git_state,
    initialize_cuda_device,
    installed_versions,
    load_experiment,
    read_json,
    write_json,
)
from defectops.patchcore_evaluation import load_patchcore_evaluation_config
from defectops.patchcore_experiment import (
    EXPECTED_COMPARISON_STATUS,
    EXPECTED_PREREGISTERED_STATUS,
    load_patchcore_experiment,
)


class InputValidationError(ValueError):
    """Raised when an inspection image violates the frozen input contract."""


@dataclass
class PadimRuntime:
    """One validated, loaded PaDiM or PatchCore runtime."""

    current_git: dict[str, str]
    fit_path: Path
    evaluation_path: Path
    fit_run: dict[str, Any]
    fit_config: dict[str, Any]
    calibration_contract: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    versions: dict[str, str]
    torch: Any
    functional: Any
    model: Any
    device: Any
    setup_ms: float


def colorize_anomaly_values(values: np.ndarray) -> np.ndarray:
    """Map fixed-scale values in [0, 1] to a deterministic blue-to-red palette."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("display values must be a finite two-dimensional array")
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.rint(np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def scale_bbox_to_original(
    box: list[int], map_shape: tuple[int, int], image_size: tuple[int, int]
) -> list[int]:
    """Scale an exclusive XYXY map box to exclusive original-image coordinates."""
    map_height, map_width = map_shape
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = box
    return [
        math.floor(x_min * image_width / map_width),
        math.floor(y_min * image_height / map_height),
        min(image_width, math.ceil(x_max * image_width / map_width)),
        min(image_height, math.ceil(y_max * image_height / map_height)),
    ]


def _safe_named_file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"Unsafe artifact filename: {name!r}")
    path = (root / name).resolve(strict=True)
    path.relative_to(root)
    return path


def _require_ancestor(project_root: Path, commit: str, label: str) -> None:
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{label} is missing a full Git commit")
    result = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} commit is not an ancestor of the inspection commit")


def _verify_manifest_hash(manifest: dict[str, Any]) -> None:
    recorded = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = hashlib.sha256(canonical(payload)).hexdigest()
    if recorded != actual:
        raise ValueError("Fit manifest snapshot hash is invalid")


def _patchcore_bundle_statuses(
    comparison_status: Any,
    allow_preregistered_additional: bool,
) -> tuple[str, str, str]:
    """Return the immutable status contract allowed for this consumer."""
    if comparison_status == EXPECTED_COMPARISON_STATUS:
        return (
            "fit_calibrated_exploratory",
            "evaluated_exploratory",
            "exploratory_rules_frozen_before_test_images_loaded",
        )
    if (
        allow_preregistered_additional
        and comparison_status == EXPECTED_PREREGISTERED_STATUS
    ):
        return (
            "fit_calibrated_preregistered",
            "evaluated_preregistered",
            "preregistered_rules_frozen_before_test_images_loaded",
        )
    raise ValueError("PatchCore bundle comparison status is not allowed here")


def validate_inference_bundles(
    fit_path: Path,
    evaluation_path: Path,
    project_root: Path,
    *,
    allow_preregistered_additional: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Validate immutable PaDiM or PatchCore bundles without reopening test data."""
    fit_run = read_json(_safe_named_file(fit_path, "run.json"))
    evaluation = read_json(_safe_named_file(evaluation_path, "evaluation.json"))
    model_name = fit_run.get("config", {}).get("model")
    if model_name == "padim":
        expected_fit_status = "fit_calibrated"
        expected_evaluation_status = "evaluated"
        expected_calibration_status = "frozen_before_test_images_loaded"
        fit_config_name = "pilot.toml"
        evaluation_config_name = "evaluation-v1.toml"
    elif model_name == "patchcore":
        fit_config_name = "patchcore-pilot.toml"
        evaluation_config_name = "patchcore-evaluation-v1.toml"
        comparison_status = fit_run.get("comparison_status")
        (
            expected_fit_status,
            expected_evaluation_status,
            expected_calibration_status,
        ) = _patchcore_bundle_statuses(
            comparison_status, allow_preregistered_additional
        )
        if evaluation.get("comparison_status") != comparison_status:
            raise ValueError("PatchCore fit and evaluation comparison statuses differ")
    else:
        raise ValueError(f"Unsupported inference bundle model: {model_name!r}")
    if (
        fit_run.get("status") != expected_fit_status
        or fit_run.get("test_evaluated") is not False
    ):
        raise ValueError("Fit bundle is not a completed calibration-only run")
    if (
        evaluation.get("status") != expected_evaluation_status
        or evaluation.get("test_evaluated") is not True
    ):
        raise ValueError("Evaluation bundle is not complete")
    if evaluation.get("evaluation_id") != evaluation_path.name:
        raise ValueError("Evaluation ID differs from its directory name")

    manifest = read_json(_safe_named_file(fit_path, "manifest.json"))
    _verify_manifest_hash(manifest)
    if (
        not allow_preregistered_additional
        and manifest.get("categories") != ["transistor"]
    ):
        raise ValueError("The inspection service accepts transistor bundles only")
    if fit_run.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Fit run and manifest hashes differ")
    evaluation_manifest = read_json(_safe_named_file(evaluation_path, "manifest.json"))
    if canonical(evaluation_manifest) != canonical(manifest):
        raise ValueError("Evaluation and fit manifest snapshots differ")
    fit_snapshot = read_json(_safe_named_file(evaluation_path, "fit-run.json"))
    if fit_snapshot != fit_run:
        raise ValueError("Evaluation fit-run snapshot differs from the source fit bundle")

    model_seed = fit_run.get("config", {}).get("model_seed")
    fit_config_path = _safe_named_file(fit_path, fit_config_name)
    if model_name == "padim":
        fit_config = load_experiment(fit_config_path, manifest, model_seed)
    else:
        fit_config = load_patchcore_experiment(fit_config_path, manifest, model_seed)
    if fit_run.get("config") != fit_config:
        raise ValueError("Fit configuration snapshot differs from run.json")
    evaluation_config_path = _safe_named_file(evaluation_path, evaluation_config_name)
    if model_name == "padim":
        evaluation_config = load_evaluation_config(evaluation_config_path)
    else:
        evaluation_config = load_patchcore_evaluation_config(evaluation_config_path)
    if evaluation.get("evaluation_config") != evaluation_config:
        raise ValueError("Evaluation configuration snapshot differs from evaluation.json")

    fit_commit = fit_run.get("git", {}).get("commit")
    evaluation_commit = evaluation.get("evaluator_git", {}).get("commit")
    _require_ancestor(project_root, fit_commit, "Fit")
    _require_ancestor(project_root, evaluation_commit, "Evaluation")
    if evaluation.get("fit", {}).get("run_id") != fit_run.get("run_id"):
        raise ValueError("Evaluation references a different fit run")
    if evaluation.get("fit", {}).get("git_commit") != fit_commit:
        raise ValueError("Evaluation references a different fit commit")
    if evaluation.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Evaluation references a different manifest")

    fit_artifacts = fit_run.get("artifacts", {})
    checkpoint_path = _safe_named_file(fit_path, fit_artifacts.get("checkpoint"))
    checkpoint_sha256 = checksum(checkpoint_path)
    if checkpoint_sha256 != fit_artifacts.get("checkpoint_sha256"):
        raise ValueError("Checkpoint checksum differs from fit run")
    if checkpoint_sha256 != evaluation.get("fit", {}).get("checkpoint_sha256"):
        raise ValueError("Evaluation references a different checkpoint")

    evaluation_artifacts = evaluation.get("artifacts", {})
    calibration_path = _safe_named_file(
        evaluation_path, evaluation_artifacts.get("calibration_contract")
    )
    if checksum(calibration_path) != evaluation_artifacts.get("calibration_contract_sha256"):
        raise ValueError("Calibration contract checksum differs from evaluation.json")
    calibration_contract = read_json(calibration_path)
    if calibration_contract != evaluation.get("calibration_contract"):
        raise ValueError("Calibration contract differs from evaluation.json")
    if calibration_contract.get("status") != expected_calibration_status:
        raise ValueError("Calibration contract was not frozen before test evaluation")
    if model_name == "patchcore" and (
        calibration_contract.get("comparison_status") != fit_run.get("comparison_status")
    ):
        raise ValueError("PatchCore calibration contract lost its comparison status")
    if calibration_contract.get("fit_run_id") != fit_run.get("run_id"):
        raise ValueError("Calibration contract references a different fit")
    if float(calibration_contract.get("image", {}).get("threshold", math.nan)) != float(
        fit_run.get("calibration", {}).get("threshold", math.nan)
    ):
        raise ValueError("Image threshold differs from the fit calibration threshold")
    pixel_contract = calibration_contract.get("pixel", {})
    expected_pixel_contract = {
        "quantile_method": "higher",
        "decision_rule": "score >= threshold",
        "morphology": "none",
        "connectivity": 4,
        "minimum_region_area_pixels": 1,
    }
    for field, expected in expected_pixel_contract.items():
        if pixel_contract.get(field) != expected:
            raise ValueError(f"Unexpected pixel calibration contract field: {field}")
    for level in ("image", "pixel"):
        threshold = calibration_contract.get(level, {}).get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError(f"Calibration contract has no finite {level} threshold")
    return fit_run, fit_config, calibration_contract, checkpoint_path


def _load_validated_png(
    path: Path,
    expected_size: tuple[int, int] | None = (1024, 1024),
) -> tuple[np.ndarray, tuple[int, int]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise InputValidationError("Input must be a non-empty file")
    if path.suffix.lower() != ".png":
        raise InputValidationError("The first inspection contract accepts PNG files only")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise InputValidationError("Input PNG exceeds the 50 MiB safety limit")
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise InputValidationError("Input content is not PNG")
            if image.mode != "RGB":
                raise InputValidationError(f"Expected RGB input, found {image.mode}")
            if expected_size is not None and image.size != expected_size:
                raise InputValidationError(
                    f"Expected {expected_size[0]}x{expected_size[1]} input, found {image.size}"
                )
            image.load()
            return np.array(image, dtype=np.uint8, copy=True), image.size
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"PNG decode failed: {exc}") from exc


def load_inference_runtime(
    project_root: Path,
    fit_path: Path,
    evaluation_path: Path,
    *,
    allow_preregistered_additional: bool = False,
) -> PadimRuntime:
    """Validate the immutable bundles and load one reusable CUDA model."""
    setup_started = time.perf_counter()
    current_git = git_state(project_root)
    fit_run, fit_config, calibration_contract, checkpoint_path = validate_inference_bundles(
        fit_path,
        evaluation_path,
        project_root,
        allow_preregistered_additional=allow_preregistered_additional,
    )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The lab inspection requires Python 3.12, found {platform.python_version()}"
        )
    versions = installed_versions()

    import torch
    import torch.nn.functional as functional
    import torchvision
    from anomalib.models import Padim, Patchcore

    if (
        torch.__version__ != EXPECTED_PACKAGES["torch"]
        or torchvision.__version__ != EXPECTED_PACKAGES["torchvision"]
    ):
        raise RuntimeError(
            "Imported torch/torchvision versions differ from the validated combination"
        )
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("Inspection requires the validated CUDA 12.8 GPU environment")

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
        or checkpoint.get("manifest_sha256") != fit_run["manifest_sha256"]
        or checkpoint.get("git_commit") != fit_run["git"]["commit"]
        or float(checkpoint.get("threshold"))
        != float(calibration_contract["image"]["threshold"])
    ):
        raise ValueError("Checkpoint metadata differs from the validated bundles")
    if fit_config["model"] == "padim":
        model = Padim(
            backbone=fit_config["backbone"],
            layers=fit_config["layers"],
            pre_trained=False,
            n_features=fit_config["n_features"],
            pre_processor=False,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    elif fit_config["model"] == "patchcore":
        if (
            checkpoint.get("memory_bank_shape")
            != fit_run.get("memory_bank", {}).get("shape")
        ):
            raise ValueError("PatchCore checkpoint memory-bank metadata differs")
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
            raise ValueError("PatchCore memory-bank tensor shape differs")
        model.model.memory_bank = torch.empty_like(saved_memory_bank)
        model.load_state_dict(state_dict, strict=True)
    else:  # pragma: no cover - bundle validation rejects this first
        raise ValueError(f"Unsupported inference model: {fit_config['model']!r}")
    model.to(device)
    model._is_fitted.fill_(True)
    model.model.eval()
    return PadimRuntime(
        current_git=current_git,
        fit_path=fit_path.resolve(strict=True),
        evaluation_path=evaluation_path.resolve(strict=True),
        fit_run=fit_run,
        fit_config=fit_config,
        calibration_contract=calibration_contract,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checksum(checkpoint_path),
        versions=versions,
        torch=torch,
        functional=functional,
        model=model,
        device=device,
        setup_ms=(time.perf_counter() - setup_started) * 1000.0,
    )


def predict_validated_png(
    runtime: PadimRuntime,
    input_path: Path,
    *,
    expected_size: tuple[int, int] | None = (1024, 1024),
) -> dict[str, Any]:
    """Decode, preprocess and score one image without reloading the model."""
    sample_started = time.perf_counter()
    decode_started = time.perf_counter()
    original, original_size = _load_validated_png(input_path, expected_size)
    input_sha256 = checksum(input_path)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    torch = runtime.torch
    functional = runtime.functional
    model_size = runtime.fit_config["image_size"]
    preprocess_started = time.perf_counter()
    tensor = torch.from_numpy(original).permute(2, 0, 1).to(torch.float32).div_(255.0)
    tensor = functional.interpolate(
        tensor.unsqueeze(0),
        size=(model_size, model_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    image_batch = ((tensor - mean) / std).unsqueeze(0).to(runtime.device)
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

    inference_started = time.perf_counter()
    with torch.no_grad():
        predictions = runtime.model.model(image_batch)
    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    score_values = predictions.pred_score.detach().cpu().reshape(-1)
    if score_values.numel() != 1:
        raise RuntimeError("Expected one score for one inspection image")
    raw_score = float(score_values.item())
    map_tensor = predictions.anomaly_map.detach().cpu()
    if map_tensor.ndim == 4 and map_tensor.shape[:2] == (1, 1):
        map_tensor = map_tensor[0, 0]
    elif map_tensor.ndim == 3 and map_tensor.shape[0] == 1:
        map_tensor = map_tensor[0]
    elif map_tensor.ndim != 2:
        raise RuntimeError(f"Unexpected anomaly map shape: {tuple(map_tensor.shape)}")
    anomaly_map = map_tensor.numpy().astype(np.float32, copy=True)
    if anomaly_map.shape != (model_size, model_size) or not np.isfinite(anomaly_map).all():
        raise RuntimeError("Model returned an invalid anomaly map")
    if not math.isfinite(raw_score):
        raise RuntimeError("Model returned a non-finite image score")
    return {
        "original": original,
        "original_size": original_size,
        "input_sha256": input_sha256,
        "raw_score": raw_score,
        "anomaly_map": anomaly_map,
        "latency_ms": {
            "decode_and_checksum": decode_ms,
            "preprocess": preprocess_ms,
            "gpu_inference_and_map_transfer": inference_ms,
            "sample_total": (time.perf_counter() - sample_started) * 1000.0,
        },
    }


def validate_expected_model(actual_model: str, expected_model: str | None) -> None:
    """Prevent a model-specific CLI from silently loading the wrong bundle type."""
    supported = {"padim", "patchcore"}
    if actual_model not in supported:
        raise ValueError(f"Unsupported inspection model: {actual_model!r}")
    if expected_model is not None and expected_model not in supported:
        raise ValueError(f"Unsupported expected inspection model: {expected_model!r}")
    if expected_model is not None and actual_model != expected_model:
        raise ValueError(
            f"This command requires a {expected_model} fit bundle, found {actual_model}"
        )


def inspection_limitations(model: str) -> list[str]:
    """Return model-specific limitations for the frozen transistor baselines."""
    common = [
        "The anomaly score is not a defect probability, severity or root-cause label.",
        "PASS is a model output for review workflow design, not a safety certification.",
    ]
    if model == "padim":
        return [
            common[0],
            "The frozen PaDiM threshold recalled only 22 of 40 anomalies in the test set.",
            common[1],
        ]
    if model == "patchcore":
        return [
            common[0],
            "The frozen PatchCore threshold recalled 37 of 40 anomalies in the test set.",
            (
                "This PatchCore configuration is exploratory because it was fixed after "
                "the PaDiM test result had been observed."
            ),
            common[1],
        ]
    raise ValueError(f"Unsupported inspection model: {model!r}")


def _execute_inspection(
    output_path: Path,
    project_root: Path,
    fit_path: Path,
    evaluation_path: Path,
    input_path: Path,
    expected_model: str | None,
    runtime: PadimRuntime | None = None,
    input_source: str | None = None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    model_reused = runtime is not None
    if runtime is None:
        runtime = load_inference_runtime(project_root, fit_path, evaluation_path)
    elif (
        runtime.fit_path != fit_path.resolve(strict=True)
        or runtime.evaluation_path != evaluation_path.resolve(strict=True)
    ):
        raise ValueError("Resident runtime does not match the requested fit/evaluation")
    model_name = runtime.fit_config["model"]
    validate_expected_model(model_name, expected_model)
    prediction = predict_validated_png(runtime, input_path)
    original = prediction["original"]
    original_size = prediction["original_size"]
    input_sha256 = prediction["input_sha256"]
    raw_score = prediction["raw_score"]
    anomaly_map = prediction["anomaly_map"]
    prediction_latency = prediction["latency_ms"]

    postprocess_started = time.perf_counter()
    image_threshold = float(runtime.calibration_contract["image"]["threshold"])
    pixel_threshold = float(runtime.calibration_contract["pixel"]["threshold"])
    if pixel_threshold <= 0:
        raise ValueError("Pixel threshold must be positive for the fixed display scale")
    decision = "REVIEW_REQUIRED" if raw_score >= image_threshold else "PASS"
    predicted_mask = anomaly_map >= pixel_threshold
    regions = []
    for component in connected_components_4(predicted_mask):
        map_box = component["bbox_xyxy_exclusive"]
        regions.append(
            {
                "area_map_pixels": component["area"],
                "bbox_map_xyxy_exclusive": map_box,
                "bbox_original_xyxy_exclusive": scale_bbox_to_original(
                    map_box, anomaly_map.shape, original_size
                ),
            }
        )

    normalized_map = np.clip(anomaly_map / pixel_threshold, 0.0, 1.0)
    normalized_image = Image.fromarray(np.rint(normalized_map * 255.0).astype(np.uint8))
    normalized_original = np.asarray(
        normalized_image.resize(original_size, resample=Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    heatmap_array = colorize_anomaly_values(normalized_original)
    alpha = (0.15 + 0.45 * normalized_original)[..., None]
    overlay_array = np.rint(original * (1.0 - alpha) + heatmap_array * alpha).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_array)
    draw = ImageDraw.Draw(overlay_image)
    line_width = max(2, original_size[0] // 512)
    for region in regions:
        x_min, y_min, x_max, y_max = region["bbox_original_xyxy_exclusive"]
        draw.rectangle(
            [x_min, y_min, max(x_min, x_max - 1), max(y_min, y_max - 1)],
            outline=(255, 0, 0),
            width=line_width,
        )

    input_copy_path = output_path / "input.png"
    heatmap_path = output_path / "heatmap.png"
    overlay_path = output_path / "overlay.png"
    raw_map_path = output_path / "anomaly-map.npz"
    shutil.copyfile(input_path, input_copy_path)
    Image.fromarray(heatmap_array).save(heatmap_path, format="PNG")
    overlay_image.save(overlay_path, format="PNG")
    with raw_map_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            anomaly_map=anomaly_map,
            image_threshold=np.asarray(image_threshold, dtype=np.float64),
            pixel_threshold=np.asarray(pixel_threshold, dtype=np.float64),
        )
    input_copy_sha256 = checksum(input_copy_path)
    heatmap_sha256 = checksum(heatmap_path)
    overlay_sha256 = checksum(overlay_path)
    raw_map_sha256 = checksum(raw_map_path)
    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0

    result = {
        "schema_version": 1,
        "status": "completed",
        "inspection_id": output_path.name,
        "category": "transistor",
        "model": model_name,
        "comparison_status": runtime.fit_run.get("comparison_status"),
        "model_reused": model_reused,
        "decision": decision,
        "automatic_pass": False,
        "automatic_pass_allowed": False,
        "raw_score": raw_score,
        "image_threshold": image_threshold,
        "decision_rule": "score >= threshold",
        "pixel": {
            "threshold": pixel_threshold,
            "score_space": "raw",
            "anomalous_map_pixels": int(np.count_nonzero(predicted_mask)),
            "anomalous_map_fraction": float(np.mean(predicted_mask)),
            "region_count": len(regions),
            "regions": regions,
            "morphology": "none",
            "connectivity": 4,
            "bbox_coordinates": "xyxy_exclusive",
        },
        "display": {
            "normalization": "fixed linear clip",
            "minimum_raw_score": 0.0,
            "maximum_raw_score": pixel_threshold,
            "per_image_min_max": False,
            "overlay_alpha": "0.15 + 0.45 * normalized_score",
        },
        "input": {
            "source_path": input_source or str(input_path),
            "filename": input_path.name,
            "sha256": input_sha256,
            "format": "PNG",
            "mode": "RGB",
            "size_wh": list(original_size),
        },
        "trace": {
            "model": model_name,
            "comparison_status": runtime.fit_run.get("comparison_status"),
            "inspection_git": runtime.current_git,
            "fit_run_id": runtime.fit_run["run_id"],
            "fit_git_commit": runtime.fit_run["git"]["commit"],
            "evaluation_id": evaluation_path.name,
            "manifest_sha256": runtime.fit_run["manifest_sha256"],
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "model_seed": runtime.fit_config["model_seed"],
        },
        "latency_ms": {
            "setup_and_model_load": 0.0 if model_reused else runtime.setup_ms,
            "decode": prediction_latency["decode_and_checksum"],
            "preprocess": prediction_latency["preprocess"],
            "gpu_inference_and_map_transfer": prediction_latency[
                "gpu_inference_and_map_transfer"
            ],
            "postprocess_and_artifact_write": postprocess_ms,
            "total": total_ms,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": runtime.versions,
            "torch_cuda_runtime": runtime.torch.version.cuda,
            "gpu": runtime.torch.cuda.get_device_name(),
        },
        "artifacts": {
            "input": input_copy_path.name,
            "input_sha256": input_copy_sha256,
            "heatmap": heatmap_path.name,
            "heatmap_sha256": heatmap_sha256,
            "overlay": overlay_path.name,
            "overlay_sha256": overlay_sha256,
            "anomaly_map": raw_map_path.name,
            "anomaly_map_sha256": raw_map_sha256,
        },
        "limitations": inspection_limitations(model_name),
    }
    return result


def run_inspection(
    args: argparse.Namespace,
    *,
    expected_model: str | None = None,
    runtime: PadimRuntime | None = None,
    input_source: str | None = None,
) -> dict[str, Any]:
    """Create one immutable inspection bundle, including explicit failure records."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts").resolve()
    fit_root = (artifacts_root / "runs").resolve()
    evaluation_root = (artifacts_root / "evaluations").resolve()
    inspection_root = (artifacts_root / "inspections").resolve()
    output_path = args.output.resolve()
    output_path.relative_to(inspection_root)
    if output_path.exists():
        raise FileExistsError(
            f"Inspection output already exists; choose a new --output: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    requested_input_path = args.input.resolve()
    try:
        fit_path = args.fit_run.resolve(strict=True)
        evaluation_path = args.evaluation.resolve(strict=True)
        input_path = args.input.resolve(strict=True)
        fit_path.relative_to(fit_root)
        evaluation_path.relative_to(evaluation_root)
        result = _execute_inspection(
            output_path,
            project_root,
            fit_path,
            evaluation_path,
            input_path,
            expected_model,
            runtime,
            input_source,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "INSPECTION_FAILED",
            "inspection_id": output_path.name,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "input_path": input_source or str(requested_input_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "automatic_pass": False,
            "automatic_pass_allowed": False,
        }
        write_json(output_path / "inspection.json", failure)
        raise
    result["started_at"] = started_at.isoformat()
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_path / "inspection.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the single-image inspection command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One 1024x1024 RGB PNG")
    parser.add_argument("--fit-run", type=Path, required=True, help="Completed fit directory")
    parser.add_argument(
        "--evaluation", type=Path, required=True, help="Completed evaluation directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory below artifacts/inspections",
    )
    return parser


def main(
    argv: list[str] | None = None, *, expected_model: str | None = None
) -> int:
    """Run one inspection and print a compact machine-readable result."""
    args = build_parser().parse_args(argv)
    try:
        result = run_inspection(args, expected_model=expected_model)
    except Exception as exc:  # noqa: BLE001 - CLI boundary records failure before returning
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "inspection_id": result["inspection_id"],
                "model": result["model"],
                "comparison_status": result["comparison_status"],
                "decision": result["decision"],
                "raw_score": result["raw_score"],
                "image_threshold": result["image_threshold"],
                "region_count": result["pixel"]["region_count"],
                "total_latency_ms": result["latency_ms"]["total"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
