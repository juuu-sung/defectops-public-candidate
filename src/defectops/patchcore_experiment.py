"""Manifest-locked exploratory PatchCore fit and normal-only calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defectops.manifest import verify_manifest
from defectops.padim_experiment import (
    EXPECTED_PACKAGES,
    EXPECTED_PROTOCOL,
    checksum,
    git_state,
    higher_quantile,
    initialize_cuda_device,
    installed_versions,
    read_json,
    records_for_split,
    write_json,
)


EXPECTED_COMPARISON_STATUS = "exploratory_after_padim_test_observed"


def load_patchcore_experiment(
    config_path: Path,
    manifest: dict[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    """Load the fixed exploratory PatchCore settings."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = config.get("experiment", {})
    calibration = config.get("calibration", {})
    benchmark = config.get("benchmark", {})
    checks = {
        "protocol": config.get("protocol") == EXPECTED_PROTOCOL,
        "manifest_protocol": manifest.get("protocol") == EXPECTED_PROTOCOL,
        "comparison_status": config.get("comparison_status") == EXPECTED_COMPARISON_STATUS,
        "split_seed": config.get("split_seed") == manifest.get("split_seed"),
        "categories": config.get("categories") == manifest.get("categories") == ["transistor"],
        "model": experiment.get("model") == "patchcore",
        "backbone": experiment.get("backbone") == "wide_resnet50_2",
        "layers": experiment.get("layers") == ["layer2", "layer3"],
        "quantile_method": calibration.get("quantile_method") == "higher",
        "score_space": calibration.get("score_space") == "raw",
        "decision_rule": calibration.get("decision_rule") == "score >= threshold",
        "batch_size": benchmark.get("batch_size") == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("PatchCore configuration contract mismatch: " + ", ".join(failed))
    model_seeds = experiment.get("model_seeds")
    if not isinstance(model_seeds, list) or model_seeds != [42]:
        raise ValueError("PatchCore pilot model_seeds must be [42]")
    selected_seed = model_seeds[0] if seed is None else seed
    if selected_seed not in model_seeds:
        raise ValueError(f"seed must be one of the configured model seeds: {model_seeds}")
    image_quantile = calibration.get("image_quantile")
    coreset_ratio = experiment.get("coreset_sampling_ratio")
    num_neighbors = experiment.get("num_neighbors")
    image_size = experiment.get("image_size")
    if not isinstance(image_quantile, (int, float)) or not 0 < float(image_quantile) < 1:
        raise ValueError("calibration.image_quantile must be between zero and one")
    if not isinstance(coreset_ratio, (int, float)) or not 0 < float(coreset_ratio) <= 1:
        raise ValueError("experiment.coreset_sampling_ratio must be in (0, 1]")
    if type(num_neighbors) is not int or num_neighbors < 1:
        raise ValueError("experiment.num_neighbors must be a positive integer")
    if type(image_size) is not int or image_size != 256:
        raise ValueError("PatchCore exploratory comparison is locked to image_size=256")
    return {
        "model": "patchcore",
        "comparison_status": EXPECTED_COMPARISON_STATUS,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": float(coreset_ratio),
        "num_neighbors": num_neighbors,
        "image_size": image_size,
        "model_seed": selected_seed,
        "split_seed": int(config["split_seed"]),
        "batch_size": 1,
        "num_workers": 0,
        "image_quantile": float(image_quantile),
        "quantile_method": "higher",
        "score_space": "raw",
        "decision_rule": "score >= threshold",
        "pre_trained": True,
        "resize": "stretch_to_square_bilinear_antialias",
        "normalization": "imagenet_mean_std",
        "augmentation": "none",
    }


def _safe_data_path(data_root: Path, relative_path: str) -> Path:
    resolved_root = data_root.resolve(strict=True)
    path = (resolved_root / relative_path).resolve(strict=True)
    path.relative_to(resolved_root)
    return path


def run_patchcore_fit(args: argparse.Namespace) -> dict[str, Any]:
    """Build a PatchCore memory bank and calibrate a normal-only image threshold."""
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    runs_root = (project_root / "artifacts" / "runs").resolve()
    output_path = args.output.resolve()
    output_path.relative_to(runs_root)
    if output_path.exists():
        raise FileExistsError(f"PatchCore run output already exists: {output_path}")

    manifest = read_json(manifest_path)
    verify_manifest(data_root, manifest)
    experiment = load_patchcore_experiment(config_path, manifest, args.seed)
    train_records = records_for_split(manifest, "train")
    calibration_records = records_for_split(manifest, "calibration")
    test_records = records_for_split(manifest, "test")
    all_paths = [
        record["image_path"]
        for records in (train_records, calibration_records, test_records)
        for record in records
    ]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Manifest splits overlap")
    git = git_state(project_root)
    versions = installed_versions()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The lab PatchCore fit requires Python 3.12, found {platform.python_version()}"
        )

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
        raise RuntimeError("PatchCore fit requires the validated CUDA 12.8 GPU environment")

    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output_path / "patchcore-pilot.toml")
    shutil.copyfile(manifest_path, output_path / "manifest.json")
    seed = experiment["model_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = initialize_cuda_device(torch)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def load_image(record: dict[str, Any]) -> Any:
        image_path = _safe_data_path(data_root, record["image_path"])
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (1024, 1024):
                raise ValueError(f"Unexpected PatchCore input contract: {record['image_path']}")
            array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        tensor = functional.interpolate(
            tensor.unsqueeze(0),
            size=(experiment["image_size"], experiment["image_size"]),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return (tensor - mean) / std

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    model = Patchcore(
        backbone=experiment["backbone"],
        layers=experiment["layers"],
        pre_trained=True,
        coreset_sampling_ratio=experiment["coreset_sampling_ratio"],
        num_neighbors=experiment["num_neighbors"],
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).to(device)
    model.model.train()
    model.model.feature_extractor.eval()
    print(f"Collecting PatchCore embeddings from {len(train_records)} frozen normal images...")
    with torch.no_grad():
        for index, record in enumerate(train_records, start=1):
            model.model(load_image(record).unsqueeze(0).to(device))
            if index % 10 == 0 or index == len(train_records):
                print(f"  train {index}/{len(train_records)}")
    embedding_count = sum(int(value.shape[0]) for value in model.model.embedding_store)
    print(
        f"Selecting {experiment['coreset_sampling_ratio']:.0%} coreset "
        f"from {embedding_count} patch embeddings..."
    )
    model.fit()
    model._is_fitted.fill_(True)
    model.model.eval()
    memory_bank_shape = list(model.model.memory_bank.shape)
    memory_bank_bytes = model.model.memory_bank.nelement() * model.model.memory_bank.element_size()
    if not memory_bank_shape or memory_bank_shape[0] < experiment["num_neighbors"]:
        raise RuntimeError("PatchCore memory bank is unexpectedly small")

    print(f"Scoring {len(calibration_records)} frozen normal calibration images...")
    calibration_rows = []
    with torch.no_grad():
        for index, record in enumerate(calibration_records, start=1):
            predictions = model.model(load_image(record).unsqueeze(0).to(device))
            score_values = predictions.pred_score.detach().cpu().reshape(-1)
            if score_values.numel() != 1:
                raise RuntimeError("Expected one PatchCore calibration score")
            score = float(score_values.item())
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite calibration score: {record['image_path']}")
            calibration_rows.append({"image_path": record["image_path"], "raw_score": score})
            if index % 10 == 0 or index == len(calibration_records):
                print(f"  calibration {index}/{len(calibration_records)}")
    threshold = higher_quantile(
        [row["raw_score"] for row in calibration_rows], experiment["image_quantile"]
    )
    for row in calibration_rows:
        row["decision"] = "REVIEW_REQUIRED" if row["raw_score"] >= threshold else "PASS"

    checkpoint_path = output_path / "model.pt"
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "format_version": 1,
            "model": experiment,
            "manifest_sha256": manifest["manifest_sha256"],
            "git_commit": git["commit"],
            "state_dict": cpu_state,
            "threshold": threshold,
            "memory_bank_shape": memory_bank_shape,
        },
        checkpoint_path,
    )
    calibration_path = output_path / "calibration-scores.jsonl"
    with calibration_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in calibration_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    elapsed_seconds = time.perf_counter() - start_time
    finished_at = datetime.now(timezone.utc)
    run = {
        "schema_version": 1,
        "status": "fit_calibrated_exploratory",
        "comparison_status": EXPECTED_COMPARISON_STATUS,
        "test_evaluated": False,
        "purpose": (
            "exploratory PatchCore fit and normal-only calibration after PaDiM test observation"
        ),
        "run_id": output_path.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "git": git,
        "manifest_sha256": manifest["manifest_sha256"],
        "config": experiment,
        "counts": {
            "train": len(train_records),
            "calibration": len(calibration_records),
            "test_reserved_not_loaded_by_this_fit": len(test_records),
        },
        "sample_ids": {
            "train": [record["image_path"] for record in train_records],
            "calibration": [record["image_path"] for record in calibration_records],
        },
        "memory_bank": {
            "embedding_count_before_coreset": embedding_count,
            "coreset_sampling_ratio": experiment["coreset_sampling_ratio"],
            "shape": memory_bank_shape,
            "bytes": memory_bank_bytes,
            "frozen_feature_extractor_mode": "eval",
        },
        "calibration": {
            "threshold": threshold,
            "quantile": experiment["image_quantile"],
            "method": experiment["quantile_method"],
            "score_space": "raw",
            "decision_rule": "score >= threshold",
            "minimum_raw_score": min(row["raw_score"] for row in calibration_rows),
            "maximum_raw_score": max(row["raw_score"] for row in calibration_rows),
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
            "pid": os.getpid(),
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checksum(checkpoint_path),
            "calibration_scores": calibration_path.name,
            "calibration_scores_sha256": checksum(calibration_path),
            "config_snapshot": "patchcore-pilot.toml",
            "manifest_snapshot": "manifest.json",
        },
        "limitations": [
            (
                "The PaDiM official test result was observed before this PatchCore "
                "configuration was fixed."
            ),
            (
                "This fit is eligible only for exploratory comparison, not confirmatory "
                "model selection."
            ),
            "The official test images were not loaded by this fit command.",
        ],
    }
    write_json(output_path / "run.json", run)
    print(
        json.dumps(
            {
                "status": run["status"],
                "run_id": run["run_id"],
                "threshold": threshold,
                "memory_bank_shape": memory_bank_shape,
                "elapsed_seconds": elapsed_seconds,
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return run


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset parent directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "transistor-v1.json",
    )
    parser.add_argument(
        "--config", type=Path, default=project_root / "configs" / "patchcore-pilot.toml"
    )
    parser.add_argument("--seed", type=int, help="Configured model seed")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory below artifacts/runs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_patchcore_fit(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
