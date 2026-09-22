"""Run preregistered Real-IAD development candidates without opening final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from defectops.evaluation_metrics import (
    aupro_fpr_grid,
    binary_auroc,
    binary_average_precision,
    confusion_counts,
    rates_from_confusion,
    wilson_interval,
)
from defectops.manifest import canonical, verify_manifest
from defectops.padim_evaluation import _extract_prediction, _fixed_image_metrics
from defectops.padim_experiment import (
    EXPECTED_PACKAGES,
    checksum,
    git_state,
    higher_quantile,
    initialize_cuda_device,
    installed_versions,
    read_json,
    records_for_split,
    write_json,
)
from defectops.realiad_followup_protocol import (
    FOLLOWUP_CATEGORIES,
    load_realiad_followup_protocol,
)
from defectops.realiad_followup_registration import verify_followup_final_manifest


EXPECTED_PURPOSE = (
    "preregistered_development_candidate_selection_before_final_holdout_opened"
)
EXPECTED_PLAN_SHA256 = (
    "c68562288b378d44d969c3eda68a69a00c6c31e8dad0f4ff21e5c05073ac8f20"
)
EXPECTED_MANIFESTS = {
    "realiad_pcb": {
        "development_sha256": (
            "2d16e21b2e665defbf54266c1d5e8b08f41838dfae679fff8f65dff615b49ed4"
        ),
        "final_sha256": (
            "9b5a319600967c8e73bf3e4b53d0b31c9e7df9ff7261222c31db489b435478d2"
        ),
    },
    "realiad_terminalblock": {
        "development_sha256": (
            "a5f6e45935e2a16f64a23f4757fa383f569efec84b25e7ed3ddf197456c2b338"
        ),
        "final_sha256": (
            "b2ff0bc8f52aeed8db97ce853869ecc9ec6a46d0bc939b1c4442a7d31507891b"
        ),
    },
}


def load_followup_development_execution(path: Path) -> dict[str, Any]:
    """Load the post-registration execution contract fixed before development scoring."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema_version": config.get("schema_version") == 1,
        "protocol": config.get("protocol") == "A-normal-only-v2",
        "purpose": config.get("purpose") == EXPECTED_PURPOSE,
        "plan_sha256": config.get("plan_sha256") == EXPECTED_PLAN_SHA256,
        "development_seed": config.get("development_seed") == 42,
        "pixel_ground_truth_threshold": config.get("pixel_ground_truth_threshold") == 128,
        "aupro_fpr_limit": config.get("aupro_fpr_limit") == 0.30,
        "aupro_fpr_points": config.get("aupro_fpr_points") == 301,
        "confidence_level": config.get("confidence_level") == 0.95,
        "benchmark": config.get("benchmark")
        == {
            "batch_size": 1,
            "warmup": 5,
            "repeats": 1,
            "count": 100,
            "scope": (
                "image decode, preprocessing, GPU inference, map transfer, "
                "candidate image score"
            ),
        },
        "manifests": config.get("manifests") == EXPECTED_MANIFESTS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Real-IAD follow-up development execution mismatch: " + ", ".join(failed)
        )
    config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    return config


def candidate_image_score(
    raw_score: float,
    anomaly_map: np.ndarray,
    candidate: dict[str, Any],
) -> float:
    """Return the candidate's preregistered image score."""
    if not math.isfinite(raw_score):
        raise ValueError("raw PatchCore score must be finite")
    score_kind = candidate.get("image_score")
    if score_kind == "anomalib_patchcore_raw":
        if candidate.get("top_fraction") != 0.0:
            raise ValueError("raw PatchCore candidate must use top_fraction=0")
        return float(raw_score)
    if score_kind != "mean_of_top_map_fraction":
        raise ValueError(f"Unexpected candidate score: {score_kind}")
    fraction = candidate.get("top_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < float(fraction) <= 1:
        raise ValueError("top-map candidate fraction must be in (0, 1]")
    values = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("anomaly map must contain finite values")
    count = max(1, math.ceil(values.size * float(fraction)))
    selected = np.partition(values, values.size - count)[-count:]
    return float(np.mean(selected, dtype=np.float64))


def _safe_data_path(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    path.relative_to(root)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _assert_final_not_extracted(data_root: Path, final_manifest: dict[str, Any]) -> None:
    """Reject development scoring if any sealed final image or mask is on disk."""
    if final_manifest.get("sealed") is not True or final_manifest.get("images_extracted") is not False:
        raise ValueError("Final manifest is not sealed")
    extracted_root = data_root.resolve(strict=True) / "real_iad_followup" / "images"
    for row in final_manifest["records"]:
        for key in ("source_image_member", "source_mask_member"):
            member = row.get(key)
            if member is not None and (extracted_root / member).exists():
                raise RuntimeError(f"Refusing to continue: sealed final member is extracted: {member}")


def _load_category_contract(
    data_root: Path,
    manifest_path: Path,
    final_manifest_path: Path,
    protocol_config_path: Path,
    execution_config_path: Path,
    category: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_realiad_followup_protocol(protocol_config_path)
    execution = load_followup_development_execution(execution_config_path)
    if category not in FOLLOWUP_CATEGORIES:
        raise ValueError(f"Unexpected follow-up category: {category}")
    manifest = read_json(manifest_path)
    final_manifest = read_json(final_manifest_path)
    expected = execution["manifests"][category]
    if (
        manifest.get("manifest_sha256") != expected["development_sha256"]
        or final_manifest.get("manifest_sha256") != expected["final_sha256"]
        or manifest.get("split_plan_sha256") != execution["plan_sha256"]
        or final_manifest.get("split_plan_sha256") != execution["plan_sha256"]
        or manifest.get("categories") != [category]
        or final_manifest.get("categories") != [category]
        or manifest.get("final_holdout_included") is not False
    ):
        raise ValueError("Development/final manifests differ from the execution contract")
    verify_manifest(data_root, manifest)
    verify_followup_final_manifest(data_root, final_manifest)
    _assert_final_not_extracted(data_root, final_manifest)
    candidate_ids = [candidate["id"] for candidate in protocol["candidates"]]
    if len(candidate_ids) != 4 or len(set(candidate_ids)) != 4:
        raise ValueError("Expected exactly four unique preregistered candidates")
    return protocol, execution, manifest, final_manifest


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _candidate_metrics(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    threshold: float,
    confidence_level: float,
) -> dict[str, Any]:
    labels = np.asarray([record["label"] for record in records], dtype=np.bool_)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    counts = confusion_counts(labels, scores, threshold)
    by_defect: dict[str, Any] = {}
    for defect in sorted(
        {record["defect_type"] for record in records if record["label"] == 1}
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if record["label"] == 1 and record["defect_type"] == defect
        ]
        reviewed = sum(rows[index]["decision"] == "REVIEW_REQUIRED" for index in indices)
        by_defect[defect] = {
            "count": len(indices),
            "reviewed": reviewed,
            "recall": reviewed / len(indices),
            "confidence_interval": wilson_interval(
                reviewed, len(indices), confidence_level
            ),
        }
    return {
        "image_auroc": binary_auroc(labels, scores),
        "image_average_precision": binary_average_precision(labels, scores),
        "fixed_threshold": {
            "threshold": threshold,
            **_fixed_image_metrics(counts, confidence_level),
        },
        "image_by_defect_type": by_defect,
    }


def _pixel_metrics(
    maps: np.ndarray,
    masks: np.ndarray,
    threshold: float,
    execution: dict[str, Any],
) -> dict[str, Any]:
    labels = masks.reshape(-1)
    scores = maps.reshape(-1)
    counts = confusion_counts(labels, scores, threshold)
    return {
        "threshold": threshold,
        "auroc": binary_auroc(labels, scores),
        "aupro": aupro_fpr_grid(
            maps,
            masks,
            fpr_limit=execution["aupro_fpr_limit"],
            fpr_points=execution["aupro_fpr_points"],
        ),
        "fixed_threshold": {"counts": counts, "rates": rates_from_confusion(counts)},
        "uncertainty": "not estimated from pixels as independent observations",
    }


def _latency_summary(values: list[float], scope: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "scope": scope,
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95_higher": float(np.quantile(array, 0.95, method="higher")),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _run_followup_category(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the two unique layer groups and score four candidates on development."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts" / "realiad-followup-development-v1").resolve()
    output_path = args.output.resolve()
    output_path.relative_to(artifacts_root)
    if output_path.exists():
        raise FileExistsError(f"Development output already exists: {output_path}")
    data_root = args.root.resolve(strict=True)
    protocol_path = args.protocol_config.resolve(strict=True)
    execution_path = args.execution_config.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    final_manifest_path = args.final_manifest.resolve(strict=True)
    protocol, execution, manifest, final_manifest = _load_category_contract(
        data_root,
        manifest_path,
        final_manifest_path,
        protocol_path,
        execution_path,
        args.category,
    )
    train_records = records_for_split(manifest, "train")
    calibration_records = records_for_split(manifest, "calibration")
    development_records = records_for_split(manifest, "test")
    if any(record.get("protocol_split") != "development" for record in development_records):
        raise ValueError("Development manifest test split contains a non-development record")
    if not any(record["label"] == 0 for record in development_records) or not any(
        record["label"] == 1 for record in development_records
    ):
        raise ValueError("Development split must contain normal and anomalous images")
    git = git_state(project_root)
    versions = installed_versions()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"The lab development run requires Python 3.12, found {platform.python_version()}"
        )

    import torch
    import torch.nn.functional as functional
    import torchvision
    from anomalib.models import Patchcore
    from PIL import Image

    if (
        torch.__version__ != EXPECTED_PACKAGES["torch"]
        or torchvision.__version__ != EXPECTED_PACKAGES["torchvision"]
    ):
        raise RuntimeError("Imported torch/torchvision versions differ from the validated combination")
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("Development comparison requires the validated CUDA 12.8 GPU environment")

    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output_path / protocol_path.name)
    shutil.copyfile(execution_path, output_path / execution_path.name)
    shutil.copyfile(manifest_path, output_path / "development-manifest.json")
    shutil.copyfile(final_manifest_path, output_path / "final-sealed-manifest.json")
    device = initialize_cuda_device(torch)
    seed = protocol["base_model"]["development_seed"]
    _seed_everything(seed, torch)
    image_size = protocol["base_model"]["image_size"]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def load_image(record: dict[str, Any]) -> tuple[Any, tuple[int, int]]:
        image_path = _safe_data_path(data_root, record["image_path"])
        with Image.open(image_path) as image:
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError(f"Unexpected Real-IAD input: {record['image_path']}")
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

    def load_mask(record: dict[str, Any], original_size: tuple[int, int]) -> np.ndarray:
        if record["label"] == 0:
            return np.zeros((image_size, image_size), dtype=np.bool_)
        mask_path = _safe_data_path(data_root, record["mask_path"])
        with Image.open(mask_path) as mask:
            if mask.size != original_size:
                raise RuntimeError(f"Mask dimensions differ for {record['image_path']}")
            resized = mask.convert("L").resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
            return np.asarray(resized, dtype=np.uint8) >= execution[
                "pixel_ground_truth_threshold"
            ]

    candidates_by_layers: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in protocol["candidates"]:
        candidates_by_layers.setdefault(tuple(candidate["layers"]), []).append(candidate)
    if set(candidates_by_layers) != {("layer2", "layer3"), ("layer1", "layer2")}:
        raise ValueError("Unexpected candidate layer groups")

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    candidate_results: dict[str, Any] = {}
    layer_results: dict[str, Any] = {}
    for layers, layer_candidates in candidates_by_layers.items():
        _seed_everything(seed, torch)
        layer_id = "-".join(layers)
        print(f"[{args.category}] fitting PatchCore {layer_id} on {len(train_records)} normal images...")
        model = Patchcore(
            backbone=protocol["base_model"]["backbone"],
            layers=list(layers),
            pre_trained=True,
            coreset_sampling_ratio=protocol["base_model"]["coreset_sampling_ratio"],
            num_neighbors=protocol["base_model"]["num_neighbors"],
            pre_processor=False,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        ).to(device)
        model.model.train()
        model.model.feature_extractor.eval()
        fit_start = time.perf_counter()
        with torch.no_grad():
            for index, record in enumerate(train_records, start=1):
                image_batch, _ = load_image(record)
                model.model(image_batch)
                if index % 25 == 0 or index == len(train_records):
                    print(f"  train {index}/{len(train_records)}")
        embedding_count = sum(int(value.shape[0]) for value in model.model.embedding_store)
        model.fit()
        model._is_fitted.fill_(True)
        model.model.eval()
        memory_shape = list(model.model.memory_bank.shape)
        fit_seconds = time.perf_counter() - fit_start
        if len(memory_shape) != 2 or memory_shape[0] < protocol["base_model"]["num_neighbors"]:
            raise RuntimeError("PatchCore memory bank is unexpectedly small")
        checkpoint_path = output_path / f"model-{layer_id}.pt"
        cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save(
            {
                "format_version": 1,
                "category": args.category,
                "layers": list(layers),
                "model_seed": seed,
                "git_commit": git["commit"],
                "manifest_sha256": manifest["manifest_sha256"],
                "execution_config_sha256": execution["config_sha256"],
                "memory_bank_shape": memory_shape,
                "state_dict": cpu_state,
            },
            checkpoint_path,
        )
        del cpu_state

        def predict(record: dict[str, Any]) -> tuple[float, np.ndarray, tuple[int, int]]:
            image_batch, original_size = load_image(record)
            with torch.no_grad():
                prediction = model.model(image_batch)
            raw_score, anomaly_map = _extract_prediction(prediction, image_size)
            return raw_score, anomaly_map, original_size

        print(f"[{args.category}] calibrating {layer_id} on {len(calibration_records)} normals...")
        calibration_base: list[dict[str, Any]] = []
        calibration_maps: list[np.ndarray] = []
        for index, record in enumerate(calibration_records, start=1):
            raw_score, anomaly_map, _ = predict(record)
            calibration_base.append({"image_path": record["image_path"], "raw_score": raw_score})
            calibration_maps.append(anomaly_map)
            if index % 25 == 0 or index == len(calibration_records):
                print(f"  calibration {index}/{len(calibration_records)}")
        calibration_map_array = np.stack(calibration_maps).astype(np.float32, copy=False)
        pixel_threshold = float(
            np.quantile(
                calibration_map_array.reshape(-1),
                protocol["calibration"]["pixel_quantile"],
                method="higher",
            )
        )

        print(f"[{args.category}] scoring {len(development_records)} development images...")
        development_base: list[dict[str, Any]] = []
        development_maps: list[np.ndarray] = []
        development_masks: list[np.ndarray] = []
        evaluation_latencies: list[float] = []
        for index, record in enumerate(development_records, start=1):
            sample_start = time.perf_counter()
            raw_score, anomaly_map, original_size = predict(record)
            torch.cuda.synchronize()
            evaluation_latencies.append((time.perf_counter() - sample_start) * 1000.0)
            development_base.append(
                {
                    "image_path": record["image_path"],
                    "label": record["label"],
                    "defect_type": record["defect_type"],
                    "raw_score": raw_score,
                }
            )
            development_maps.append(anomaly_map)
            development_masks.append(load_mask(record, original_size))
            if index % 25 == 0 or index == len(development_records):
                print(f"  development {index}/{len(development_records)}")
        maps_array = np.stack(development_maps).astype(np.float32, copy=False)
        masks_array = np.stack(development_masks).astype(np.bool_, copy=False)
        pixel = _pixel_metrics(maps_array, masks_array, pixel_threshold, execution)

        def benchmark(candidate: dict[str, Any]) -> dict[str, Any]:
            settings = execution["benchmark"]
            records = development_records[: min(settings["count"], len(development_records))]
            if len(records) != settings["count"]:
                raise ValueError("Development split is smaller than the locked benchmark count")
            for index in range(settings["warmup"]):
                raw, anomaly_map, _ = predict(records[index % len(records)])
                candidate_image_score(raw, anomaly_map, candidate)
            torch.cuda.synchronize()
            values: list[float] = []
            for _ in range(settings["repeats"]):
                for record in records:
                    sample_start = time.perf_counter()
                    raw, anomaly_map, _ = predict(record)
                    candidate_image_score(raw, anomaly_map, candidate)
                    torch.cuda.synchronize()
                    values.append((time.perf_counter() - sample_start) * 1000.0)
            return _latency_summary(values, settings["scope"])

        for candidate in layer_candidates:
            candidate_id = candidate["id"]
            calibration_scores = [
                candidate_image_score(row["raw_score"], anomaly_map, candidate)
                for row, anomaly_map in zip(
                    calibration_base, calibration_map_array, strict=True
                )
            ]
            threshold = higher_quantile(
                calibration_scores, protocol["calibration"]["image_quantile"]
            )
            calibration_rows = []
            for record, score in zip(calibration_records, calibration_scores, strict=True):
                calibration_rows.append(
                    {
                        "image_path": record["image_path"],
                        "candidate_id": candidate_id,
                        "score": score,
                        "threshold": threshold,
                        "decision": "REVIEW_REQUIRED" if score >= threshold else "PASS",
                    }
                )
            rows: list[dict[str, Any]] = []
            for record, base, anomaly_map in zip(
                development_records, development_base, maps_array, strict=True
            ):
                score = candidate_image_score(base["raw_score"], anomaly_map, candidate)
                decision = "REVIEW_REQUIRED" if score >= threshold else "PASS"
                rows.append(
                    {
                        "image_path": record["image_path"],
                        "label": record["label"],
                        "defect_type": record["defect_type"],
                        "candidate_id": candidate_id,
                        "score": score,
                        "threshold": threshold,
                        "decision": decision,
                        "correct": (decision == "REVIEW_REQUIRED") == bool(record["label"]),
                    }
                )
            calibration_path = output_path / f"calibration-{candidate_id}.jsonl"
            predictions_path = output_path / f"predictions-{candidate_id}.jsonl"
            _write_jsonl(calibration_path, calibration_rows)
            _write_jsonl(predictions_path, rows)
            metrics = _candidate_metrics(
                development_records,
                rows,
                threshold,
                execution["confidence_level"],
            )
            candidate_results[candidate_id] = {
                "candidate": candidate,
                "threshold": threshold,
                "calibration_count": len(calibration_rows),
                "development_count": len(rows),
                "metrics": metrics,
                "pixel": pixel,
                "evaluation_latency_ms": _latency_summary(
                    evaluation_latencies,
                    "image decode, preprocessing, GPU inference, map transfer",
                ),
                "warm_latency_ms": benchmark(candidate),
                "artifacts": {
                    "calibration_scores": calibration_path.name,
                    "calibration_scores_sha256": checksum(calibration_path),
                    "predictions": predictions_path.name,
                    "predictions_sha256": checksum(predictions_path),
                },
            }
        layer_results[layer_id] = {
            "layers": list(layers),
            "candidate_ids": [candidate["id"] for candidate in layer_candidates],
            "embedding_count_before_coreset": embedding_count,
            "memory_bank_shape": memory_shape,
            "fit_seconds": fit_seconds,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checksum(checkpoint_path),
        }
        del model, calibration_map_array, maps_array, masks_array
        torch.cuda.empty_cache()

    finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "status": "evaluated_preregistered_development",
        "purpose": execution["purpose"],
        "category": args.category,
        "development_evaluated": True,
        "final_holdout_evaluated": False,
        "automatic_pass_allowed": False,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "git": git,
        "plan_sha256": execution["plan_sha256"],
        "execution_config_sha256": execution["config_sha256"],
        "development_manifest_sha256": manifest["manifest_sha256"],
        "final_manifest_sha256": final_manifest["manifest_sha256"],
        "seed": seed,
        "counts": {
            "train_normal": len(train_records),
            "calibration_normal": len(calibration_records),
            "development_normal": sum(record["label"] == 0 for record in development_records),
            "development_anomaly": sum(record["label"] == 1 for record in development_records),
        },
        "layer_groups": layer_results,
        "candidates": candidate_results,
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
        "limitations": [
            "Only the preregistered development split was scored.",
            "The sealed final holdout was not extracted, decoded, previewed, or scored.",
            "Development results select one candidate; they are not final performance estimates.",
            "Automatic PASS remains prohibited.",
        ],
    }
    write_json(output_path / "category-result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "category": args.category,
                "candidate_ids": sorted(candidate_results),
                "final_holdout_evaluated": False,
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return result


def run_followup_category(args: argparse.Namespace) -> dict[str, Any]:
    """Run one category and remove only output created by a failed invocation."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (
        project_root / "artifacts" / "realiad-followup-development-v1"
    ).resolve()
    output_path = args.output.resolve()
    output_path.relative_to(artifacts_root)
    existed_before = output_path.exists()
    try:
        return _run_followup_category(args)
    except BaseException:
        if not existed_before and output_path.exists():
            shutil.rmtree(output_path)
        raise


def select_followup_candidate(
    category_results: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    execution: dict[str, Any],
    git: dict[str, str],
) -> dict[str, Any]:
    """Apply the preregistered constraint and deterministic tie breakers."""
    expected_categories = set(FOLLOWUP_CATEGORIES)
    if set(category_results) != expected_categories:
        raise ValueError("Selection requires exactly PCB and terminalblock results")
    candidate_ids = [candidate["id"] for candidate in protocol["candidates"]]
    rows = []
    for candidate_id in candidate_ids:
        per_category: dict[str, Any] = {}
        escape_values = []
        auroc_values = []
        overkill_values = []
        latency_values = []
        for category in sorted(expected_categories):
            result = category_results[category]
            if (
                result.get("status") != "evaluated_preregistered_development"
                or result.get("final_holdout_evaluated") is not False
                or result.get("plan_sha256") != execution["plan_sha256"]
                or result.get("execution_config_sha256") != execution["config_sha256"]
                or result.get("git") != git
                or set(result.get("candidates", {})) != set(candidate_ids)
            ):
                raise ValueError(f"Invalid category result for selection: {category}")
            candidate = result["candidates"][candidate_id]
            metrics = candidate["metrics"]
            rates = metrics["fixed_threshold"]["rates"]
            values = {
                "escape_rate": rates["escape_rate"],
                "overkill_rate": rates["overkill_rate"],
                "image_auroc": metrics["image_auroc"],
                "warm_p95_latency_ms": candidate["warm_latency_ms"]["p95_higher"],
            }
            if any(value is None or not math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"Candidate metrics are incomplete: {category}/{candidate_id}")
            per_category[category] = values
            escape_values.append(float(values["escape_rate"]))
            auroc_values.append(float(values["image_auroc"]))
            overkill_values.append(float(values["overkill_rate"]))
            latency_values.append(float(values["warm_p95_latency_ms"]))
        maximum_overkill = max(overkill_values)
        rows.append(
            {
                "candidate_id": candidate_id,
                "eligible": maximum_overkill
                <= protocol["selection"]["maximum_category_overkill_rate"],
                "per_category": per_category,
                "macro_escape_rate": float(np.mean(escape_values)),
                "macro_image_auroc": float(np.mean(auroc_values)),
                "macro_overkill_rate": float(np.mean(overkill_values)),
                "macro_warm_p95_latency_ms": float(np.mean(latency_values)),
                "maximum_category_overkill_rate": maximum_overkill,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["macro_escape_rate"],
            -row["macro_image_auroc"],
            row["macro_overkill_rate"],
            row["macro_warm_p95_latency_ms"],
            row["candidate_id"],
        ),
    )
    selected = ranked[0]["candidate_id"] if ranked else None
    status = "candidate_selected_final_still_sealed" if selected else "no_eligible_candidate"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "purpose": execution["purpose"],
        "selected_candidate_id": selected,
        "final_gate_ready": selected is not None,
        "automatic_pass_allowed": False,
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "plan_sha256": execution["plan_sha256"],
        "execution_config_sha256": execution["config_sha256"],
        "final_manifest_sha256": {
            category: category_results[category]["final_manifest_sha256"]
            for category in sorted(expected_categories)
        },
        "selection_rule": protocol["selection"],
        "candidate_results": rows,
        "final_holdout_evaluated": False,
        "next_step": (
            "lock this report and run the selected candidate once on final seeds 42, 43, 44"
            if selected
            else "keep final sealed because no candidate satisfied the overkill constraint"
        ),
        "limitations": [
            "Selection uses development data and is not a final performance estimate.",
            "Final holdout remains sealed after selection.",
            "Any candidate change after final observation makes the result exploratory.",
        ],
    }
    report["selection_report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    return report


def _validate_result_artifacts(path: Path, result: dict[str, Any]) -> None:
    root = path.parent.resolve(strict=True)
    for layer in result.get("layer_groups", {}).values():
        artifact = (root / layer.get("checkpoint", "")).resolve(strict=True)
        artifact.relative_to(root)
        if checksum(artifact) != layer.get("checkpoint_sha256"):
            raise ValueError("Category checkpoint checksum mismatch")
    for candidate in result.get("candidates", {}).values():
        artifacts = candidate.get("artifacts", {})
        for name_key, sha_key in (
            ("calibration_scores", "calibration_scores_sha256"),
            ("predictions", "predictions_sha256"),
        ):
            artifact = (root / artifacts.get(name_key, "")).resolve(strict=True)
            artifact.relative_to(root)
            if checksum(artifact) != artifacts.get(sha_key):
                raise ValueError("Category score artifact checksum mismatch")


def run_followup_selection(args: argparse.Namespace) -> dict[str, Any]:
    """Validate both category results and write the immutable selection report."""
    project_root = Path(__file__).resolve().parents[2]
    artifacts_root = (project_root / "artifacts" / "realiad-followup-development-v1").resolve()
    output_path = args.output.resolve()
    output_path.relative_to(artifacts_root)
    if output_path.exists():
        raise FileExistsError(f"Selection output already exists: {output_path}")
    protocol = load_realiad_followup_protocol(args.protocol_config.resolve(strict=True))
    execution = load_followup_development_execution(
        args.execution_config.resolve(strict=True)
    )
    git = git_state(project_root)
    paths = {
        "realiad_pcb": args.pcb_result.resolve(strict=True),
        "realiad_terminalblock": args.terminalblock_result.resolve(strict=True),
    }
    results = {}
    for category, path in paths.items():
        result = read_json(path)
        if result.get("category") != category:
            raise ValueError(f"Category result path does not contain {category}")
        _validate_result_artifacts(path, result)
        results[category] = result
    report = select_followup_candidate(results, protocol, execution, git)
    output_path.mkdir(parents=True, exist_ok=False)
    report_path = output_path / "selection-report.json"
    write_json(report_path, report)
    selected = report["selected_candidate_id"] or "NONE"
    lines = [
        "# Real-IAD follow-up development selection v1",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selected candidate: `{selected}`",
        "",
        f"Selection report SHA-256: `{report['selection_report_sha256']}`",
        "",
        "| Candidate | Eligible | Macro escape | Macro AUROC | Macro overkill | Macro warm p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["candidate_results"]:
        lines.append(
            f"| `{row['candidate_id']}` | {row['eligible']} | "
            f"{row['macro_escape_rate']:.6f} | {row['macro_image_auroc']:.6f} | "
            f"{row['macro_overkill_rate']:.6f} | {row['macro_warm_p95_latency_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Final holdout was not extracted, decoded, previewed, or scored by this step.",
            "Automatic PASS remains prohibited.",
            "",
        ]
    )
    summary_path = output_path / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_candidate_id": report["selected_candidate_id"],
                "selection_report_sha256": report["selection_report_sha256"],
                "final_gate_ready": report["final_gate_ready"],
                "final_holdout_evaluated": False,
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--protocol-config",
        type=Path,
        default=project_root / "configs" / "realiad-followup-preregistration-v1.toml",
    )
    shared.add_argument(
        "--execution-config",
        type=Path,
        default=(
            project_root
            / "configs"
            / "realiad-followup-development-execution-v1.toml"
        ),
    )
    category = subparsers.add_parser("category", parents=[shared])
    category.add_argument("--root", type=Path, required=True)
    category.add_argument("--manifest", type=Path, required=True)
    category.add_argument("--final-manifest", type=Path, required=True)
    category.add_argument("--category", choices=sorted(FOLLOWUP_CATEGORIES), required=True)
    category.add_argument("--output", type=Path, required=True)
    selection = subparsers.add_parser("select", parents=[shared])
    selection.add_argument("--pcb-result", type=Path, required=True)
    selection.add_argument("--terminalblock-result", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "category":
            run_followup_category(args)
        else:
            run_followup_selection(args)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
