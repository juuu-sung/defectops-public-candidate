"""Locked, resumable Real-IAD final evaluation; final images are never used in fit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from defectops.evaluation_metrics import (
    binary_auroc,
    binary_average_precision,
    stratified_bootstrap_interval,
)
from defectops.manifest import canonical, verify_manifest
from defectops.padim_evaluation import _extract_prediction
from defectops.padim_experiment import (
    EXPECTED_PACKAGES,
    checksum,
    git_state,
    higher_quantile,
    initialize_cuda_device,
    installed_versions,
    read_json,
    records_for_split,
)
from defectops.realiad_followup_development import (
    _assert_final_not_extracted,
    _candidate_metrics,
    _pixel_metrics,
    _seed_everything,
    candidate_image_score,
    load_followup_development_execution,
)
from defectops.realiad_followup_protocol import FOLLOWUP_CATEGORIES, load_realiad_followup_protocol
from defectops.realiad_followup_registration import verify_followup_final_manifest


EXPECTED_SELECTION_SHA256 = "1dd08ad5ea0aa3156240a9400f6fa8f7cbb5777eeea077cc448ffe94b3ea13d3"
EXPECTED_HANDOFF_SHA256 = "373177ef8eaa0c63c632f58e2a32154542a6b121ccf2a1230f193bc30338b936"
EXPECTED_CANDIDATE = "fine_features_top_1pct"
FINAL_ROOT_NAME = "realiad-followup-final-v1"
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_REPLICATES = 1000


def _canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def _checked_json(path: Path, claimed_key: str | None = None) -> dict[str, Any]:
    payload = read_json(path)
    if claimed_key:
        unsigned = dict(payload)
        claimed = unsigned.pop(claimed_key, None)
        if claimed != _canonical_sha(unsigned):
            raise ValueError(f"Checksum mismatch: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.stem}-",
        suffix=".json", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _paths(project_root: Path, category: str, seed: int) -> dict[str, Path]:
    short = FOLLOWUP_CATEGORIES[category]
    artifacts = project_root / "artifacts"
    final_root = artifacts / FINAL_ROOT_NAME
    return {
        "project": project_root,
        "root": final_root,
        "fit": final_root / "fits" / f"{short}-seed{seed}",
        "evaluation": final_root / "evaluations" / f"{short}-seed{seed}",
        "development": artifacts / "realiad-followup-development-v1" / short,
        "selection": artifacts / "realiad-followup-development-v1" / "selection" / "selection-report.json",
        "development_manifest": project_root / "data" / "manifests" / f"realiad-{short}-development-v1.json",
        "final_manifest": project_root / "data" / "manifests" / f"realiad-{short}-final-sealed-v1.json",
        "source_archive": Path.home() / "Downloads" / f"{short}.zip",
    }


def _load_lock(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = tomllib.loads(
        (project_root / "configs" / "realiad-followup-final-lock-v1.toml").read_text(
            encoding="utf-8"
        )
    )
    protocol = load_realiad_followup_protocol(
        project_root / "configs" / "realiad-followup-preregistration-v1.toml"
    )
    execution = load_followup_development_execution(
        project_root / "configs" / "realiad-followup-development-execution-v1.toml"
    )
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != "A-normal-only-v2"
        or lock.get("status") != "candidate_selected_final_holdout_unopened"
        or lock.get("selected_candidate_id") != EXPECTED_CANDIDATE
        or lock.get("selection_report_sha256") != EXPECTED_SELECTION_SHA256
        or lock.get("handoff_zip_sha256") != EXPECTED_HANDOFF_SHA256
        or lock.get("selection_git_commit") != "795e70eaa04d6a783909a7f44b44e431eadc3abd"
        or lock.get("split_plan_sha256") != execution["plan_sha256"]
        or lock.get("development_execution_config_sha256") != execution["config_sha256"]
        or lock.get("final_seeds") != protocol["base_model"]["final_seeds"]
        or lock.get("automatic_pass_allowed") is not False
        or lock.get("final_manifest_sha256")
        != {category: execution["manifests"][category]["final_sha256"] for category in FOLLOWUP_CATEGORIES}
    ):
        raise ValueError("Final lock differs from the preregistered selection contract")
    return lock, protocol, execution


def validate_gate(project_root: Path, category: str, seed: int) -> dict[str, Any]:
    """Validate the locked selection and manifests without opening final image bytes."""
    if category not in FOLLOWUP_CATEGORIES or seed not in (42, 43, 44):
        raise ValueError("Only the locked categories and seeds 42, 43, 44 are allowed")
    lock, protocol, execution = _load_lock(project_root)
    git = git_state(project_root)
    paths = _paths(project_root, category, seed)
    selection = _checked_json(paths["selection"], "selection_report_sha256")
    if (
        selection["selection_report_sha256"] != lock["selection_report_sha256"]
        or selection["selected_candidate_id"] != lock["selected_candidate_id"]
        or selection["git_commit"] != lock["selection_git_commit"]
        or selection["final_gate_ready"] is not True
        or selection["final_holdout_evaluated"] is not False
        or selection["automatic_pass_allowed"] is not False
        or selection["plan_sha256"] != lock["split_plan_sha256"]
        or selection["execution_config_sha256"] != lock["development_execution_config_sha256"]
        or selection["final_manifest_sha256"] != lock["final_manifest_sha256"]
    ):
        raise ValueError("Selection report does not match the final lock")
    development = _checked_json(paths["development_manifest"], "manifest_sha256")
    final = _checked_json(paths["final_manifest"], "manifest_sha256")
    if (
        development["manifest_sha256"] != execution["manifests"][category]["development_sha256"]
        or final["manifest_sha256"] != lock["final_manifest_sha256"][category]
        or final.get("sealed") is not True
        or final.get("images_extracted") is not False
    ):
        raise ValueError("Development/final manifest does not match the final lock")
    data_root = project_root / "data"
    verify_manifest(data_root, development)
    verify_followup_final_manifest(data_root, final)
    _assert_final_not_extracted(data_root, final)
    result_path = paths["development"] / "category-result.json"
    dev_result = read_json(result_path)
    handoff = project_root / "artifacts" / "handoff" / "realiad-followup-development-v1.zip"
    if checksum(handoff) != lock["handoff_zip_sha256"]:
        raise ValueError("Development handoff ZIP differs from the locked evidence")
    with zipfile.ZipFile(handoff) as archive:
        expected_name = f"{FOLLOWUP_CATEGORIES[category]}/category-result.json"
        matches = [
            info for info in archive.infolist() if info.filename.replace("\\", "/") == expected_name
        ]
        if len(matches) != 1 or json.loads(archive.read(matches[0])) != dev_result:
            raise ValueError("Development category result differs from the locked handoff ZIP")
    layer = dev_result["layer_groups"]["layer1-layer2"]
    checkpoint_path = paths["development"] / layer["checkpoint"]
    selected = dev_result["candidates"][EXPECTED_CANDIDATE]
    if (
        dev_result.get("status") != "evaluated_preregistered_development"
        or dev_result.get("git", {}).get("commit") != lock["selection_git_commit"]
        or dev_result.get("category") != category
        or dev_result.get("final_holdout_evaluated") is not False
        or dev_result.get("development_manifest_sha256") != development["manifest_sha256"]
        or dev_result.get("final_manifest_sha256") != final["manifest_sha256"]
        or checksum(checkpoint_path) != layer["checkpoint_sha256"]
        or selected["candidate"] != next(
            item for item in protocol["candidates"] if item["id"] == EXPECTED_CANDIDATE
        )
    ):
        raise ValueError("Development result/checkpoint does not match the final lock")
    return {
        "lock": lock,
        "protocol": protocol,
        "execution": execution,
        "git": git,
        "paths": paths,
        "development": development,
        "final": final,
        "dev_result": dev_result,
        "dev_checkpoint": checkpoint_path,
        "candidate": selected["candidate"],
    }


def _gpu_modules() -> tuple[Any, Any, Any, Any, Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Validated lab Python is 3.12, found {platform.python_version()}")
    import torch
    import torch.nn.functional as functional
    import torchvision
    from anomalib.models import Patchcore
    from PIL import Image

    if (
        torch.__version__ != EXPECTED_PACKAGES["torch"]
        or torchvision.__version__ != EXPECTED_PACKAGES["torchvision"]
        or not torch.cuda.is_available()
        or torch.version.cuda != "12.8"
    ):
        raise RuntimeError("Final evaluation requires the validated Windows CUDA environment")
    installed_versions()
    return torch, functional, Patchcore, Image, initialize_cuda_device(torch)


def _model(protocol: dict[str, Any], checkpoint_path: Path, context: dict[str, Any], gpu: tuple[Any, ...]) -> Any:
    torch, _, Patchcore, _, device = gpu
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("category") != context["category"]
        or checkpoint.get("layers") != ["layer1", "layer2"]
        or checkpoint.get("model_seed") != context["seed"]
        or checkpoint.get("manifest_sha256") != context["manifest_sha256"]
        or checkpoint.get("execution_config_sha256") != context["execution_config_sha256"]
        or list(checkpoint.get("state_dict", {}).get("model.memory_bank", torch.empty(0)).shape)
        != checkpoint.get("memory_bank_shape")
    ):
        raise ValueError("Checkpoint metadata does not match the locked fit")
    model = Patchcore(
        backbone=protocol["base_model"]["backbone"],
        layers=["layer1", "layer2"],
        pre_trained=False,
        coreset_sampling_ratio=protocol["base_model"]["coreset_sampling_ratio"],
        num_neighbors=protocol["base_model"]["num_neighbors"],
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model.model.memory_bank = torch.empty_like(checkpoint["state_dict"]["model.memory_bank"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    model._is_fitted.fill_(True)
    model.model.eval()
    return model


def _preprocess(image: Any, size: int, gpu: tuple[Any, ...]) -> Any:
    torch, functional, _, _, device = gpu
    values = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    values = values.to(torch.float32).div_(255.0)
    values = functional.interpolate(
        values.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return ((values - mean) / std).unsqueeze(0).to(device)


def _predict(model: Any, image: Any, protocol: dict[str, Any], gpu: tuple[Any, ...]) -> tuple[float, np.ndarray]:
    torch = gpu[0]
    with torch.no_grad():
        prediction = model.model(_preprocess(image, protocol["base_model"]["image_size"], gpu))
    return _extract_prediction(prediction, protocol["base_model"]["image_size"])


def _calibrate(model: Any, gate: dict[str, Any], gpu: tuple[Any, ...]) -> tuple[float, float, list[dict[str, Any]]]:
    Image = gpu[3]
    root = gate["paths"]["project"] / "data"
    candidate = gate["candidate"]
    protocol = gate["protocol"]
    execution = gate["execution"]
    rows = []
    all_maps = []
    for record in records_for_split(gate["development"], "calibration"):
        path = (root / record["image_path"]).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
        with Image.open(path) as image:
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError(f"Invalid calibration image: {path}")
            raw, anomaly_map = _predict(model, image, protocol, gpu)
        rows.append({"image_path": record["image_path"], "score": candidate_image_score(raw, anomaly_map, candidate)})
        all_maps.append(anomaly_map)
    threshold = higher_quantile([row["score"] for row in rows], protocol["calibration"]["image_quantile"])
    pixel_threshold = float(np.quantile(
        np.stack(all_maps).reshape(-1), protocol["calibration"]["pixel_quantile"], method="higher"
    ))
    if len(rows) == 0 or not math.isfinite(pixel_threshold):
        raise ValueError("Calibration failed")
    return threshold, pixel_threshold, rows


def _fit_record(gate: dict[str, Any], seed: int) -> dict[str, Any]:
    paths = gate["paths"]
    path = paths["fit"] / "fit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fit not complete: {path}")
    fit = read_json(path)
    checkpoint = paths["project"] / fit.get("checkpoint", "")
    calibration_path = paths["project"] / fit.get("calibration_scores", "")
    if (
        fit.get("status") != "fit_calibrated_final_locked"
        or fit.get("category") != gate["final"]["categories"][0]
        or fit.get("seed") != seed
        or fit.get("candidate_id") != EXPECTED_CANDIDATE
        or fit.get("execution_git_commit") != gate["git"]["commit"]
        or fit.get("selection_report_sha256") != EXPECTED_SELECTION_SHA256
        or fit.get("development_manifest_sha256") != gate["development"]["manifest_sha256"]
        or fit.get("final_manifest_sha256") != gate["final"]["manifest_sha256"]
        or fit.get("checkpoint_sha256") != checksum(checkpoint)
        or fit.get("calibration_scores_sha256") != checksum(calibration_path)
        or not math.isfinite(fit.get("threshold", float("nan")))
        or not math.isfinite(fit.get("pixel_threshold", float("nan")))
        or fit.get("final_holdout_opened") is not False
        or fit.get("automatic_pass_allowed") is not False
    ):
        raise ValueError(f"Existing fit failed contract validation: {path}")
    calibration = [json.loads(line) for line in calibration_path.read_text(encoding="utf-8").splitlines()]
    expected_paths = [record["image_path"] for record in records_for_split(gate["development"], "calibration")]
    if (
        [row.get("image_path") for row in calibration] != expected_paths
        or higher_quantile([row["score"] for row in calibration], gate["protocol"]["calibration"]["image_quantile"])
        != fit["threshold"]
    ):
        raise ValueError(f"Existing calibration scores differ from the locked fit: {path}")
    return fit


def prepare_fit(gate: dict[str, Any], category: str, seed: int) -> dict[str, Any]:
    paths = gate["paths"]
    if paths["fit"].exists():
        return _fit_record(gate, seed)
    checkpoint = gate["dev_checkpoint"] if seed == 42 else None
    if seed == 42:
        selected = gate["dev_result"]["candidates"][EXPECTED_CANDIDATE]
        fit = {
            "status": "fit_calibrated_final_locked", "category": category, "seed": seed,
            "execution_git_commit": gate["git"]["commit"],
            "candidate_id": EXPECTED_CANDIDATE,
            "selection_report_sha256": EXPECTED_SELECTION_SHA256,
            "development_manifest_sha256": gate["development"]["manifest_sha256"],
            "final_manifest_sha256": gate["final"]["manifest_sha256"],
            "checkpoint": checkpoint.relative_to(paths["project"]).as_posix(),
            "checkpoint_sha256": checksum(checkpoint),
            "calibration_scores": (
                paths["development"] / selected["artifacts"]["calibration_scores"]
            ).relative_to(paths["project"]).as_posix(),
            "calibration_scores_sha256": selected["artifacts"]["calibration_scores_sha256"],
            "threshold": selected["threshold"],
            "pixel_threshold": selected["pixel"]["threshold"],
            "calibration_source": "locked_development_seed42",
            "final_holdout_opened": False, "automatic_pass_allowed": False,
        }
    else:
        gpu = _gpu_modules()
        torch, _, Patchcore, Image, device = gpu
        _seed_everything(seed, torch)
        base = gate["protocol"]["base_model"]
        model = Patchcore(
            backbone=base["backbone"], layers=["layer1", "layer2"], pre_trained=True,
            coreset_sampling_ratio=base["coreset_sampling_ratio"], num_neighbors=base["num_neighbors"],
            pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
        ).to(device)
        model.model.train()
        model.model.feature_extractor.eval()
        start = time.perf_counter()
        root = paths["project"] / "data"
        train = records_for_split(gate["development"], "train")
        with torch.no_grad():
            for index, record in enumerate(train, start=1):
                path = (root / record["image_path"]).resolve(strict=True)
                path.relative_to(root.resolve(strict=True))
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.mode != "RGB":
                        raise ValueError(f"Invalid train image: {path}")
                    model.model(_preprocess(image, base["image_size"], gpu))
                if index % 25 == 0 or index == len(train):
                    print(f"[{category}/seed{seed}] train {index}/{len(train)}", flush=True)
        model.fit()
        model._is_fitted.fill_(True)
        model.model.eval()
        paths["fit"].parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{paths['fit'].name}-", dir=paths["fit"].parent))
        checkpoint = stage / "model.pt"
        cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save({
            "format_version": 1, "category": category, "layers": ["layer1", "layer2"],
            "model_seed": seed, "git_commit": gate["git"]["commit"],
            "manifest_sha256": gate["development"]["manifest_sha256"],
            "execution_config_sha256": gate["execution"]["config_sha256"],
            "memory_bank_shape": list(model.model.memory_bank.shape), "state_dict": cpu_state,
        }, checkpoint)
        del cpu_state
        threshold, pixel_threshold, calibration = _calibrate(model, gate, gpu)
        # Checkpoint reload must reproduce calibration scores before final can open.
        del model
        torch.cuda.empty_cache()
        model = _model(gate["protocol"], checkpoint, {
            "category": category, "seed": seed,
            "manifest_sha256": gate["development"]["manifest_sha256"],
            "execution_config_sha256": gate["execution"]["config_sha256"],
        }, gpu)
        reload_threshold, reload_pixel, reload_rows = _calibrate(model, gate, gpu)
        difference = max(abs(a["score"] - b["score"]) for a, b in zip(calibration, reload_rows, strict=True))
        if difference > 1e-5 or threshold != reload_threshold or pixel_threshold != reload_pixel:
            raise ValueError(f"Calibration reload mismatch: {difference}")
        fit = {
            "status": "fit_calibrated_final_locked", "category": category, "seed": seed,
            "execution_git_commit": gate["git"]["commit"],
            "candidate_id": EXPECTED_CANDIDATE,
            "selection_report_sha256": EXPECTED_SELECTION_SHA256,
            "development_manifest_sha256": gate["development"]["manifest_sha256"],
            "final_manifest_sha256": gate["final"]["manifest_sha256"],
            "checkpoint": (paths["fit"] / "model.pt").relative_to(paths["project"]).as_posix(),
            "checkpoint_sha256": checksum(checkpoint), "threshold": threshold,
            "calibration_scores": (paths["fit"] / "calibration-scores.jsonl").relative_to(paths["project"]).as_posix(),
            "pixel_threshold": pixel_threshold, "calibration_source": "normal_only_seed_fit",
            "calibration_count": len(calibration), "reload_max_absolute_score_difference": difference,
            "fit_calibration_seconds": time.perf_counter() - start,
            "final_holdout_opened": False, "automatic_pass_allowed": False,
        }
        calibration_path = stage / "calibration-scores.jsonl"
        with calibration_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in calibration:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fit["calibration_scores_sha256"] = checksum(calibration_path)
        _atomic_write_json(stage / "fit.json", fit)
        stage.rename(paths["fit"])
        return _fit_record(gate, seed)
    paths["fit"].parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{paths['fit'].name}-", dir=paths["fit"].parent))
    _atomic_write_json(stage / "fit.json", fit)
    stage.rename(paths["fit"])
    return _fit_record(gate, seed)


def _sample_digest(metadata: dict[str, Any], anomaly_map: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256(canonical(metadata))
    digest.update(np.ascontiguousarray(anomaly_map, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.bool_).tobytes())
    return digest.hexdigest()


def _read_sample(path: Path, expected: dict[str, Any], size: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        anomaly_map = np.asarray(saved["anomaly_map"], dtype=np.float32)
        mask = np.asarray(saved["mask"], dtype=np.bool_)
        digest = str(saved["digest"].item())
    if (
        any(metadata.get(key) != value for key, value in expected.items())
        or anomaly_map.shape != (size, size)
        or mask.shape != (size, size)
        or not np.isfinite(anomaly_map).all()
        or not math.isfinite(metadata.get("score", float("nan")))
        or digest != _sample_digest(metadata, anomaly_map, mask)
    ):
        raise ValueError(f"Final sample artifact is invalid; it was not rescored: {path}")
    return metadata, anomaly_map, mask


def _write_sample(path: Path, metadata: dict[str, Any], anomaly_map: np.ndarray, mask: np.ndarray) -> None:
    digest = _sample_digest(metadata, anomaly_map, mask)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream, metadata=json.dumps(metadata, sort_keys=True, ensure_ascii=False),
            anomaly_map=anomaly_map.astype(np.float32), mask=mask.astype(np.bool_), digest=digest,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _archive_member(archive: zipfile.ZipFile, member: str, expected_sha: str) -> bytes:
    if member.startswith("/") or ".." in Path(member).parts or "\\" in member:
        raise ValueError(f"Unsafe final archive member: {member}")
    info = archive.getinfo(member)
    if info.is_dir() or info.file_size <= 0:
        raise ValueError(f"Invalid final archive member: {member}")
    data = archive.read(info)
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"Final archive member SHA-256 mismatch: {member}")
    return data


def _final_sample(
    archive: zipfile.ZipFile, record: dict[str, Any], model: Any, gate: dict[str, Any], gpu: tuple[Any, ...]
) -> tuple[float, np.ndarray, np.ndarray]:
    Image = gpu[3]
    image_bytes = _archive_member(archive, record["source_image_member"], record["image_sha256"])
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError("Final image is not RGB JPEG")
        original_size = image.size
        raw, anomaly_map = _predict(model, image, gate["protocol"], gpu)
    size = gate["protocol"]["base_model"]["image_size"]
    if record["label"] == 0:
        mask = np.zeros((size, size), dtype=np.bool_)
    else:
        mask_bytes = _archive_member(archive, record["source_mask_member"], record["mask_sha256"])
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            if source_mask.format != "PNG" or source_mask.size != original_size:
                raise ValueError("Final mask format or dimensions differ from image")
            resized = source_mask.convert("L").resize((size, size), resample=Image.Resampling.NEAREST)
            mask = np.asarray(resized, dtype=np.uint8) >= gate["execution"]["pixel_ground_truth_threshold"]
    return candidate_image_score(raw, anomaly_map, gate["candidate"]), anomaly_map, mask


def _fit_context(gate: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": fit["category"], "seed": fit["seed"],
        "manifest_sha256": fit["development_manifest_sha256"],
        "execution_config_sha256": gate["execution"]["config_sha256"],
    }


def evaluate(gate: dict[str, Any], category: str, seed: int) -> dict[str, Any]:
    """Score each final image once and resume only from complete per-image artifacts."""
    paths = gate["paths"]
    fit = _fit_record(gate, seed)
    # Enforce all six fits/calibrations before any final image can be opened.
    for other in FOLLOWUP_CATEGORIES:
        for other_seed in gate["lock"]["final_seeds"]:
            _fit_record(validate_gate(paths["project"], other, other_seed), other_seed)
    archive_path = paths["source_archive"].resolve(strict=True)
    if checksum(archive_path) != gate["final"]["source_archive_sha256"]:
        raise ValueError("Source archive SHA-256 differs from the sealed manifest")
    evaluation_dir = paths["evaluation"]
    samples_dir = evaluation_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    records = gate["final"]["records"]
    if len({record["source_image_member"] for record in records}) != len(records):
        raise ValueError("Final manifest contains duplicate members")
    size = gate["protocol"]["base_model"]["image_size"]
    context = {
        "selection_report_sha256": EXPECTED_SELECTION_SHA256,
        "final_manifest_sha256": gate["final"]["manifest_sha256"],
        "checkpoint_sha256": fit["checkpoint_sha256"],
        "threshold": fit["threshold"],
        "category": category, "seed": seed,
    }
    sample_rows = []
    maps = []
    masks = []
    gpu = None
    model = None
    with zipfile.ZipFile(archive_path) as archive:
        for index, record in enumerate(records):
            sample_path = samples_dir / f"sample-{index:04d}.npz"
            expected = {
                **context, "source_image_member": record["source_image_member"],
                "image_sha256": record["image_sha256"], "label": record["label"],
                "defect_type": record["defect_type"],
            }
            if sample_path.exists():
                metadata, anomaly_map, mask = _read_sample(sample_path, expected, size)
            else:
                if gpu is None:
                    gpu = _gpu_modules()
                    model = _model(gate["protocol"], paths["project"] / fit["checkpoint"], _fit_context(gate, fit), gpu)
                    threshold, pixel_threshold, _ = _calibrate(model, gate, gpu)
                    if abs(threshold - fit["threshold"]) > 1e-5 or abs(pixel_threshold - fit["pixel_threshold"]) > 1e-5:
                        raise ValueError("Calibration differs after model reload; final was not scored")
                score, anomaly_map, mask = _final_sample(archive, record, model, gate, gpu)
                metadata = {**expected, "score": score}
                _write_sample(sample_path, metadata, anomaly_map, mask)
            sample_rows.append(metadata)
            maps.append(anomaly_map)
            masks.append(mask)
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(f"[{category}/seed{seed}] final {index + 1}/{len(records)}", flush=True)
    rows = [
        {
            "source_image_member": row["source_image_member"], "label": row["label"],
            "defect_type": row["defect_type"], "candidate_id": EXPECTED_CANDIDATE,
            "score": row["score"], "threshold": fit["threshold"],
            "decision": "REVIEW_REQUIRED" if row["score"] >= fit["threshold"] else "PASS",
        }
        for row in sample_rows
    ]
    metrics = _candidate_metrics(records, rows, fit["threshold"], CONFIDENCE_LEVEL)
    labels = np.asarray([record["label"] for record in records], dtype=np.bool_)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    metrics["image_auroc_95pct_interval"] = stratified_bootstrap_interval(
        labels, scores, binary_auroc, seed=20260923 + seed, replicates=BOOTSTRAP_REPLICATES,
        confidence_level=CONFIDENCE_LEVEL,
    )
    metrics["image_average_precision_95pct_interval"] = stratified_bootstrap_interval(
        labels, scores, binary_average_precision, seed=20261023 + seed,
        replicates=BOOTSTRAP_REPLICATES, confidence_level=CONFIDENCE_LEVEL,
    )
    pixel = _pixel_metrics(np.stack(maps), np.stack(masks), fit["pixel_threshold"], gate["execution"])
    predictions = evaluation_dir / "predictions.jsonl"
    if not predictions.exists():
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".predictions-",
            suffix=".jsonl", dir=evaluation_dir, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, predictions)
    else:
        existing = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        if existing != rows:
            raise ValueError("Existing final predictions differ from sample artifacts")
    report = {
        "schema_version": 1, "status": "evaluated_final_locked", "category": category,
        "seed": seed, "candidate_id": EXPECTED_CANDIDATE,
        "execution_git_commit": gate["git"]["commit"],
        "selection_report_sha256": EXPECTED_SELECTION_SHA256,
        "final_manifest_sha256": gate["final"]["manifest_sha256"],
        "checkpoint_sha256": fit["checkpoint_sha256"],
        "threshold": fit["threshold"], "pixel_threshold": fit["pixel_threshold"],
        "sample_count": len(records), "metrics": metrics, "pixel": pixel,
        "sample_artifact_sha256": [
            checksum(samples_dir / f"sample-{index:04d}.npz") for index in range(len(records))
        ],
        "predictions_sha256": checksum(predictions),
        "automatic_pass_allowed": False,
        "limitations": ["One locked final protocol; no final-based tuning or automatic PASS."],
    }
    report_path = evaluation_dir / "evaluation.json"
    if report_path.exists():
        if read_json(report_path) != report:
            raise ValueError("Existing final evaluation report differs; it was not overwritten")
    else:
        _atomic_write_json(report_path, report)
    return report


def summarize(project_root: Path) -> dict[str, Any]:
    reports = []
    for category in FOLLOWUP_CATEGORIES:
        for seed in (42, 43, 44):
            gate = validate_gate(project_root, category, seed)
            fit = _fit_record(gate, seed)
            report_path = gate["paths"]["evaluation"] / "evaluation.json"
            report = read_json(report_path)
            if (
                report.get("status") != "evaluated_final_locked"
                or report.get("category") != category
                or report.get("seed") != seed
                or report.get("candidate_id") != EXPECTED_CANDIDATE
                or report.get("execution_git_commit") != gate["git"]["commit"]
                or report.get("final_manifest_sha256") != gate["final"]["manifest_sha256"]
                or report.get("checkpoint_sha256") != fit["checkpoint_sha256"]
                or report.get("predictions_sha256") != checksum(gate["paths"]["evaluation"] / "predictions.jsonl")
                or report.get("sample_artifact_sha256") != [
                    checksum(gate["paths"]["evaluation"] / "samples" / f"sample-{index:04d}.npz")
                    for index in range(report["sample_count"])
                ]
            ):
                raise ValueError(f"Invalid final evaluation report: {category}/seed{seed}")
            reports.append(report)
    fields = {
        "image_auroc": lambda item: item["metrics"]["image_auroc"],
        "image_average_precision": lambda item: item["metrics"]["image_average_precision"],
        "escape_rate": lambda item: item["metrics"]["fixed_threshold"]["rates"]["escape_rate"],
        "overkill_rate": lambda item: item["metrics"]["fixed_threshold"]["rates"]["overkill_rate"],
        "pixel_auroc": lambda item: item["pixel"]["auroc"],
        "pixel_aupro": lambda item: item["pixel"]["aupro"]["value"],
    }
    per_category = {
        category: {
            name: float(np.mean([fn(report) for report in reports if report["category"] == category]))
            for name, fn in fields.items()
        }
        for category in FOLLOWUP_CATEGORIES
    }
    summary = {
        "schema_version": 1, "status": "final_evaluation_complete_no_automatic_pass",
        "selection_report_sha256": EXPECTED_SELECTION_SHA256,
        "candidate_id": EXPECTED_CANDIDATE,
        "execution_git_commit": reports[0]["execution_git_commit"],
        "category_seed_reports": [
            {"category": report["category"], "seed": report["seed"],
             "threshold": report["threshold"], "metrics": report["metrics"], "pixel": report["pixel"]}
            for report in reports
        ],
        "category_means_across_seeds": per_category,
        "macro_means_across_categories": {
            name: float(np.mean([row[name] for row in per_category.values()])) for name in fields
        },
        "automatic_pass_allowed": False,
        "post_final_tuning_allowed": False,
    }
    output = project_root / "artifacts" / FINAL_ROOT_NAME / "summary.json"
    if output.exists():
        if read_json(output) != summary:
            raise ValueError("Existing final summary differs; it was not overwritten")
    else:
        _atomic_write_json(output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "fit", "evaluate", "summarize"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--category", choices=sorted(FOLLOWUP_CATEGORIES))
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = args.project_root.resolve(strict=True)
        if args.command == "summarize":
            result = summarize(project_root)
        else:
            if args.category is None or args.seed is None:
                raise ValueError("--category and --seed are required")
            gate = validate_gate(project_root, args.category, args.seed)
            if args.command == "verify":
                result = {"status": "final_gate_verified_unopened", "category": args.category, "seed": args.seed}
            elif args.command == "fit":
                result = prepare_fit(gate, args.category, args.seed)
            else:
                result = evaluate(gate, args.category, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
