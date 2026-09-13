"""Manifest-locked PaDiM fit and normal-only calibration entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from defectops.manifest import verify_manifest


EXPECTED_PROTOCOL = "A-normal-only-v1"
EXPECTED_MODEL = "padim"
EXPECTED_PACKAGES = {
    "anomalib": "2.2.0",
    "lightning": "2.5.5",
    "numpy": "2.3.3",
    "opencv-python": "4.11.0.86",
    "pillow": "11.3.0",
    "pytorch-lightning": "2.5.5",
    "requests": "2.34.2",
    "timm": "1.0.20",
    "torch": "2.8.0+cu128",
    "torchmetrics": "1.8.2",
    "torchvision": "0.23.0+cu128",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checksum(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def higher_quantile(values: list[float], quantile: float) -> float:
    """Match NumPy's ``method='higher'`` quantile without another runtime dependency."""
    if not values:
        raise ValueError("Cannot calculate a quantile from an empty list")
    if not 0 <= quantile <= 1 or not math.isfinite(quantile):
        raise ValueError("quantile must be finite and between 0 and 1")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All scores must be finite")
    ordered = sorted(values)
    index = math.ceil((len(ordered) - 1) * quantile)
    return float(ordered[index])


def records_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Return sorted, validated records for one frozen logical split."""
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest records must be a list")
    selected = [record for record in records if isinstance(record, dict) and record.get("split") == split]
    selected.sort(key=lambda record: str(record.get("image_path", "")))
    if not selected:
        raise ValueError(f"Manifest split is empty: {split}")
    paths = [record.get("image_path") for record in selected]
    if not all(isinstance(path, str) and path for path in paths) or len(paths) != len(set(paths)):
        raise ValueError(f"Manifest split has missing or duplicate image paths: {split}")
    if split in {"train", "calibration"} and any(record.get("label") != 0 for record in selected):
        raise ValueError(f"{split} must contain normal images only")
    return selected


def load_experiment(config_path: Path, manifest: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """Load and validate the first PaDiM experiment settings."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = config.get("experiment", {})
    calibration = config.get("calibration", {})
    benchmark = config.get("benchmark", {})
    if config.get("protocol") != EXPECTED_PROTOCOL or manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"Expected protocol {EXPECTED_PROTOCOL}")
    if config.get("split_seed") != manifest.get("split_seed"):
        raise ValueError("Config and manifest split seeds differ")
    if config.get("categories") != manifest.get("categories") or config.get("categories") != ["transistor"]:
        raise ValueError("The first run must use only the transistor category")
    if experiment.get("model") != EXPECTED_MODEL:
        raise ValueError("The first run must use PaDiM")
    model_seeds = experiment.get("model_seeds")
    if not isinstance(model_seeds, list) or not model_seeds or not all(type(item) is int for item in model_seeds):
        raise ValueError("experiment.model_seeds must be a non-empty integer list")
    selected_seed = model_seeds[0] if seed is None else seed
    if selected_seed not in model_seeds:
        raise ValueError(f"seed must be one of the configured model seeds: {model_seeds}")
    if calibration.get("quantile_method") != "higher" or calibration.get("score_space") != "raw":
        raise ValueError("Calibration must use the higher quantile in raw score space")
    if calibration.get("decision_rule") != "score >= threshold":
        raise ValueError("Unexpected calibration decision rule")
    image_quantile = calibration.get("image_quantile")
    if not isinstance(image_quantile, (int, float)) or not 0 < float(image_quantile) < 1:
        raise ValueError("calibration.image_quantile must be between 0 and 1")
    layers = experiment.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(layer, str) for layer in layers):
        raise ValueError("experiment.layers must be a non-empty string list")
    batch_size = int(benchmark.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("The first PaDiM run is locked to batch_size=1")
    return {
        "model": EXPECTED_MODEL,
        "backbone": str(experiment.get("backbone")),
        "layers": layers,
        "n_features": int(experiment.get("n_features")),
        "image_size": int(experiment.get("image_size")),
        "model_seed": selected_seed,
        "split_seed": int(config["split_seed"]),
        "batch_size": batch_size,
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


def git_state(project_root: Path) -> dict[str, str]:
    """Require and return a clean, committed Git state."""
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty = run("status", "--porcelain")
    if dirty:
        raise RuntimeError("Refusing to run from a dirty Git worktree; commit or revert code/config changes first")
    return {"commit": commit, "branch": branch}


def installed_versions() -> dict[str, str]:
    """Validate the tested Windows package combination."""
    actual: dict[str, str] = {}
    mismatches = []
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = "missing"
        if actual[package] != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual[package]}")
    if mismatches:
        raise RuntimeError("Package version mismatch:\n" + "\n".join(mismatches))
    return actual


def _safe_data_path(data_root: Path, relative_path: str) -> Path:
    path = (data_root / relative_path).resolve(strict=True)
    path.relative_to(data_root)
    return path


def run_fit_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Fit PaDiM on frozen train records and calibrate on frozen normal records."""
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output_root = (project_root / "artifacts").resolve()
    output_path = args.output.resolve()
    output_path.relative_to(output_root)
    if output_path.exists():
        raise FileExistsError(f"Run output already exists and will not be overwritten: {output_path}")

    manifest = read_json(manifest_path)
    verify_manifest(data_root, manifest)
    experiment = load_experiment(config_path, manifest, args.seed)
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
        raise RuntimeError(f"The lab run requires Python 3.12, found {platform.python_version()}")

    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchvision
    from anomalib.models import Padim
    from PIL import Image

    if torch.__version__ != EXPECTED_PACKAGES["torch"] or torchvision.__version__ != EXPECTED_PACKAGES["torchvision"]:
        raise RuntimeError("Imported torch/torchvision versions differ from the validated combination")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the lab fit on CPU")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA runtime 12.8, found {torch.version.cuda}")

    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output_path / "pilot.toml")
    shutil.copyfile(manifest_path, output_path / "manifest.json")

    seed = experiment["model_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def load_image(record: dict[str, Any]) -> torch.Tensor:
        image_path = _safe_data_path(data_root, record["image_path"])
        with Image.open(image_path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)
        tensor = functional.interpolate(
            tensor.unsqueeze(0),
            size=(experiment["image_size"], experiment["image_size"]),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return (tensor - mean) / std

    def batches(records: list[dict[str, Any]]):
        batch_size = experiment["batch_size"]
        for start in range(0, len(records), batch_size):
            subset = records[start : start + batch_size]
            yield subset, torch.stack([load_image(record) for record in subset]).to(device)

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    model = Padim(
        backbone=experiment["backbone"],
        layers=experiment["layers"],
        pre_trained=True,
        n_features=experiment["n_features"],
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).to(device)
    model.model.train()
    print(f"Fitting PaDiM with {len(train_records)} frozen normal images...")
    with torch.no_grad():
        for index, (_, image_batch) in enumerate(batches(train_records), start=1):
            model.model(image_batch)
            if index % 25 == 0 or index == len(train_records):
                print(f"  train {index}/{len(train_records)}")
    model.fit()
    model._is_fitted.fill_(True)
    model.model.eval()

    print(f"Scoring {len(calibration_records)} frozen normal calibration images...")
    calibration_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        processed = 0
        for records, image_batch in batches(calibration_records):
            predictions = model.model(image_batch)
            scores = predictions.pred_score.detach().cpu().reshape(-1).tolist()
            for record, score in zip(records, scores, strict=True):
                value = float(score)
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite calibration score for {record['image_path']}")
                calibration_rows.append({"image_path": record["image_path"], "raw_score": value})
            processed += len(records)
            if processed % 10 == 0 or processed == len(calibration_records):
                print(f"  calibration {processed}/{len(calibration_records)}")

    threshold = higher_quantile(
        [row["raw_score"] for row in calibration_rows],
        experiment["image_quantile"],
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
        "status": "fit_calibrated",
        "test_evaluated": False,
        "purpose": "PaDiM fit on frozen train normals and threshold calibration on frozen calibration normals",
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
            "test_reserved_not_loaded": len(test_records),
        },
        "sample_ids": {
            "train": [record["image_path"] for record in train_records],
            "calibration": [record["image_path"] for record in calibration_records],
        },
        "calibration": {
            "threshold": threshold,
            "quantile": experiment["image_quantile"],
            "method": experiment["quantile_method"],
            "score_space": experiment["score_space"],
            "decision_rule": experiment["decision_rule"],
            "minimum_raw_score": min(row["raw_score"] for row in calibration_rows),
            "maximum_raw_score": max(row["raw_score"] for row in calibration_rows),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": versions,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "pid": os.getpid(),
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checksum(checkpoint_path),
            "calibration_scores": calibration_path.name,
            "calibration_scores_sha256": checksum(calibration_path),
            "config_snapshot": "pilot.toml",
            "manifest_snapshot": "manifest.json",
        },
    }
    write_json(output_path / "run.json", run)
    print(json.dumps({
        "status": run["status"],
        "run_id": run["run_id"],
        "threshold": threshold,
        "elapsed_seconds": elapsed_seconds,
        "output": str(output_path),
    }, indent=2, ensure_ascii=False))
    return run


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Parent directory of the transistor dataset")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "transistor-v1.json",
    )
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "pilot.toml")
    parser.add_argument("--seed", type=int, help="Configured model seed; defaults to the first seed")
    parser.add_argument("--output", type=Path, required=True, help="New immutable directory below artifacts/")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the manifest-locked fit and calibration command."""
    args = build_parser().parse_args(argv)
    try:
        run_fit_calibration(args)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary records a concise failure
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
