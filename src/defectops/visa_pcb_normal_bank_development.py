"""Post-test exploratory VisA normal-bank fit; never opens official test payloads."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import tarfile
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from defectops.padim_experiment import git_state, higher_quantile
from defectops.realiad_followup_development import _seed_everything, candidate_image_score
from defectops.realiad_followup_final import _gpu_modules, _model, _predict, _preprocess
from defectops.visa_pcb_external_evaluation import _jpeg, _member_bytes, _pixel_sha
from defectops.visa_pcb_external_protocol import (
    EXPECTED_COUNTS, _safe_member, build_plan, canonical, load_config,
)
from defectops.visa_pcb_external_registration import CSV_MEMBER, file_sha256


EXPECTED = {
    "schema_version": 1,
    "status": "post_test_exploratory_normal_bank_development",
    "protocol": "A-normal-only-v2",
    "source_evaluation_handoff_sha256": "ba2f5200757e212f97f0a431679a429aa068ee9140b0a28a760641e9427a0d2a",
    "source_preflight_handoff_sha256": "c6568b6691d03c1a244da29a7505b0d6f56dc2970f618e8f7dacd9d35d4ce69d",
    "source_archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
    "source_plan_sha256": "e134445218bfb71c3cbcfcad9da45548024ef6166a2acd8b0ab64ff23e5ba34a",
    "categories": ["pcb1", "pcb2", "pcb3", "pcb4"],
    "first_pilot_category": "pcb3", "model_seed": 42,
    "backbone": "wide_resnet50_2", "layers": ["layer1", "layer2"],
    "pre_trained": True, "image_size": 256,
    "preprocessing": "stretch_to_square_bilinear_antialias_imagenet",
    "coreset_sampling_ratio": 0.10, "num_neighbors": 9,
    "image_score": "mean_of_top_map_fraction", "top_fraction": 0.01,
    "image_quantile": 0.95, "pixel_quantile": 0.995,
    "quantile_method": "higher",
    "calibration_source": "existing_56_official_train_normals_per_category",
    "fit_source": "remaining_official_train_normals_after_calibration_exclusion",
    "official_test_payloads_allowed": False,
    "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
}


def load_execution(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config != EXPECTED:
        raise ValueError("Post-test VisA normal-bank execution contract differs")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def training_members(csv_bytes: bytes, plan: dict[str, Any], category: str) -> list[str]:
    """Use official train normals except the already-fixed 56 calibration images."""
    if category not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown VisA PCB category: {category}")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
        raise ValueError("Unexpected official VisA split CSV columns")
    normal_paths: list[str] = []
    for row in reader:
        if row["object"] != category or row["split"] != "train":
            continue
        if row["label"] != "normal" or row["mask"]:
            raise ValueError(f"Unexpected VisA train label/mask: {row['image']}")
        normal_paths.append(_safe_member(row["image"], category, ".jpg"))
    calibration_paths = {
        record["source_image_member"] for record in plan["records"]
        if record["category"] == category and record["protocol_split"] == "calibration"
    }
    if (
        len(normal_paths) != EXPECTED_COUNTS[category]["train_normal"]
        or len(set(normal_paths)) != len(normal_paths)
        or len(calibration_paths) != 56
        or not calibration_paths.issubset(normal_paths)
    ):
        raise ValueError("Official train/calibration split differs from the locked plan")
    selected = sorted(set(normal_paths) - calibration_paths)
    if len(selected) != len(normal_paths) - 56:
        raise ValueError("Train/calibration paths overlap or have unexpected count")
    return selected


def checked_sources(
    project: Path, archive_path: Path, category: str, execution: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read only metadata CSV and prior calibration audit, not test JPEG/PNG."""
    plan_path = project / "data" / "manifests" / "visa-pcb-external-plan-v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    plan_hash = unsigned.pop("plan_sha256", None)
    if (
        plan_hash != hashlib.sha256(canonical(unsigned)).hexdigest()
        or plan_hash != execution["source_plan_sha256"]
    ):
        raise ValueError("Original VisA metadata plan hash differs")
    if file_sha256(project / "artifacts" / "handoff" / "visa-pcb-external-evaluation-v2.zip") != execution["source_evaluation_handoff_sha256"]:
        raise ValueError("Previously reviewed VisA v2 evaluation handoff differs")
    preflight_path = project / "artifacts" / "handoff" / "visa-pcb-external-preflight-v1.zip"
    if file_sha256(preflight_path) != execution["source_preflight_handoff_sha256"]:
        raise ValueError("Original normal calibration audit handoff differs")
    with zipfile.ZipFile(preflight_path) as handoff:
        if sorted(handoff.namelist()) != ["calibration-audit.json", "preflight.json"] or handoff.testzip() is not None:
            raise ValueError("Normal calibration audit ZIP members/CRC differ")
        audit = json.loads(handoff.read("calibration-audit.json"))
    if audit.get("status") != "calibration_jpeg_decoded_no_exact_pixel_duplicates" or audit.get("count") != 224:
        raise ValueError("Original normal calibration audit status/count differs")
    expected_calibration = {
        record["source_image_member"] for record in plan["records"]
        if record["protocol_split"] == "calibration"
    }
    audit_by_name = {row["image"]: row for row in audit["records"]}
    if len(audit_by_name) != 224 or set(audit_by_name) != expected_calibration:
        raise ValueError("Original normal calibration audit paths differ")
    if file_sha256(archive_path) != execution["source_archive_sha256"]:
        raise ValueError("Registered VisA source archive SHA-256 differs")
    protocol = load_config(project / "configs" / "visa-pcb-external-v1.toml")
    with tarfile.open(archive_path, "r:") as archive:
        csv_bytes = _member_bytes(archive, CSV_MEMBER)
    if build_plan(csv_bytes, protocol) != plan:
        raise ValueError("Official CSV and original VisA metadata plan differ")
    train_paths = training_members(csv_bytes, plan, category)
    calibration = [
        record for record in plan["records"]
        if record["category"] == category and record["protocol_split"] == "calibration"
    ]
    if len(calibration) != 56 or any(record["label"] != 0 for record in calibration):
        raise ValueError("Expected 56 fixed normal calibration images")
    return plan, train_paths, calibration, audit_by_name


def _json_once(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validated_existing(output: Path, category: str, plan: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    try:
        report = json.loads((output / "fit.json").read_text(encoding="utf-8"))
        file_hashes = {
            "checkpoint_sha256": file_sha256(output / "model.pt"),
            "calibration_rows_sha256": file_sha256(output / "calibration.jsonl"),
            "training_audit_sha256": file_sha256(output / "training-audit.jsonl"),
        }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Existing fit is incomplete; it was not changed: {output}") from exc
    if (
        report.get("status") != "fit_calibrated_post_test_exploratory_normal_bank"
        or report.get("category") != category
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("execution_config_sha256") != execution["config_sha256"]
        or report.get("source_archive_sha256") != execution["source_archive_sha256"]
        or report.get("source_evaluation_handoff_sha256") != execution["source_evaluation_handoff_sha256"]
        or report.get("source_preflight_handoff_sha256") != execution["source_preflight_handoff_sha256"]
        or report.get("normal_train_count") != EXPECTED_COUNTS[category]["train_normal"] - 56
        or report.get("normal_calibration_count") != 56
        or report.get("model_seed") != execution["model_seed"]
        or report.get("official_test_images_decoded") != 0
        or report.get("official_test_scores_computed") != 0
        or report.get("automatic_pass_allowed") is not False
        or any(report.get(key) != value for key, value in file_hashes.items())
    ):
        raise ValueError(f"Existing fit is incomplete or differs; it was not changed: {output}")
    return report


def audit_training_normals(
    archive_path: Path, train_paths: list[str], calibration: list[dict[str, Any]],
    audit_by_name: dict[str, dict[str, Any]], category: str,
) -> list[dict[str, Any]]:
    """Fail on unreadable/duplicate normal data before the expensive GPU fit."""
    known_calibration_pixels = {
        audit_by_name[row["source_image_member"]]["decoded_pixel_sha256"] for row in calibration
    }
    seen_training_pixels: set[str] = set()
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:") as archive:
        for index, name in enumerate(train_paths, start=1):
            raw = _member_bytes(archive, name)
            image = _jpeg(raw, name)
            pixel_sha = _pixel_sha(image)
            if pixel_sha in known_calibration_pixels or pixel_sha in seen_training_pixels:
                raise ValueError(f"Exact normal train/calibration duplicate: {name}")
            seen_training_pixels.add(pixel_sha)
            rows.append({
                "image": name, "image_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_pixel_sha256": pixel_sha, "width": image.width, "height": image.height,
            })
            if index % 100 == 0 or index == len(train_paths):
                print(f"{category} normal data audit {index}/{len(train_paths)}", flush=True)
    return rows


def _fit_one(
    archive_path: Path, stage: Path, category: str, execution: dict[str, Any], plan: dict[str, Any],
    train_paths: list[str], calibration: list[dict[str, Any]], audit_by_name: dict[str, dict[str, Any]],
    git: dict[str, str],
) -> dict[str, Any]:
    overall_started = time.perf_counter()
    training_audit = audit_training_normals(archive_path, train_paths, calibration, audit_by_name, category)
    audit_seconds = time.perf_counter() - overall_started
    gpu = _gpu_modules()
    torch, _, Patchcore, _, device = gpu
    _seed_everything(execution["model_seed"], torch)
    model = Patchcore(
        backbone=execution["backbone"], layers=execution["layers"], pre_trained=True,
        coreset_sampling_ratio=execution["coreset_sampling_ratio"],
        num_neighbors=execution["num_neighbors"], pre_processor=False, post_processor=False,
        evaluator=False, visualizer=False,
    ).to(device)
    model.model.train()
    model.model.feature_extractor.eval()
    audited_training = {row["image"]: row for row in training_audit}
    protocol = {"base_model": {
        "backbone": execution["backbone"], "image_size": execution["image_size"],
        "coreset_sampling_ratio": execution["coreset_sampling_ratio"],
        "num_neighbors": execution["num_neighbors"],
    }}
    started = time.perf_counter()
    with tarfile.open(archive_path, "r:") as archive:
        with torch.no_grad():
            for index, name in enumerate(train_paths, start=1):
                raw = _member_bytes(archive, name)
                if hashlib.sha256(raw).hexdigest() != audited_training[name]["image_sha256"]:
                    raise ValueError(f"Normal train JPEG changed after audit: {name}")
                image = _jpeg(raw, name)
                model.model(_preprocess(image, execution["image_size"], gpu))
                if index % 25 == 0 or index == len(train_paths):
                    print(f"{category} normal fit {index}/{len(train_paths)}", flush=True)
        embedding_count = sum(int(value.shape[0]) for value in model.model.embedding_store)
        print(f"{category} selecting 10% normal-patch coreset from {embedding_count} embeddings...", flush=True)
        model.fit()
        model._is_fitted.fill_(True)
        model.model.eval()
        memory_shape = list(model.model.memory_bank.shape)
        if len(memory_shape) != 2 or memory_shape[0] < execution["num_neighbors"]:
            raise RuntimeError("Unexpected VisA normal memory bank shape")
        fit_seconds = time.perf_counter() - started
        checkpoint = stage / "model.pt"
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save({
            "format_version": 1, "category": category, "layers": execution["layers"],
            "model_seed": execution["model_seed"], "git_commit": git["commit"],
            "manifest_sha256": plan["plan_sha256"],
            "execution_config_sha256": execution["config_sha256"],
            "memory_bank_shape": memory_shape, "state_dict": state,
        }, checkpoint)
        del state
        calibration_rows: list[dict[str, Any]] = []
        maps: list[np.ndarray] = []
        for index, record in enumerate(calibration, start=1):
            name = record["source_image_member"]
            raw = _member_bytes(archive, name)
            audited = audit_by_name[name]
            if hashlib.sha256(raw).hexdigest() != audited["image_sha256"]:
                raise ValueError(f"Normal calibration JPEG checksum differs: {name}")
            image = _jpeg(raw, name)
            if _pixel_sha(image) != audited["decoded_pixel_sha256"]:
                raise ValueError(f"Normal calibration decoded pixels differ: {name}")
            raw_score, anomaly_map = _predict(model, image, protocol, gpu)
            score = candidate_image_score(raw_score, anomaly_map, {
                "image_score": execution["image_score"], "top_fraction": execution["top_fraction"],
            })
            calibration_rows.append({"image": name, "raw_score": raw_score, "score": score})
            maps.append(anomaly_map)
            if index % 10 == 0 or index == len(calibration):
                print(f"{category} normal calibration {index}/{len(calibration)}", flush=True)
    threshold = higher_quantile([row["score"] for row in calibration_rows], execution["image_quantile"])
    pixel_threshold = float(np.quantile(
        np.stack(maps).reshape(-1), execution["pixel_quantile"], method="higher",
    ))
    if not math.isfinite(threshold) or not math.isfinite(pixel_threshold):
        raise ValueError("Non-finite normal-only threshold")
    _jsonl_once(stage / "training-audit.jsonl", training_audit)
    _jsonl_once(stage / "calibration.jsonl", calibration_rows)
    del model, maps
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = _model(protocol, checkpoint, {
        "category": category, "seed": execution["model_seed"],
        "manifest_sha256": plan["plan_sha256"],
        "execution_config_sha256": execution["config_sha256"],
    }, gpu)
    differences: list[float] = []
    with tarfile.open(archive_path, "r:") as archive:
        for record, previous in zip(calibration, calibration_rows, strict=True):
            name = record["source_image_member"]
            image = _jpeg(_member_bytes(archive, name), name)
            raw_score, anomaly_map = _predict(reloaded, image, protocol, gpu)
            score = candidate_image_score(raw_score, anomaly_map, {
                "image_score": execution["image_score"], "top_fraction": execution["top_fraction"],
            })
            differences.append(abs(score - previous["score"]))
    reload_max = max(differences)
    if reload_max > 1e-4:
        raise ValueError(f"Checkpoint reload changed normal calibration scores: {reload_max}")
    report = {
        "schema_version": 1, "status": "fit_calibrated_post_test_exploratory_normal_bank",
        "comparison_status": "post_test_exploratory_after_visa_802_test_observed",
        "category": category, "git": git, "model_seed": execution["model_seed"],
        "plan_sha256": plan["plan_sha256"], "execution_config_sha256": execution["config_sha256"],
        "source_archive_sha256": execution["source_archive_sha256"],
        "source_evaluation_handoff_sha256": execution["source_evaluation_handoff_sha256"],
        "source_preflight_handoff_sha256": execution["source_preflight_handoff_sha256"],
        "normal_train_count": len(train_paths), "normal_calibration_count": len(calibration),
        "normal_train_audit_count": len(training_audit),
        "official_test_images_decoded": 0, "official_test_scores_computed": 0,
        "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
        "backbone": execution["backbone"], "layers": execution["layers"],
        "image_size": execution["image_size"], "coreset_sampling_ratio": execution["coreset_sampling_ratio"],
        "num_neighbors": execution["num_neighbors"], "image_score": execution["image_score"],
        "top_fraction": execution["top_fraction"], "image_quantile": execution["image_quantile"],
        "pixel_quantile": execution["pixel_quantile"],
        "embedding_count_before_coreset": embedding_count, "memory_bank_shape": memory_shape,
        "normal_data_audit_seconds": audit_seconds,
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - overall_started,
        "image_threshold": threshold, "pixel_threshold": pixel_threshold,
        "reload_max_absolute_score_difference": reload_max,
        "checkpoint_sha256": file_sha256(checkpoint),
        "calibration_rows_sha256": file_sha256(stage / "calibration.jsonl"),
        "training_audit_sha256": file_sha256(stage / "training-audit.jsonl"),
        "limitations": [
            "Official VisA test was already observed before this development fit; no confirmatory claim is allowed.",
            "Only official train normals excluding the 56 fixed calibration normals were used for the memory bank.",
            "No official test image or mask payload was read or scored in this fit.",
            "No automatic PASS or production decision is authorized.",
        ],
    }
    _json_once(stage / "fit.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project.resolve(strict=True)
    archive_path = args.archive.resolve(strict=True)
    execution = load_execution(project / "configs" / "visa-pcb-normal-bank-development-v1.toml")
    if args.category not in execution["categories"]:
        raise ValueError(f"Category must be one of {execution['categories']}")
    git = git_state(project)
    print("Checking locked source, official split CSV and existing normal calibration audit...", flush=True)
    plan, train_paths, calibration, audited = checked_sources(project, archive_path, args.category, execution)
    print(
        f"{args.category}: fitting {len(train_paths)} train normals, then calibrating "
        f"{len(calibration)} disjoint normals; no official test decode/scoring.", flush=True,
    )
    root = project / "artifacts" / "visa-pcb-normal-bank-development-v1"
    output = root / args.category
    handoff = project / "artifacts" / "handoff" / f"visa-pcb-normal-bank-{args.category}-v1.zip"
    root.mkdir(parents=True, exist_ok=True)
    if output.exists():
        report = _validated_existing(output, args.category, plan, execution)
        print(f"Verified existing completed fit: {output}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix=f".{args.category}-", dir=root) as temporary:
            stage = Path(temporary)
            report = _fit_one(
                archive_path, stage, args.category, execution, plan, train_paths, calibration, audited, git,
            )
            os.replace(stage, output)
    handoff.parent.mkdir(parents=True, exist_ok=True)
    if handoff.exists():
        with zipfile.ZipFile(handoff) as archive:
            if sorted(archive.namelist()) != ["calibration.jsonl", "fit.json", "training-audit.jsonl"] or archive.testzip() is not None:
                raise ValueError(f"Existing handoff is invalid; it was not changed: {handoff}")
            for name in archive.namelist():
                if archive.read(name) != (output / name).read_bytes():
                    raise ValueError(f"Existing handoff differs from verified fit: {handoff}")
    else:
        with tempfile.NamedTemporaryFile(
            prefix=f".{handoff.stem}-", suffix=".zip", dir=handoff.parent, delete=False,
        ) as stream:
            temporary_handoff = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary_handoff, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ("fit.json", "training-audit.jsonl", "calibration.jsonl"):
                    archive.write(output / name, name)
            os.rename(temporary_handoff, handoff)
        finally:
            temporary_handoff.unlink(missing_ok=True)
    return {
        "status": report["status"], "category": args.category,
        "normal_train_count": report["normal_train_count"],
        "normal_calibration_count": report["normal_calibration_count"],
        "image_threshold": report["image_threshold"],
        "memory_bank_shape": report["memory_bank_shape"],
        "fit_seconds": report["fit_seconds"],
        "official_test_images_decoded": 0,
        "handoff_zip": str(handoff), "handoff_sha256": file_sha256(handoff),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--category", choices=list(EXPECTED_COUNTS), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
