"""One locked, resumable external-transfer evaluation on official VisA PCB1–4 test."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defectops.padim_experiment import git_state, higher_quantile
from defectops.realiad_followup_development import (
    _candidate_metrics, _pixel_metrics, _seed_everything, candidate_image_score,
)
from defectops.realiad_followup_final import (
    _gpu_modules, _model, _predict, _read_sample, _write_sample, validate_gate,
)
from defectops.visa_pcb_external_preflight import (
    audit_calibration, checked_registration, load_preflight_config, _write_json_once,
)
from defectops.visa_pcb_external_protocol import canonical, load_config
from defectops.visa_pcb_external_registration import file_sha256


EXPECTED_EVALUATION = {
    "schema_version": 1,
    "status": "pretest_external_transfer_evaluation_locked",
    "protocol": "A-normal-only-v2",
    "preflight_handoff_sha256": "c6568b6691d03c1a244da29a7505b0d6f56dc2970f618e8f7dacd9d35d4ce69d",
    "preflight_git_commit": "1489ccd8e78bcda3fc10bcccda721417d711e482",
    "realiad_checkpoint_sha256": "cd0e12d3d2daadfc1f30fde6e666002287af29cc610a870a6196899468c0de13",
    "calibration_images_per_category": 56,
    "external_final_normal_total": 402,
    "external_final_anomaly_total": 400,
    "image_quantile": 0.95,
    "quantile_method": "higher",
    "image_decision_rule": "score >= threshold",
    "baseline_map_top_fraction": 0.01,
    "challenger_map_top_fraction": 0.0025,
    "pixel_ground_truth_threshold": 128,
    "pixel_calibration_quantile": 0.995,
    "pixel_aupro_fpr_limit": 0.30,
    "pixel_aupro_fpr_points": 301,
    "fail_on_unreadable_image_or_mask": True,
    "fail_on_exact_calibration_test_pixel_duplicate": True,
    "fail_on_exact_test_pixel_duplicate": True,
    "fail_on_empty_anomaly_mask": True,
    "near_duplicate_dhash_max_distance": 4,
    "near_duplicates_report_only": True,
    "paired_bootstrap_replicates": 5000,
    "paired_bootstrap_seed": 20260924,
    "paired_bootstrap_quantile_method": "linear",
    "confidence_level": 0.95,
    "maximum_challenger_overkill_per_category": 0.10,
    "maximum_category_escape_worsening": 0.05,
    "require_positive_paired_ci_lower_bound": True,
    "failed_images_count_in_denominator": True,
    "post_test_tuning_allowed": False,
    "automatic_pass_allowed": False,
}

EXPECTED_MASK_CORRECTION_V2 = {
    **EXPECTED_EVALUATION,
    "schema_version": 2,
    "status": "post_test_mask_parser_correction_exploratory",
    "mask_binarization": "nonzero_before_nearest",
    "protocol_deviation": "Original VisA class-ID mask value 1 was misread as background in the first test audit",
}


def load_evaluation_config(path: Path, *, mask_correction_v2: bool = False) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = EXPECTED_MASK_CORRECTION_V2 if mask_correction_v2 else EXPECTED_EVALUATION
    if config != expected:
        raise ValueError("VisA evaluation contract differs from the selected locked values")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def checked_preflight(handoff_path: Path, plan: dict[str, Any], config: dict[str, Any],
                      preflight_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if file_sha256(handoff_path) != config["preflight_handoff_sha256"]:
        raise ValueError("VisA preflight ZIP checksum differs")
    with zipfile.ZipFile(handoff_path) as handoff:
        if sorted(handoff.namelist()) != ["calibration-audit.json", "preflight.json"] or handoff.testzip() is not None:
            raise ValueError("VisA preflight ZIP members or CRC differ")
        report = json.loads(handoff.read("preflight.json"))
        audit = json.loads(handoff.read("calibration-audit.json"))
    selected = {(row["category"], row["source_image_member"]) for row in plan["records"]
                if row["protocol_split"] == "calibration"}
    audited = {(row["category"], row["image"]) for row in audit["records"]}
    if (
        report["status"] != "preflight_passed_external_final_unopened"
        or report["git"]["commit"] != config["preflight_git_commit"]
        or report["plan_sha256"] != plan["plan_sha256"]
        or report["protocol_config_sha256"] != plan["config_sha256"]
        or report["preflight_config_sha256"] != preflight_config["config_sha256"]
        or report["registration_handoff_sha256"] != preflight_config["registration_handoff_sha256"]
        or report["realiad_checkpoint_sha256"] != config["realiad_checkpoint_sha256"]
        or report["source_archive_sha256"] != preflight_config["source_archive_sha256"]
        or report["calibration_audit_sha256"] != hashlib.sha256(canonical(audit)).hexdigest()
        or report["calibration_images_decoded"] != 224
        or report["external_final_images_decoded"] != 0
        or report["external_final_scores_computed"] != 0
        or report["automatic_pass_allowed"] is not False
        or audit["status"] != "calibration_jpeg_decoded_no_exact_pixel_duplicates"
        or audit["external_final_payloads_decoded"] is not False
        or audit["exact_pixel_duplicate_count"] != 0
        or audit["count"] != 224
        or len(audit["records"]) != len(selected)
        or len(selected) != 224
        or audited != selected
        or len({row["decoded_pixel_sha256"] for row in audit["records"]}) != 224
    ):
        raise ValueError("VisA preflight evidence differs from the locked plan")
    return report, audit


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise ValueError(f"Missing VisA archive member: {name}") from exc
    if not member.isfile() or member.size <= 0:
        raise ValueError(f"Invalid VisA archive member: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Unreadable VisA archive member: {name}")
    data = stream.read()
    if len(data) != member.size:
        raise ValueError(f"Incomplete VisA archive member: {name}")
    return data


def _jpeg(data: bytes, name: str) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.mode != "RGB":
            raise ValueError(f"VisA image is not RGB JPEG: {name}")
        image.load()
        return image.copy()


def _pixel_sha(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()


def _dhash(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")


def _binary_mask(mask: Image.Image, output_size: int, threshold: int,
                 binarization: str | None = None) -> np.ndarray:
    if binarization == "nonzero_before_nearest":
        raw = np.asarray(mask)
        if raw.ndim != 2:
            raise ValueError(f"Expected a single-channel VisA class-ID mask, got {mask.mode}")
        source = Image.fromarray(np.where(raw != 0, 255, 0).astype(np.uint8))
    elif binarization is None:
        source = mask.convert("L")
    else:
        raise ValueError(f"Unknown VisA mask binarization: {binarization}")
    resized = source.resize((output_size, output_size), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) >= threshold


def audit_final_payloads(archive_path: Path, final_records: list[dict[str, Any]],
                         calibration_audit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Open every sealed test image/mask once in the same run that will score them."""
    if len(final_records) != config["external_final_normal_total"] + config["external_final_anomaly_total"]:
        raise ValueError("Official VisA final image count differs")
    calibration_pixels = {row["decoded_pixel_sha256"]: row["image"] for row in calibration_audit["records"]}
    calibration_hashes: dict[str, int] = {}
    seen_test: dict[str, str] = {}
    results = []
    with tarfile.open(archive_path, "r:") as archive:
        for calibration in calibration_audit["records"]:
            data = _member_bytes(archive, calibration["image"])
            if hashlib.sha256(data).hexdigest() != calibration["image_sha256"]:
                raise ValueError("Calibration bytes changed before test was opened")
            with _jpeg(data, calibration["image"]) as image:
                if _pixel_sha(image) != calibration["decoded_pixel_sha256"]:
                    raise ValueError("Calibration pixels changed before test was opened")
                calibration_hashes[calibration["image"]] = _dhash(image)
        for index, row in enumerate(final_records, start=1):
            name = row["source_image_member"]
            data = _member_bytes(archive, name)
            with _jpeg(data, name) as image:
                pixel_sha = _pixel_sha(image)
                width, height = image.size
                perceptual_hash = _dhash(image)
            if pixel_sha in calibration_pixels:
                raise ValueError(f"Exact calibration/test image duplicate: {calibration_pixels[pixel_sha]} and {name}")
            if pixel_sha in seen_test:
                raise ValueError(f"Exact duplicate within official test: {seen_test[pixel_sha]} and {name}")
            seen_test[pixel_sha] = name
            mask_sha = None
            mask_nonzero = 0
            if row["label"] == 1:
                mask_name = row["source_mask_member"]
                mask_data = _member_bytes(archive, mask_name)
                mask_sha = hashlib.sha256(mask_data).hexdigest()
                with Image.open(io.BytesIO(mask_data)) as mask:
                    if mask.format != "PNG" or mask.size != (width, height):
                        raise ValueError(f"VisA mask is not PNG or dimensions differ: {mask_name}")
                    mask.load()
                    mask_nonzero = int(np.count_nonzero(_binary_mask(
                        mask, 256, config["pixel_ground_truth_threshold"], config.get("mask_binarization")
                    )))
                if mask_nonzero == 0:
                    raise ValueError(f"Empty anomaly mask after fixed resizing: {mask_name}")
            elif row["source_mask_member"]:
                raise ValueError(f"Normal VisA test image unexpectedly has a mask: {name}")
            results.append({
                "category": row["category"], "image": name, "mask": row["source_mask_member"],
                "label": row["label"], "width": width, "height": height,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "decoded_pixel_sha256": pixel_sha, "dhash64": f"{perceptual_hash:016x}",
                "mask_sha256": mask_sha, "mask_nonzero_256": mask_nonzero,
            })
            if index % 50 == 0 or index == len(final_records):
                print(f"test payload audit {index}/{len(final_records)}", flush=True)
    # Perceptual similarity is a review flag, not proof of the same physical board.
    near_count = 0
    near_examples = []
    for row in results:
        observed = int(row["dhash64"], 16)
        for cal_name, cal_hash in calibration_hashes.items():
            distance = (observed ^ cal_hash).bit_count()
            if distance <= config["near_duplicate_dhash_max_distance"]:
                near_count += 1
                if len(near_examples) < 20:
                    near_examples.append({"calibration_image": cal_name, "test_image": row["image"],
                                          "dhash_hamming_distance": distance})
    return {
        "schema_version": 1, "status": "all_official_test_payloads_audited_before_scoring",
        "count": len(results),
        "normal_count": sum(row["label"] == 0 for row in results),
        "anomaly_count": sum(row["label"] == 1 for row in results),
        "mask_count": sum(bool(row["mask"]) for row in results),
        "exact_duplicate_count": 0,
        "calibration_test_dhash_near_pair_count": near_count,
        "calibration_test_dhash_near_examples": near_examples,
        "near_duplicate_heuristic_is_identity_proof": False,
        "records": results,
    }


def paired_macro_escape_difference(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Paired, anomaly-stratified by category; do not compare unpaired aggregate counts."""
    categories = ("pcb1", "pcb2", "pcb3", "pcb4")
    differences = []
    for category in categories:
        group = [row for row in cases if row["category"] == category and row["label"] == 1]
        if len(group) != 100:
            raise ValueError(f"Expected exactly 100 anomaly pairs for {category}")
        differences.append(np.asarray([
            int(row["baseline_decision"] == "PASS") - int(row["challenger_decision"] == "PASS")
            for row in group
        ], dtype=np.float64))
    observed = float(np.mean([np.mean(group) for group in differences]))
    generator = np.random.default_rng(config["paired_bootstrap_seed"])
    samples = np.empty(config["paired_bootstrap_replicates"], dtype=np.float64)
    for index in range(samples.size):
        samples[index] = float(np.mean([
            np.mean(group[generator.integers(0, group.size, group.size)]) for group in differences
        ]))
    alpha = 1.0 - config["confidence_level"]
    interval = np.quantile(samples, [alpha / 2, 1 - alpha / 2], method="linear")
    return {"value": observed, "confidence_interval": [float(value) for value in interval],
            "replicates": int(samples.size), "seed": config["paired_bootstrap_seed"],
            "higher_is_better": True}


def _read_mask(archive: tarfile.TarFile, row: dict[str, Any], image_size: tuple[int, int],
               output_size: int, threshold: int, expected_sha: str | None,
               binarization: str | None = None) -> np.ndarray:
    if row["label"] == 0:
        return np.zeros((output_size, output_size), dtype=np.bool_)
    data = _member_bytes(archive, row["source_mask_member"])
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("VisA mask bytes changed after test payload audit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.size != image_size:
            raise ValueError("VisA anomaly mask format or dimensions changed")
        image.load()
        return _binary_mask(image, output_size, threshold, binarization)


def _write_text_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing external evaluation file differs: {path}")
    else:
        path.write_text(content, encoding="utf-8", newline="\n")


def human_summary(result: dict[str, Any]) -> str:
    corrected = result["comparison_status"] == "post_test_mask_parser_correction_exploratory"
    lines = [
        "# VisA PCB1–4 외부 전이 탐색 결과 (마스크 해석 수정 후)" if corrected
        else "# VisA PCB1–4 외부 전이 시험 결과", "",
        "다른 기판 사진에서 기존 점수 방식과 도전자 점수 방식을 한 번 비교했습니다.",
        "이 결과는 같은 제품의 현장 성능 또는 자동 합격 근거가 아닙니다.", "",
        "| 기판 | 기존 미탐 | 도전자 미탐 | 기존 정상 오검출 | 도전자 정상 오검출 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, metrics in result["categories"].items():
        baseline = metrics["baseline"]["fixed_threshold"]["counts"]
        challenger = metrics["challenger"]["fixed_threshold"]["counts"]
        lines.append(
            f"| {category} | {baseline['fn']}/{baseline['fn'] + baseline['tp']} | "
            f"{challenger['fn']}/{challenger['fn'] + challenger['tp']} | "
            f"{baseline['fp']}/{baseline['fp'] + baseline['tn']} | "
            f"{challenger['fp']}/{challenger['fp'] + challenger['tn']} |"
        )
    paired = result["paired_macro_escape_difference"]
    lines += [
        "",
        f"네 기판 동일 가중 미탐률 개선(기존−도전자): {paired['value'] * 100:.2f}%p; "
        f"짝지은 95% 구간 {paired['confidence_interval'][0] * 100:.2f}~"
        f"{paired['confidence_interval'][1] * 100:.2f}%p.",
        (f"기존 수치 기준 충족: {'예' if result['candidate_criteria_met'] else '아니요'}"
         " (시험 마스크를 본 뒤 파서를 수정한 탐색 결과; 원래 사전 고정 평가로 주장하지 않음).")
        if corrected else
        f"사전 기준 충족: {'예' if result['external_transfer_improvement_evidence'] else '아니요'}.",
        "근접 중복 유사도 표시는 동일 물리 기판의 증명이 아닙니다. 자동 PASS는 금지합니다.", "",
    ]
    return "\n".join(lines)


def run_evaluation(project_root: Path, archive_path: Path, *, mask_correction_v2: bool = False) -> dict[str, Any]:
    """Open the official test only after all configuration and normal thresholds are locked."""
    project_root = project_root.resolve(strict=True)
    git = git_state(project_root)
    protocol = load_config(project_root / "configs" / "visa-pcb-external-v1.toml")
    preflight_config = load_preflight_config(project_root / "configs" / "visa-pcb-external-preflight-v1.toml")
    version = "v2" if mask_correction_v2 else "v1"
    execution = load_evaluation_config(
        project_root / "configs" / f"visa-pcb-external-evaluation-{version}.toml",
        mask_correction_v2=mask_correction_v2,
    )
    plan = json.loads((project_root / "data" / "manifests" / "visa-pcb-external-plan-v1.json").read_text(encoding="utf-8"))
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_sha256", None)
    if claimed != hashlib.sha256(canonical(unsigned)).hexdigest() or plan["config_sha256"] != protocol["config_sha256"]:
        raise ValueError("VisA external plan or source protocol differs")
    print("1/4 Verifying locked preflight, 1.93 GB archive, and frozen checkpoint...", flush=True)
    preflight_zip = project_root / "artifacts" / "handoff" / "visa-pcb-external-preflight-v1.zip"
    preflight_report, calibration_audit = checked_preflight(preflight_zip, plan, execution, preflight_config)
    registration_zip = project_root / "artifacts" / "handoff" / "visa-pcb-external-registration-v1.zip"
    checked_registration(registration_zip, archive_path, plan, protocol, preflight_config)
    gate = validate_gate(project_root, "realiad_pcb", 42)
    layer = gate["dev_result"]["layer_groups"]["layer1-layer2"]
    if layer["checkpoint_sha256"] != execution["realiad_checkpoint_sha256"]:
        raise ValueError("Frozen Real-IAD PCB checkpoint changed")
    replayed_audit = audit_calibration(archive_path, plan["records"], expected_count=224)
    if replayed_audit != calibration_audit:
        raise ValueError("VisA calibration audit differs from the unopened-test preflight")
    calibration_records = [row for row in plan["records"] if row["protocol_split"] == "calibration"]
    final_records = [row for row in plan["records"] if row["protocol_split"] == "external_final"]
    if (
        len(final_records) != 802
        or sum(row["label"] == 0 for row in final_records) != execution["external_final_normal_total"]
        or sum(row["label"] == 1 for row in final_records) != execution["external_final_anomaly_total"]
        or any(sum(row["category"] == category for row in calibration_records) != 56
               for category in ("pcb1", "pcb2", "pcb3", "pcb4"))
    ):
        raise ValueError("VisA category counts differ before official test opening")
    output = project_root / "artifacts" / f"visa-pcb-external-evaluation-{version}"
    handoff = project_root / "artifacts" / "handoff" / f"visa-pcb-external-evaluation-{version}.zip"
    if handoff.exists():
        raise FileExistsError(f"Completed external evaluation exists and was not overwritten: {handoff}")
    output.mkdir(parents=True, exist_ok=True)

    # Model setup and both normal-only threshold calculations happen before test payload audit.
    print("2/4 Calculating thresholds from 224 normal calibration images; test remains sealed...", flush=True)
    gpu = _gpu_modules()
    _seed_everything(42, gpu[0])
    model = _model(gate["protocol"], gate["dev_checkpoint"], {
        "category": "realiad_pcb", "seed": 42,
        "manifest_sha256": gate["development"]["manifest_sha256"],
        "execution_config_sha256": gate["execution"]["config_sha256"],
    }, gpu)
    audit_by_name = {row["image"]: row for row in calibration_audit["records"]}
    calibration_rows = []
    calibration_maps: dict[str, list[np.ndarray]] = {category: [] for category in plan["categories"]}
    with tarfile.open(archive_path, "r:") as archive:
        for index, record in enumerate(calibration_records, start=1):
            name = record["source_image_member"]
            data = _member_bytes(archive, name)
            if hashlib.sha256(data).hexdigest() != audit_by_name[name]["image_sha256"]:
                raise ValueError(f"Calibration JPEG bytes changed: {name}")
            with _jpeg(data, name) as image:
                raw, anomaly_map = _predict(model, image, gate["protocol"], gpu)
            if anomaly_map.shape != (256, 256) or not np.isfinite(anomaly_map).all():
                raise ValueError(f"Invalid calibration anomaly map: {name}")
            calibration_maps[record["category"]].append(anomaly_map)
            calibration_rows.append({
                "category": record["category"], "image": name,
                "baseline_score": candidate_image_score(raw, anomaly_map, {
                    "image_score": "mean_of_top_map_fraction",
                    "top_fraction": execution["baseline_map_top_fraction"],
                }),
                "challenger_score": candidate_image_score(raw, anomaly_map, {
                    "image_score": "mean_of_top_map_fraction",
                    "top_fraction": execution["challenger_map_top_fraction"],
                }),
            })
            if index % 25 == 0 or index == len(calibration_records):
                print(f"normal calibration {index}/{len(calibration_records)}", flush=True)
    thresholds = {}
    for category in plan["categories"]:
        rows = [row for row in calibration_rows if row["category"] == category]
        thresholds[category] = {
            "baseline": higher_quantile([row["baseline_score"] for row in rows], execution["image_quantile"]),
            "challenger": higher_quantile([row["challenger_score"] for row in rows], execution["image_quantile"]),
            "pixel": float(np.quantile(
                np.stack(calibration_maps[category]).reshape(-1),
                execution["pixel_calibration_quantile"], method="higher",
            )),
        }
        if any(not math.isfinite(value) for value in thresholds[category].values()):
            raise ValueError("Non-finite normal-only VisA threshold")
    calibration_evidence = {
        "status": ("normal_only_thresholds_recomputed_after_test_mask_parser_correction"
                   if mask_correction_v2 else "normal_only_thresholds_locked_before_external_test"),
        "plan_sha256": plan["plan_sha256"],
        "execution_config_sha256": execution["config_sha256"],
        "checkpoint_sha256": layer["checkpoint_sha256"],
        "execution_git_commit": git["commit"],
        "count": len(calibration_rows), "thresholds": thresholds, "rows": calibration_rows,
        "external_final_images_decoded": 0,
    }
    if mask_correction_v2:
        calibration_evidence["prior_v1_test_mask_observed"] = True
    _write_json_once(output / "calibration.json", calibration_evidence)

    # If any test file is unreadable, duplicated exactly, or has an empty mask, stop without metrics.
    print("3/4 Opening all 802 official test images and 400 masks for data integrity audit...", flush=True)
    test_audit = audit_final_payloads(archive_path, final_records, calibration_audit, execution)
    test_audit["execution_git_commit"] = git["commit"]
    test_audit["evaluation_config_sha256"] = execution["config_sha256"]
    _write_json_once(output / "test-audit.json", test_audit)
    test_by_name = {row["image"]: row for row in test_audit["records"]}
    cases = []
    maps_by_category: dict[str, list[np.ndarray]] = {category: [] for category in plan["categories"]}
    masks_by_category: dict[str, list[np.ndarray]] = {category: [] for category in plan["categories"]}
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    print("4/4 Scoring the same 802 test images with both locked score rules...", flush=True)
    with tarfile.open(archive_path, "r:") as archive:
        for index, record in enumerate(final_records, start=1):
            name = record["source_image_member"]
            audited = test_by_name[name]
            expected = {
                "category": record["category"], "label": record["label"], "image": name,
                "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
                "plan_sha256": plan["plan_sha256"], "execution_config_sha256": execution["config_sha256"],
                "checkpoint_sha256": layer["checkpoint_sha256"], "source_archive_sha256":
                preflight_config["source_archive_sha256"], "execution_git_commit": git["commit"],
            }
            sample_path = samples / f"sample-{index:04d}.npz"
            if sample_path.exists():
                metadata, anomaly_map, mask = _read_sample(sample_path, expected, 256)
                baseline_score = candidate_image_score(metadata["raw_score"], anomaly_map, {
                    "image_score": "mean_of_top_map_fraction", "top_fraction": execution["baseline_map_top_fraction"],
                })
                challenger_score = candidate_image_score(metadata["raw_score"], anomaly_map, {
                    "image_score": "mean_of_top_map_fraction", "top_fraction": execution["challenger_map_top_fraction"],
                })
                if (abs(baseline_score - metadata["score"]) > 1e-5
                        or abs(challenger_score - metadata["challenger_score"]) > 1e-5):
                    raise ValueError(f"Cached VisA image score differs: {sample_path}")
            else:
                data = _member_bytes(archive, name)
                if hashlib.sha256(data).hexdigest() != audited["image_sha256"]:
                    raise ValueError(f"VisA image changed after test payload audit: {name}")
                with _jpeg(data, name) as image:
                    mask = _read_mask(
                        archive, record, image.size, 256, execution["pixel_ground_truth_threshold"],
                        audited["mask_sha256"], execution.get("mask_binarization"),
                    )
                    raw, anomaly_map = _predict(model, image, gate["protocol"], gpu)
                baseline_score = candidate_image_score(raw, anomaly_map, {
                    "image_score": "mean_of_top_map_fraction", "top_fraction": execution["baseline_map_top_fraction"],
                })
                challenger_score = candidate_image_score(raw, anomaly_map, {
                    "image_score": "mean_of_top_map_fraction", "top_fraction": execution["challenger_map_top_fraction"],
                })
                metadata = {**expected, "raw_score": raw, "score": baseline_score,
                            "challenger_score": challenger_score}
                _write_sample(sample_path, metadata, anomaly_map, mask)
            category = record["category"]
            maps_by_category[category].append(anomaly_map)
            masks_by_category[category].append(mask)
            cases.append({
                "category": category, "image": name, "label": record["label"],
                "baseline_score": baseline_score, "challenger_score": challenger_score,
                "baseline_threshold": thresholds[category]["baseline"],
                "challenger_threshold": thresholds[category]["challenger"],
                "baseline_decision": "REVIEW_REQUIRED" if baseline_score >= thresholds[category]["baseline"] else "PASS",
                "challenger_decision": "REVIEW_REQUIRED" if challenger_score >= thresholds[category]["challenger"] else "PASS",
                "sample_sha256": file_sha256(sample_path),
            })
            if index % 25 == 0 or index == len(final_records):
                print(f"external test scoring {index}/{len(final_records)}", flush=True)
    category_results = {}
    for category in plan["categories"]:
        category_cases = [row for row in cases if row["category"] == category]
        metric_records = [{"label": row["label"], "defect_type": "anomaly" if row["label"] else "good"}
                          for row in category_cases]
        category_results[category] = {}
        for candidate in ("baseline", "challenger"):
            score_key = f"{candidate}_score"
            decision_key = f"{candidate}_decision"
            rows = [{"score": row[score_key], "decision": row[decision_key]} for row in category_cases]
            category_results[category][candidate] = _candidate_metrics(
                metric_records, rows, thresholds[category][candidate], execution["confidence_level"]
            )
        category_results[category]["pixel"] = _pixel_metrics(
            np.stack(maps_by_category[category]), np.stack(masks_by_category[category]),
            thresholds[category]["pixel"],
            {"aupro_fpr_limit": execution["pixel_aupro_fpr_limit"],
             "aupro_fpr_points": execution["pixel_aupro_fpr_points"]},
        )
    paired = paired_macro_escape_difference(cases, execution)
    macro = {}
    for candidate in ("baseline", "challenger"):
        macro[candidate] = {
            "image_auroc": float(np.mean([
                category_results[category][candidate]["image_auroc"] for category in plan["categories"]
            ])),
            "image_average_precision": float(np.mean([
                category_results[category][candidate]["image_average_precision"]
                for category in plan["categories"]
            ])),
            "escape_rate": float(np.mean([
                category_results[category][candidate]["fixed_threshold"]["rates"]["escape_rate"]
                for category in plan["categories"]
            ])),
            "overkill_rate": float(np.mean([
                category_results[category][candidate]["fixed_threshold"]["rates"]["overkill_rate"]
                for category in plan["categories"]
            ])),
        }
    category_overkill_ok = all(
        category_results[category]["challenger"]["fixed_threshold"]["rates"]["overkill_rate"]
        <= execution["maximum_challenger_overkill_per_category"] for category in plan["categories"]
    )
    category_escape_ok = all(
        category_results[category]["challenger"]["fixed_threshold"]["rates"]["escape_rate"]
        - category_results[category]["baseline"]["fixed_threshold"]["rates"]["escape_rate"]
        <= execution["maximum_category_escape_worsening"] for category in plan["categories"]
    )
    improvement = paired["confidence_interval"][0] > 0 and category_overkill_ok and category_escape_ok
    result = {
        "schema_version": 2 if mask_correction_v2 else 1,
        "status": "evaluated_exploratory_after_mask_parser_correction" if mask_correction_v2
        else "evaluated_preregistered_cross_dataset_transfer",
        "comparison_status": "post_test_mask_parser_correction_exploratory" if mask_correction_v2
        else "prospective_visa_transfer_after_realiad_final_observed_not_same_product",
        "git": git, "plan_sha256": plan["plan_sha256"],
        "evaluation_config_sha256": execution["config_sha256"],
        "preflight_handoff_sha256": execution["preflight_handoff_sha256"],
        "source_archive_sha256": preflight_config["source_archive_sha256"],
        "checkpoint_sha256": layer["checkpoint_sha256"],
        "preflight_status": preflight_report["status"],
        "normal_calibration_count": len(calibration_rows),
        "official_test_count": len(cases), "test_payload_audit_status": test_audit["status"],
        "test_payload_audit_sha256": hashlib.sha256(canonical(test_audit)).hexdigest(),
        "thresholds": thresholds, "categories": category_results, "macro": macro,
        "paired_macro_escape_difference": paired,
        "improvement_criteria": {
            "paired_ci_lower_bound_positive": paired["confidence_interval"][0] > 0,
            "challenger_overkill_each_category_at_most_10pct": category_overkill_ok,
            "no_category_escape_worsens_more_than_5pp": category_escape_ok,
        },
        "external_transfer_improvement_evidence": improvement if not mask_correction_v2 else False,
        "candidate_criteria_met": improvement,
        "mask_binarization": execution.get("mask_binarization", "grayscale_then_threshold"),
        "protocol_deviation": execution.get("protocol_deviation"),
        "model_training_on_visa": False,
        "failed_images_excluded": 0,
        "automatic_pass_allowed": False,
        "limitations": [
            "VisA pcb1–4 are different boards from Real-IAD pcb; this is cross-dataset transfer, not same-product confirmation.",
            "Category-specific normal-only threshold adaptation was used; this is not full zero-shot evaluation.",
            "Perceptual-hash near-pair flags do not prove same physical item identity.",
            "A positive external-transfer result does not authorize automatic PASS or deployment.",
            *(["The original test audit failed on the first VisA class-ID mask; this parser correction "
               "was made after that mask was observed, so this run is exploratory, not the untouched "
               "preregistered test."] if mask_correction_v2 else []),
        ],
    }
    cases_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cases)
    _write_text_once(output / "cases.jsonl", cases_text)
    result["cases_sha256"] = file_sha256(output / "cases.jsonl")
    _write_json_once(output / "evaluation.json", result)
    _write_text_once(output / "summary.md", human_summary(result))
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("calibration.json", "test-audit.json", "evaluation.json", "cases.jsonl", "summary.md"):
            archive.write(output / name, name)
    return {"status": result["status"], "external_transfer_improvement_evidence": result["external_transfer_improvement_evidence"],
            "candidate_criteria_met": improvement,
            "paired_macro_escape_difference": paired, "handoff_zip": str(handoff),
            "handoff_zip_sha256": file_sha256(handoff)}
