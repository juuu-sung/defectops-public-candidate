"""Post-test exploratory PCB3 comparison using the completed VisA normal bank v2."""

from __future__ import annotations

import hashlib
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
from defectops.realiad_followup_development import _candidate_metrics, _pixel_metrics, candidate_image_score
from defectops.realiad_followup_final import _gpu_modules, _model, _predict, _read_sample, _write_sample
from defectops.visa_pcb_external_evaluation import (
    _jpeg, _member_bytes, _pixel_sha, _read_mask, _write_text_once, load_evaluation_config,
)
from defectops.visa_pcb_external_protocol import canonical
from defectops.visa_pcb_external_registration import file_sha256
from defectops.visa_pcb_normal_bank_development import (
    _validated_existing, checked_sources, load_execution,
)


EXPECTED = {
    "schema_version": 1,
    "status": "post_test_exploratory_pcb3_normal_bank_comparison",
    "protocol": "A-normal-only-v2",
    "category": "pcb3", "model_seed": 42,
    "fit_handoff_sha256": "93d998fd88a3983c35688aaee850de794eedabbcf2b1d79471eaa50e3c195b07",
    "fit_config_sha256": "343e2d910e582146aa44e19fc663fe13f01f493f5321e69595d999dc8da79e8e",
    "fit_checkpoint_sha256": "be8062541e2a14cb563284d34cfc9f60e3ca5e8cc1d75e0ccbd28d1287785e85",
    "baseline_evaluation_handoff_sha256": "ba2f5200757e212f97f0a431679a429aa068ee9140b0a28a760641e9427a0d2a",
    "source_archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
    "source_plan_sha256": "e134445218bfb71c3cbcfcad9da45548024ef6166a2acd8b0ab64ff23e5ba34a",
    "normal_test_count": 101, "anomaly_test_count": 100,
    "image_score": "mean_of_top_map_fraction", "top_fraction": 0.01,
    "decision_rule": "score >= normal_calibration_threshold",
    "mask_binarization": "nonzero_before_nearest", "mask_threshold": 128,
    "confidence_level": 0.95, "paired_bootstrap_replicates": 5000,
    "paired_bootstrap_seed": 20260925,
    "pixel_aupro_fpr_limit": 0.30, "pixel_aupro_fpr_points": 301,
    "post_test_exploratory": True, "automatic_pass_allowed": False,
    "confirmatory_claim_allowed": False,
}


def load_comparison(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config != EXPECTED:
        raise ValueError("PCB3 post-test comparison config differs from the locked values")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def _require_zip(path: Path, expected_names: list[str], expected_sha: str) -> dict[str, bytes]:
    if file_sha256(path) != expected_sha:
        raise ValueError(f"Handoff ZIP checksum differs: {path}")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(expected_names) or archive.testzip() is not None:
            raise ValueError(f"Handoff ZIP members or CRC differ: {path}")
        return {name: archive.read(name) for name in expected_names}


def _same_metrics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only compare stable point estimates, not last-bit CI rounding across NumPy builds."""
    return (
        left["fixed_threshold"]["counts"] == right["fixed_threshold"]["counts"]
        and all(math.isclose(left["fixed_threshold"]["rates"][key], right["fixed_threshold"]["rates"][key],
                             abs_tol=1e-12) for key in left["fixed_threshold"]["rates"])
        and all(math.isclose(left[key], right[key], abs_tol=1e-12)
                for key in ("image_auroc", "image_average_precision"))
    )


def checked_inputs(project: Path, archive_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bind new fit and old 802-image evidence before touching any test JPEG."""
    fit_execution = load_execution(project / "configs" / "visa-pcb-normal-bank-development-v2.toml", 2)
    previous_execution = load_evaluation_config(
        project / "configs" / "visa-pcb-external-evaluation-v2.toml", mask_correction_v2=True,
    )
    if fit_execution["config_sha256"] != config["fit_config_sha256"]:
        raise ValueError("Normal-bank v2 fit config hash differs")
    if fit_execution["source_evaluation_handoff_sha256"] != config["baseline_evaluation_handoff_sha256"]:
        raise ValueError("Normal-bank fit and comparison point to different prior evaluations")
    if fit_execution["source_archive_sha256"] != config["source_archive_sha256"]:
        raise ValueError("Normal-bank fit and comparison point to different VisA archives")
    plan, train_paths, calibration, _ = checked_sources(project, archive_path, "pcb3", fit_execution)
    if plan["plan_sha256"] != config["source_plan_sha256"]:
        raise ValueError("Original VisA plan differs from the comparison lock")

    fit_dir = project / "artifacts" / "visa-pcb-normal-bank-development-v2" / "pcb3"
    fit = _validated_existing(fit_dir, "pcb3", plan, fit_execution)
    fit_zip = _require_zip(
        project / "artifacts" / "handoff" / "visa-pcb-normal-bank-pcb3-v2.zip",
        ["fit.json", "training-audit.jsonl", "calibration.jsonl"], config["fit_handoff_sha256"],
    )
    if any(fit_zip[name] != (fit_dir / name).read_bytes() for name in fit_zip):
        raise ValueError("Local completed fit differs from the uploaded fit handoff")
    if json.loads(fit_zip["fit.json"]) != fit or file_sha256(fit_dir / "model.pt") != config["fit_checkpoint_sha256"]:
        raise ValueError("Completed PCB3 v2 fit report/checkpoint differs")
    if (
        fit["checkpoint_sha256"] != config["fit_checkpoint_sha256"]
        or fit["normal_train_count"] != 849 or len(train_paths) != 849
        or fit["normal_calibration_count"] != 56 or len(calibration) != 56
        or fit["embedding_count_before_coreset"] != 869376
        or fit["memory_bank_shape"] != [86937, 768]
        or fit["reload_max_absolute_score_difference"] > 1e-4
        or fit["official_test_scores_computed"] != 0
        or fit["automatic_pass_allowed"] is not False
    ):
        raise ValueError("PCB3 v2 fit does not match the completed normal-only pilot")
    training = _jsonl(fit_zip["training-audit.jsonl"])
    if [row["image"] for row in training] != train_paths or len({row["decoded_pixel_sha256"] for row in training}) != 849:
        raise ValueError("Completed fit used different normal training photos")
    fit_calibration = _jsonl(fit_zip["calibration.jsonl"])
    if (
        [row["image"] for row in fit_calibration] != [row["source_image_member"] for row in calibration]
        or higher_quantile([row["score"] for row in fit_calibration], fit_execution["image_quantile"])
        != fit["image_threshold"]
    ):
        raise ValueError("Completed fit calibration differs from the fixed 56 normals")

    old_zip = _require_zip(
        project / "artifacts" / "handoff" / "visa-pcb-external-evaluation-v2.zip",
        ["calibration.json", "test-audit.json", "evaluation.json", "cases.jsonl", "summary.md"],
        config["baseline_evaluation_handoff_sha256"],
    )
    prior = json.loads(old_zip["evaluation.json"])
    test_audit = json.loads(old_zip["test-audit.json"])
    old_cases = _jsonl(old_zip["cases.jsonl"])
    if (
        prior.get("status") != "evaluated_exploratory_after_mask_parser_correction"
        or prior.get("comparison_status") != "post_test_mask_parser_correction_exploratory"
        or prior.get("plan_sha256") != plan["plan_sha256"]
        or prior.get("evaluation_config_sha256") != previous_execution["config_sha256"]
        or prior.get("source_archive_sha256") != config["source_archive_sha256"]
        or prior.get("mask_binarization") != config["mask_binarization"]
        or prior.get("model_training_on_visa") is not False
        or prior.get("automatic_pass_allowed") is not False
        or prior.get("official_test_count") != 802
        or prior.get("cases_sha256") != hashlib.sha256(old_zip["cases.jsonl"]).hexdigest()
        or prior.get("test_payload_audit_sha256") != hashlib.sha256(canonical(test_audit)).hexdigest()
        or len(old_cases) != 802 or test_audit.get("count") != 802
    ):
        raise ValueError("Prior VisA external evaluation evidence differs")
    official_test = [row for row in plan["records"] if row["protocol_split"] == "external_final"]
    if (
        {row["source_image_member"] for row in official_test} != {row["image"] for row in old_cases}
        or {row["source_image_member"] for row in official_test}
        != {row["image"] for row in test_audit["records"]}
    ):
        raise ValueError("Prior cases/audit do not cover the original 802 official test images")
    old_by_name = {row["image"]: row for row in old_cases}
    audit_by_name = {row["image"]: row for row in test_audit["records"]}
    pcb3 = [row for row in official_test if row["category"] == "pcb3"]
    if (
        len(pcb3) != config["normal_test_count"] + config["anomaly_test_count"]
        or sum(row["label"] == 0 for row in pcb3) != config["normal_test_count"]
        or sum(row["label"] == 1 for row in pcb3) != config["anomaly_test_count"]
        or any(old_by_name[row["source_image_member"]]["label"] != row["label"]
               or old_by_name[row["source_image_member"]]["category"] != "pcb3"
               or audit_by_name[row["source_image_member"]]["label"] != row["label"]
               or audit_by_name[row["source_image_member"]]["category"] != "pcb3" for row in pcb3)
    ):
        raise ValueError("Prior PCB3 test pairing or counts differ")
    baseline_threshold = prior["thresholds"]["pcb3"]["baseline"]
    baseline_rows = [old_by_name[row["source_image_member"]] for row in pcb3]
    if any(
        row["baseline_threshold"] != baseline_threshold
        or row["baseline_decision"] != (
            "REVIEW_REQUIRED" if row["baseline_score"] >= baseline_threshold else "PASS"
        ) for row in baseline_rows
    ):
        raise ValueError("Prior PCB3 threshold or per-image decisions differ")
    records = [{"label": row["label"], "defect_type": "anomaly" if row["label"] else "good"} for row in pcb3]
    recomputed = _candidate_metrics(records, [
        {"score": row["baseline_score"], "decision": row["baseline_decision"]} for row in baseline_rows
    ], baseline_threshold, config["confidence_level"])
    if not _same_metrics(recomputed, prior["categories"]["pcb3"]["baseline"]):
        raise ValueError("Prior PCB3 aggregate does not match its per-image cases")
    return {
        "plan": plan, "fit": fit, "fit_dir": fit_dir, "test_records": pcb3,
        "baseline_cases": old_by_name, "baseline_metrics": recomputed,
        "baseline_threshold": baseline_threshold, "audit": audit_by_name,
    }


def paired_summary(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Describe paired decision changes on exactly the same 100 defects and 101 normals."""
    anomalies = [row for row in cases if row["label"] == 1]
    normals = [row for row in cases if row["label"] == 0]
    if len(anomalies) != config["anomaly_test_count"] or len(normals) != config["normal_test_count"]:
        raise ValueError("Paired PCB3 test denominator differs")
    old_missed = np.asarray([row["baseline_decision"] == "PASS" for row in anomalies], dtype=np.int8)
    new_missed = np.asarray([row["v2_decision"] == "PASS" for row in anomalies], dtype=np.int8)
    differences = old_missed - new_missed
    generator = np.random.default_rng(config["paired_bootstrap_seed"])
    replicates = np.empty(config["paired_bootstrap_replicates"], dtype=np.float64)
    for index in range(replicates.size):
        replicates[index] = float(np.mean(differences[generator.integers(0, len(differences), len(differences))]))
    interval = np.quantile(replicates, [0.025, 0.975], method="linear")
    return {
        "escape_rate_improvement_baseline_minus_v2": float(np.mean(differences)),
        "descriptive_paired_bootstrap_95pct_interval": [float(value) for value in interval],
        "bootstrap_replicates": config["paired_bootstrap_replicates"],
        "bootstrap_seed": config["paired_bootstrap_seed"],
        "anomaly_old_missed_new_caught": int(np.count_nonzero((old_missed == 1) & (new_missed == 0))),
        "anomaly_old_caught_new_missed": int(np.count_nonzero((old_missed == 0) & (new_missed == 1))),
        "anomaly_both_missed": int(np.count_nonzero((old_missed == 1) & (new_missed == 1))),
        "anomaly_both_caught": int(np.count_nonzero((old_missed == 0) & (new_missed == 0))),
        "normal_old_pass_new_review": sum(
            row["baseline_decision"] == "PASS" and row["v2_decision"] == "REVIEW_REQUIRED" for row in normals
        ),
        "normal_old_review_new_pass": sum(
            row["baseline_decision"] == "REVIEW_REQUIRED" and row["v2_decision"] == "PASS" for row in normals
        ),
        "interpretation": "post_test_descriptive_only_not_confirmatory",
    }


def human_summary(report: dict[str, Any]) -> str:
    old = report["baseline"]["fixed_threshold"]["counts"]
    new = report["visa_normal_bank_v2"]["fixed_threshold"]["counts"]
    pair = report["paired"]
    return "\n".join([
        "# VisA PCB3 정상 기준표 v2 사후 탐색 비교", "",
        "이미 관찰한 같은 PCB3 시험 사진 201장(불량 100장·정상 101장)을 비교했습니다.",
        "원래 기준표는 Real-IAD PCB 정상 사진, v2는 VisA PCB3 정상 사진을 사용합니다.",
        "v2는 학습 사진 수와 패치 후보 밀도도 달라 차이를 정상 데이터 출처만의 효과로 해석할 수 없습니다.", "",
        "| 지표 | 기존 기준표 | VisA PCB3 v2 |", "|---|---:|---:|",
        f"| 미탐 | {old['fn']}/100 | {new['fn']}/100 |",
        f"| 정상 오검출 | {old['fp']}/101 | {new['fp']}/101 |",
        f"| Image AUROC | {report['baseline']['image_auroc']:.4f} | {report['visa_normal_bank_v2']['image_auroc']:.4f} |",
        f"| Pixel AUPRO | {report['baseline_pixel']['aupro']['value']:.4f} | "
        f"{report['visa_normal_bank_v2_pixel']['aupro']['value']:.4f} |", "",
        f"기존에 놓쳤지만 v2가 잡은 불량 {pair['anomaly_old_missed_new_caught']}장; "
        f"기존에 잡았지만 v2가 놓친 불량 {pair['anomaly_old_caught_new_missed']}장.",
        "이미 본 test의 반복 사용이므로 이 수치는 탐색적입니다. 자동 PASS나 생산 적용은 금지합니다.", "",
    ])


def run(project: Path, archive_path: Path) -> dict[str, Any]:
    project = project.resolve(strict=True)
    archive_path = archive_path.resolve(strict=True)
    git = git_state(project)
    config = load_comparison(project / "configs" / "visa-pcb-normal-bank-pcb3-evaluation-v1.toml")
    print("1/3 Checking the frozen fit, previous test evidence, source archive and exact PCB3 pairing...", flush=True)
    inputs = checked_inputs(project, archive_path, config)
    fit = inputs["fit"]
    output = project / "artifacts" / "visa-pcb-normal-bank-pcb3-evaluation-v1"
    handoff = project / "artifacts" / "handoff" / "visa-pcb-normal-bank-pcb3-evaluation-v1.zip"
    names = ["evaluation.json", "cases.jsonl", "summary.md"]
    if handoff.exists():
        with zipfile.ZipFile(handoff) as previous:
            if sorted(previous.namelist()) != sorted(names) or previous.testzip() is not None:
                raise ValueError("Existing PCB3 evaluation handoff is invalid; it was not changed")
            if any(previous.read(name) != (output / name).read_bytes() for name in names):
                raise ValueError("Existing PCB3 evaluation handoff differs from local result")
        report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        if (
            report.get("status") != "evaluated_post_test_exploratory_pcb3_normal_bank"
            or report.get("comparison_config_sha256") != config["config_sha256"]
            or report.get("fit_checkpoint_sha256") != config["fit_checkpoint_sha256"]
            or report.get("baseline_evaluation_handoff_sha256") != config["baseline_evaluation_handoff_sha256"]
            or report.get("test_count") != 201
            or report.get("cases_sha256") != file_sha256(output / "cases.jsonl")
            or report.get("automatic_pass_allowed") is not False
            or report.get("confirmatory_claim_allowed") is not False
        ):
            raise ValueError("Existing PCB3 evaluation result differs from the locked comparison")
        return {"status": report["status"], "handoff_zip": str(handoff),
                "handoff_sha256": file_sha256(handoff), "reused_completed_result": True}
    if (
        fit["image_score"] != config["image_score"]
        or fit["top_fraction"] != config["top_fraction"]
        or fit["model_seed"] != config["model_seed"]
    ):
        raise ValueError("Scoring rule or seed changed since PCB3 normal-only calibration")
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    gpu = _gpu_modules()
    protocol = {"base_model": {
        "backbone": fit["backbone"], "image_size": fit["image_size"],
        "coreset_sampling_ratio": fit["coreset_sampling_ratio"], "num_neighbors": fit["num_neighbors"],
    }}
    model = _model(protocol, inputs["fit_dir"] / "model.pt", {
        "category": "pcb3", "seed": config["model_seed"],
        "manifest_sha256": inputs["plan"]["plan_sha256"],
        "execution_config_sha256": config["fit_config_sha256"],
    }, gpu)
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    print("2/3 Scoring 201 already-observed PCB3 test photos with the new normal bank...", flush=True)
    with tarfile.open(archive_path, "r:") as archive:
        for index, record in enumerate(inputs["test_records"], start=1):
            name = record["source_image_member"]
            audited = inputs["audit"][name]
            old = inputs["baseline_cases"][name]
            expected = {
                "category": "pcb3", "label": record["label"], "image": name,
                "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
                "plan_sha256": config["source_plan_sha256"],
                "comparison_config_sha256": config["config_sha256"],
                "fit_checkpoint_sha256": config["fit_checkpoint_sha256"],
                "fit_handoff_sha256": config["fit_handoff_sha256"],
                "baseline_evaluation_handoff_sha256": config["baseline_evaluation_handoff_sha256"],
                "source_archive_sha256": config["source_archive_sha256"],
                "image_threshold": fit["image_threshold"],
            }
            sample_path = samples / f"sample-{index:04d}.npz"
            if sample_path.exists():
                metadata, anomaly_map, mask = _read_sample(sample_path, expected, fit["image_size"])
                score = candidate_image_score(metadata["raw_score"], anomaly_map, config)
                if abs(score - metadata["score"]) > 1e-5:
                    raise ValueError(f"Cached PCB3 score differs: {sample_path}")
            else:
                data = _member_bytes(archive, name)
                if hashlib.sha256(data).hexdigest() != audited["image_sha256"]:
                    raise ValueError(f"PCB3 test JPEG bytes differ from previous audit: {name}")
                image = _jpeg(data, name)
                if _pixel_sha(image) != audited["decoded_pixel_sha256"] or image.size != (
                    audited["width"], audited["height"]
                ):
                    raise ValueError(f"PCB3 decoded test pixels/dimensions differ: {name}")
                mask = _read_mask(
                    archive, record, image.size, fit["image_size"], config["mask_threshold"],
                    audited["mask_sha256"], config["mask_binarization"],
                )
                if int(np.count_nonzero(mask)) != audited["mask_nonzero_256"]:
                    raise ValueError(f"PCB3 corrected test mask differs from previous audit: {name}")
                raw_score, anomaly_map = _predict(model, image, protocol, gpu)
                if anomaly_map.shape != (fit["image_size"], fit["image_size"]):
                    raise ValueError(f"PCB3 anomaly map shape differs: {name}")
                score = candidate_image_score(raw_score, anomaly_map, config)
                metadata = {**expected, "raw_score": raw_score, "score": score}
                _write_sample(sample_path, metadata, anomaly_map, mask)
            decision = "REVIEW_REQUIRED" if score >= fit["image_threshold"] else "PASS"
            cases.append({
                "category": "pcb3", "image": name, "label": record["label"],
                "baseline_score": old["baseline_score"],
                "baseline_threshold": inputs["baseline_threshold"],
                "baseline_decision": old["baseline_decision"],
                "v2_score": score, "v2_threshold": fit["image_threshold"],
                "v2_decision": decision, "sample_sha256": file_sha256(sample_path),
            })
            maps.append(anomaly_map)
            masks.append(mask)
            if index % 25 == 0 or index == len(inputs["test_records"]):
                print(f"PCB3 exploratory test {index}/{len(inputs['test_records'])}", flush=True)
    print("3/3 Calculating paired image and pixel diagnostics without changing thresholds...", flush=True)
    records = [{"label": row["label"], "defect_type": "anomaly" if row["label"] else "good"}
               for row in cases]
    v2_metrics = _candidate_metrics(records, [
        {"score": row["v2_score"], "decision": row["v2_decision"]} for row in cases
    ], fit["image_threshold"], config["confidence_level"])
    v2_pixel = _pixel_metrics(np.stack(maps), np.stack(masks), fit["pixel_threshold"], {
        "aupro_fpr_limit": config["pixel_aupro_fpr_limit"],
        "aupro_fpr_points": config["pixel_aupro_fpr_points"],
    })
    prior = _require_zip(
        project / "artifacts" / "handoff" / "visa-pcb-external-evaluation-v2.zip",
        ["calibration.json", "test-audit.json", "evaluation.json", "cases.jsonl", "summary.md"],
        config["baseline_evaluation_handoff_sha256"],
    )
    previous_pixel = json.loads(prior["evaluation.json"])["categories"]["pcb3"]["pixel"]
    paired = paired_summary(cases, config)
    cases_text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in cases)
    _write_text_once(output / "cases.jsonl", cases_text)
    report = {
        "schema_version": 1, "status": "evaluated_post_test_exploratory_pcb3_normal_bank",
        "comparison_status": "already_observed_visa_test_reuse_no_confirmatory_claim",
        "git": git, "comparison_config_sha256": config["config_sha256"],
        "source_plan_sha256": config["source_plan_sha256"],
        "source_archive_sha256": config["source_archive_sha256"],
        "baseline_evaluation_handoff_sha256": config["baseline_evaluation_handoff_sha256"],
        "fit_handoff_sha256": config["fit_handoff_sha256"],
        "fit_checkpoint_sha256": config["fit_checkpoint_sha256"],
        "test_count": len(cases), "test_normal_count": config["normal_test_count"],
        "test_anomaly_count": config["anomaly_test_count"],
        "normal_train_count": fit["normal_train_count"],
        "normal_calibration_count": fit["normal_calibration_count"],
        "image_score_rule": config["image_score"], "top_fraction": config["top_fraction"],
        "mask_binarization": config["mask_binarization"],
        "baseline_pixel_source": "locked_prior_evaluation_handoff_not_recomputed",
        "baseline": inputs["baseline_metrics"], "baseline_pixel": previous_pixel,
        "visa_normal_bank_v2": v2_metrics, "visa_normal_bank_v2_pixel": v2_pixel,
        "paired": paired, "scoring_and_metrics_seconds": time.perf_counter() - started,
        "cases_sha256": file_sha256(output / "cases.jsonl"),
        "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
        "limitations": [
            "The same official VisA PCB3 test was observed in prior exploratory analyses; this is not an untouched holdout.",
            "Memory-bank source, normal training image count, and spatial candidate density all changed; their effects cannot be isolated.",
            "Raw score and threshold magnitudes across two different memory banks are not directly comparable.",
            "No automatic PASS, production decision, or confirmatory model selection is authorized.",
        ],
    }
    _write_text_once(output / "evaluation.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _write_text_once(output / "summary.md", human_summary(report))
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{handoff.stem}-", suffix=".zip", dir=handoff.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as result_zip:
            for name in names:
                result_zip.write(output / name, name)
        os.replace(temporary, handoff)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": report["status"], "baseline": report["baseline"]["fixed_threshold"]["counts"],
            "visa_normal_bank_v2": v2_metrics["fixed_threshold"]["counts"],
            "paired": paired, "handoff_zip": str(handoff), "handoff_sha256": file_sha256(handoff)}
