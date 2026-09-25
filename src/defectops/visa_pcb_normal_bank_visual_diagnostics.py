"""Read-only visual review of the 15 PCB3 errors in the completed v2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from defectops.padim_inspection import colorize_anomaly_values
from defectops.realiad_followup_development import candidate_image_score
from defectops.realiad_followup_final import _read_sample
from defectops.visa_pcb_external_evaluation import _jpeg, _member_bytes, _pixel_sha, _read_mask
from defectops.visa_pcb_external_registration import file_sha256
from defectops.visa_pcb_external_visual_diagnostics import _crop_native, _outline, load_evidence
from defectops.visa_pcb_normal_bank_evaluation import _require_zip, load_comparison


EVALUATION_SHA256 = "3453af0eb418c023d762e63d1b2344b2f175938c2a60ae7e32e40a6ae210ea6f"
EXPECTED_EVALUATION_COMMIT = "bfd29ae555392684018230228f5bb6e310421389"
MEMBERS = ["evaluation.json", "cases.jsonl", "summary.md"]
FIT_MEMBERS = ["fit.json", "training-audit.jsonl", "calibration.jsonl"]


def load_paired_evidence(
    project: Path, old_handoff: Path | None = None, evaluation_handoff: Path | None = None,
    fit_handoff: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]],
           list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bind both archived evaluations, the frozen fit, and the original test audit."""
    plan_path = project / "data/manifests/visa-pcb-external-plan-v1.json"
    old_config_path = project / "configs/visa-pcb-external-evaluation-v2.toml"
    prior, prior_cases, audit = load_evidence(
        old_handoff or project / "artifacts/handoff/visa-pcb-external-evaluation-v2.zip",
        plan_path, old_config_path,
    )
    config = load_comparison(project / "configs/visa-pcb-normal-bank-pcb3-evaluation-v1.toml")
    new = _require_zip(
        evaluation_handoff or project / "artifacts/handoff/visa-pcb-normal-bank-pcb3-evaluation-v1.zip",
        MEMBERS, EVALUATION_SHA256,
    )
    report = json.loads(new["evaluation.json"])
    cases = [json.loads(line) for line in new["cases.jsonl"].splitlines() if line]
    fit_zip = _require_zip(
        fit_handoff or project / "artifacts/handoff/visa-pcb-normal-bank-pcb3-v2.zip",
        FIT_MEMBERS, config["fit_handoff_sha256"],
    )
    fit = json.loads(fit_zip["fit.json"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pcb3_records = [row for row in plan["records"] if row["category"] == "pcb3"
                    and row["protocol_split"] == "external_final"]
    old_pcb3 = [row for row in prior_cases if row["category"] == "pcb3"]
    if (
        report.get("status") != "evaluated_post_test_exploratory_pcb3_normal_bank"
        or report.get("comparison_status") != "already_observed_visa_test_reuse_no_confirmatory_claim"
        or report.get("git", {}).get("commit") != EXPECTED_EVALUATION_COMMIT
        or report.get("comparison_config_sha256") != config["config_sha256"]
        or report.get("source_plan_sha256") != plan["plan_sha256"]
        or report.get("source_archive_sha256") != config["source_archive_sha256"]
        or report.get("baseline_evaluation_handoff_sha256") != config["baseline_evaluation_handoff_sha256"]
        or report.get("fit_handoff_sha256") != config["fit_handoff_sha256"]
        or report.get("fit_checkpoint_sha256") != fit["checkpoint_sha256"]
        or report.get("fit_checkpoint_sha256") != config["fit_checkpoint_sha256"]
        or report.get("cases_sha256") != hashlib.sha256(new["cases.jsonl"]).hexdigest()
        or report.get("test_count") != 201
        or report.get("test_anomaly_count") != 100
        or report.get("test_normal_count") != 101
        or report.get("automatic_pass_allowed") is not False
        or report.get("confirmatory_claim_allowed") is not False
        or len(cases) != len(pcb3_records) or len(cases) != len(old_pcb3)
    ):
        raise ValueError("PCB3 v2 comparison does not match the locked test evidence")
    old_by_name = {row["image"]: row for row in old_pcb3}
    records_by_name = {row["source_image_member"]: row for row in pcb3_records}
    if len(old_by_name) != 201 or len(records_by_name) != 201:
        raise ValueError("PCB3 test paths are duplicated")
    counts: Counter[tuple[int, str]] = Counter()
    for row, record in zip(cases, pcb3_records, strict=True):
        name = record["source_image_member"]
        old = old_by_name[name]
        if (
            row.get("image") != name or row.get("category") != "pcb3"
            or row.get("label") != record["label"]
            or any(row.get(key) != old.get(key) for key in
                   ("baseline_score", "baseline_threshold", "baseline_decision"))
        ):
            raise ValueError(f"PCB3 v2 case does not match the original plan and baseline: {name}")
        for prefix in ("baseline", "v2"):
            score, threshold = row.get(f"{prefix}_score"), row.get(f"{prefix}_threshold")
            if (
                not isinstance(score, (int, float)) or not math.isfinite(score)
                or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
                or row.get(f"{prefix}_decision") != (
                    "REVIEW_REQUIRED" if score >= threshold else "PASS"
                )
            ):
                raise ValueError(f"PCB3 score, threshold, or decision differs: {name}/{prefix}")
        if row["v2_threshold"] != fit["image_threshold"]:
            raise ValueError(f"PCB3 v2 threshold differs from normal calibration: {name}")
        counts[(row["label"], row["v2_decision"])] += 1
    expected = {"tp": counts[(1, "REVIEW_REQUIRED")], "fn": counts[(1, "PASS")],
                "fp": counts[(0, "REVIEW_REQUIRED")], "tn": counts[(0, "PASS")]}
    if expected != report["visa_normal_bank_v2"]["fixed_threshold"]["counts"]:
        raise ValueError("PCB3 v2 aggregate differs from the saved per-image decisions")
    return prior, report, fit, prior_cases, cases, audit, records_by_name


def select_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take all errors, without score-based sampling or retuning."""
    selected: list[dict[str, Any]] = []
    for row in cases:
        if row["label"] == 1 and row["v2_decision"] == "PASS":
            group = ("old_caught_v2_missed" if row["baseline_decision"] == "REVIEW_REQUIRED"
                     else "both_missed")
        elif row["label"] == 0 and row["v2_decision"] == "REVIEW_REQUIRED":
            group = ("new_false_positive" if row["baseline_decision"] == "PASS"
                     else "both_false_positive")
        else:
            continue
        selected.append({**row, "review_group": group})
    actual = Counter(row["review_group"] for row in selected)
    expected = {"old_caught_v2_missed": 3, "both_missed": 6,
                "new_false_positive": 5, "both_false_positive": 1}
    if dict(actual) != expected or len(selected) != 15:
        raise ValueError(f"Expected all 15 locked PCB3 errors; got {dict(actual)}")
    return sorted(selected, key=lambda row: (row["label"] == 0, row["image"]))


def render_card(row: dict[str, Any], old: dict[str, Any], record: dict[str, Any],
                audited: dict[str, Any], prior: dict[str, Any], report: dict[str, Any],
                fit: dict[str, Any], config: dict[str, Any], archive: tarfile.TarFile,
                old_sample_path: Path, v2_sample_path: Path) -> Image.Image:
    """Check source/masks/maps, then render two separately calibrated displays."""
    name = row["image"]
    image_bytes = _member_bytes(archive, name)
    if hashlib.sha256(image_bytes).hexdigest() != audited["image_sha256"]:
        raise ValueError(f"Source JPEG checksum differs: {name}")
    image = _jpeg(image_bytes, name)
    if image.size != (audited["width"], audited["height"]) or _pixel_sha(image) != audited["decoded_pixel_sha256"]:
        raise ValueError(f"Source JPEG decoded pixels differ: {name}")
    mask = _read_mask(
        archive, record, image.size, 256, config["mask_threshold"],
        audited["mask_sha256"], config["mask_binarization"],
    )
    if int(np.count_nonzero(mask)) != audited["mask_nonzero_256"]:
        raise ValueError(f"Source mask area differs: {name}")
    if file_sha256(old_sample_path) != old["sample_sha256"]:
        raise ValueError(f"Original saved map checksum differs: {name}")
    if file_sha256(v2_sample_path) != row["sample_sha256"]:
        raise ValueError(f"PCB3 v2 saved map checksum differs: {name}")
    old_expected = {
        "category": "pcb3", "label": row["label"], "image": name,
        "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
        "plan_sha256": prior["plan_sha256"],
        "execution_config_sha256": prior["evaluation_config_sha256"],
        "checkpoint_sha256": prior["checkpoint_sha256"],
        "source_archive_sha256": prior["source_archive_sha256"],
        "execution_git_commit": prior["git"]["commit"],
    }
    old_metadata, old_map, old_mask = _read_sample(old_sample_path, old_expected, 256)
    v2_expected = {
        "category": "pcb3", "label": row["label"], "image": name,
        "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
        "plan_sha256": report["source_plan_sha256"],
        "comparison_config_sha256": report["comparison_config_sha256"],
        "fit_checkpoint_sha256": report["fit_checkpoint_sha256"],
        "fit_handoff_sha256": report["fit_handoff_sha256"],
        "baseline_evaluation_handoff_sha256": report["baseline_evaluation_handoff_sha256"],
        "source_archive_sha256": report["source_archive_sha256"],
        "image_threshold": fit["image_threshold"],
    }
    v2_metadata, v2_map, v2_mask = _read_sample(v2_sample_path, v2_expected, 256)
    if (
        old_metadata["score"] != old["baseline_score"]
        or old_metadata["challenger_score"] != old["challenger_score"]
        or abs(candidate_image_score(v2_metadata["raw_score"], v2_map, config)
               - row["v2_score"]) > 1e-5
        or v2_metadata["score"] != row["v2_score"]
        or not np.array_equal(old_mask, mask)
        or not np.array_equal(v2_mask, mask)
    ):
        raise ValueError(f"Saved score or ground-truth mask differs: {name}")
    original = image.resize((256, 256), Image.Resampling.BILINEAR)
    outline = _outline(mask)
    panels: list[Image.Image] = []
    source = np.asarray(original).copy()
    source[outline] = (0, 255, 0)
    panels.append(Image.fromarray(source))
    for anomaly_map, threshold in (
        (old_map, prior["thresholds"]["pcb3"]["pixel"]),
        (v2_map, fit["pixel_threshold"]),
    ):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Pixel display scale must be positive and finite")
        heat = colorize_anomaly_values(np.clip(anomaly_map / threshold, 0, 1))
        overlay = np.rint(0.55 * np.asarray(original) + 0.45 * heat).astype(np.uint8)
        overlay[outline] = (0, 255, 0)
        panels.append(Image.fromarray(overlay))
    crop, _ = _crop_native(image, mask, v2_map)
    panels.append(crop)
    card = Image.new("RGB", (1024, 338), "white")
    for column, panel in enumerate(panels):
        card.paste(panel, (256 * column, 58))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    draw.text((4, 4), f"PCB3 {Path(name).name} | GT={'defect' if row['label'] else 'normal'} | {row['review_group']}",
              fill="black", font=font)
    draw.text((4, 18), f"old {row['baseline_score']:.3f}/{row['baseline_threshold']:.3f} {row['baseline_decision']}",
              fill="black", font=font)
    draw.text((4, 32), f"v2  {row['v2_score']:.3f}/{row['v2_threshold']:.3f} {row['v2_decision']}",
              fill="black", font=font)
    for column, label in enumerate(("Source + GT", "Old map / old pixel scale", "v2 map / v2 pixel scale",
                                    "Native crop near GT/map peak")):
        draw.text((column * 256 + 4, 317), label, fill="black", font=font)
    return card


def run(project: Path, archive_path: Path) -> dict[str, Any]:
    project = project.resolve(strict=True)
    archive_path = archive_path.resolve(strict=True)
    prior, report, fit, prior_cases, cases, audit, records = load_paired_evidence(project)
    selected = select_cases(cases)
    output = project / "artifacts/analyses/visa-pcb-normal-bank-pcb3-errors-v1"
    handoff = project / "artifacts/handoff/visa-pcb-normal-bank-pcb3-errors-v1.zip"
    if output.exists() or handoff.exists():
        raise FileExistsError("Existing PCB3 visual diagnostic output or handoff was not overwritten")
    if file_sha256(archive_path) != report["source_archive_sha256"]:
        raise ValueError("Original VisA source tar SHA-256 differs")
    config = load_comparison(project / "configs/visa-pcb-normal-bank-pcb3-evaluation-v1.toml")
    old_positions = {row["image"]: index for index, row in enumerate(prior_cases, start=1)}
    v2_positions = {row["image"]: index for index, row in enumerate(cases, start=1)}
    old_cases = {row["image"]: row for row in prior_cases}
    old_samples = project / "artifacts/visa-pcb-external-evaluation-v2/samples"
    v2_samples = project / "artifacts/visa-pcb-normal-bank-pcb3-evaluation-v1/samples"
    summary: dict[str, Any] = {
        "schema_version": 1, "status": "post_test_pcb3_visual_review_prepared_exploratory",
        "source_evaluation_handoff_sha256": EVALUATION_SHA256,
        "source_archive_sha256": report["source_archive_sha256"],
        "selected_count": len(selected),
        "groups": dict(Counter(row["review_group"] for row in selected)),
        "selection_rule": "All 9 v2 misses and all 6 v2 normal false positives, no score sampling",
        "model_changed": False, "thresholds_changed": False, "inference_performed": False,
        "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
        "interpretation": "Green is official ground truth. Color is a fixed-scale display, not proof of a physical defect. Old and v2 maps use their own normal-calibrated pixel scales and cannot be compared by raw color intensity.",
        "source_attribution": "VisA visual anomaly dataset (Amazon Science), CC BY 4.0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        sheets: dict[str, list[Image.Image]] = {"missed": [], "false-positives": []}
        with tarfile.open(archive_path, "r:") as archive:
            for index, row in enumerate(selected, start=1):
                name = row["image"]
                card = render_card(
                    row, old_cases[name], records[name], audit[name], prior, report, fit, config,
                    archive, old_samples / f"sample-{old_positions[name]:04d}.npz",
                    v2_samples / f"sample-{v2_positions[name]:04d}.npz",
                )
                kind = "missed" if row["label"] else "false-positives"
                filename = f"case-{index:02d}-{kind}-{Path(name).stem}.jpg"
                card.save(stage / filename, format="JPEG", quality=92)
                row["card"] = filename
                sheets[kind].append(card)
                print(f"PCB3 visual case {index}/{len(selected)}", flush=True)
        for kind, cards in sheets.items():
            sheet = Image.new("RGB", (1024, 338 * len(cards)), "white")
            for index, card in enumerate(cards):
                sheet.paste(card, (0, index * 338))
            sheet.save(stage / f"contact-{kind}.jpg", format="JPEG", quality=90)
        (stage / "selection.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "VisA PCB3 normal-bank v2 visual error review (post-test exploratory only).\n"
            "All 9 missed defects and 6 normal false positives are included.\n"
            "Each card: source at 256px plus green ground-truth outline; old map; v2 map; "
            "native-resolution crop near ground truth (or map peak for normals).\n"
            "Old and v2 map colors are scaled by separate normal-calibrated pixel thresholds; "
            "compare locations, not raw color intensity. Heat alone does not prove a defect.\n"
            "No retraining, inference, threshold tuning, automatic PASS or production decision.\n"
            "Source: VisA visual anomaly dataset (Amazon Science), CC BY 4.0.\n",
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory(prefix=f".{handoff.stem}-", dir=handoff.parent) as zip_temporary:
            staged_zip = Path(zip_temporary) / handoff.name
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(stage.iterdir()):
                    archive.write(path, path.name)
            os.replace(stage, output)
            os.replace(staged_zip, handoff)
    return {**summary, "output": str(output), "handoff_zip": str(handoff),
            "handoff_sha256": file_sha256(handoff)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args(argv))), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
