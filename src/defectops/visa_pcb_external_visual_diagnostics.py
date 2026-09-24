"""Read-only, post-test VisA PCB visual error review from sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
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
from defectops.realiad_followup_final import _read_sample
from defectops.visa_pcb_external_evaluation import _binary_mask, _jpeg, _member_bytes, _pixel_sha
from defectops.visa_pcb_external_protocol import canonical
from defectops.visa_pcb_external_registration import file_sha256


EVALUATION_ZIP_SHA256 = "ba2f5200757e212f97f0a431679a429aa068ee9140b0a28a760641e9427a0d2a"
EXPECTED_COMMIT = "d03a1cfc5340f177a98bd9436460c39b2a08adff"
MEMBERS = ["calibration.json", "cases.jsonl", "evaluation.json", "summary.md", "test-audit.json"]
CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4")


def load_evidence(handoff: Path, plan_path: Path, config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Require the already-reviewed v2 handoff and matching local locked contracts."""
    if file_sha256(handoff) != EVALUATION_ZIP_SHA256:
        raise ValueError("VisA v2 evaluation handoff SHA-256 differs")
    with zipfile.ZipFile(handoff) as archive:
        if sorted(archive.namelist()) != MEMBERS or archive.testzip() is not None:
            raise ValueError("VisA v2 handoff member set or CRC differs")
        evaluation = json.loads(archive.read("evaluation.json"))
        audit = json.loads(archive.read("test-audit.json"))
        cases_raw = archive.read("cases.jsonl")
    cases = [json.loads(line) for line in cases_raw.splitlines() if line.strip()]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    from defectops.visa_pcb_external_evaluation import load_evaluation_config
    config = load_evaluation_config(config_path, mask_correction_v2=True)
    final_records = [record for record in plan["records"] if record["protocol_split"] == "external_final"]
    if (
        evaluation.get("status") != "evaluated_exploratory_after_mask_parser_correction"
        or evaluation.get("git", {}).get("commit") != EXPECTED_COMMIT
        or evaluation.get("plan_sha256") != plan.get("plan_sha256")
        or evaluation.get("evaluation_config_sha256") != config["config_sha256"]
        or evaluation.get("mask_binarization") != "nonzero_before_nearest"
        or evaluation.get("official_test_count") != 802
        or evaluation.get("automatic_pass_allowed") is not False
        or audit.get("status") != "all_official_test_payloads_audited_before_scoring"
        or evaluation.get("test_payload_audit_sha256") != hashlib.sha256(canonical(audit)).hexdigest()
        or evaluation.get("cases_sha256") != hashlib.sha256(cases_raw).hexdigest()
        or len(final_records) != len(cases) or len(cases) != len(audit.get("records", []))
        or set(evaluation.get("thresholds", {})) != set(CATEGORIES)
    ):
        raise ValueError("VisA v2 evidence and locked local contract differ")
    audited: dict[str, dict[str, Any]] = {}
    counts: Counter[tuple[str, int, str]] = Counter()
    for record, row, data in zip(final_records, cases, audit["records"], strict=True):
        category, name, label = record["category"], record["source_image_member"], record["label"]
        if (
            row.get("category") != category or row.get("image") != name or row.get("label") != label
            or data.get("category") != category or data.get("image") != name or data.get("label") != label
            or data.get("mask") != record["source_mask_member"]
            or name in audited
        ):
            raise ValueError(f"VisA v2 case, audit and plan order differ: {name}")
        for candidate in ("baseline", "challenger"):
            score, threshold = row.get(f"{candidate}_score"), evaluation["thresholds"][category][candidate]
            if (
                not isinstance(score, (int, float)) or not math.isfinite(score)
                or row.get(f"{candidate}_threshold") != threshold
                or row.get(f"{candidate}_decision") != ("REVIEW_REQUIRED" if score >= threshold else "PASS")
            ):
                raise ValueError(f"VisA v2 score/decision differs: {name}")
            counts[(category, label, candidate + ":" + row[f"{candidate}_decision"])] += 1
        audited[name] = data
    for category in CATEGORIES:
        for candidate in ("baseline", "challenger"):
            actual = {
                "tp": counts[(category, 1, candidate + ":REVIEW_REQUIRED")],
                "fn": counts[(category, 1, candidate + ":PASS")],
                "fp": counts[(category, 0, candidate + ":REVIEW_REQUIRED")],
                "tn": counts[(category, 0, candidate + ":PASS")],
            }
            reported = evaluation["categories"][category][candidate]["fixed_threshold"]["counts"]
            if actual != reported:
                raise ValueError(f"VisA v2 confusion counts differ: {category}/{candidate}")
    return evaluation, cases, audited


def _stable_subset(rows: list[dict[str, Any]], limit: int, name: str) -> list[dict[str, Any]]:
    """Choose by source path hash, never by observed score or severity."""
    return sorted(rows, key=lambda row: (
        hashlib.sha256((name + ":" + row["image"]).encode()).hexdigest(), row["image"]
    ))[:limit]


def select_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], group: str) -> None:
        for row in rows:
            entry = selected.setdefault(row["image"], {**row, "review_groups": []})
            entry["review_groups"].append(group)

    lost = [row for row in cases if row["category"] == "pcb3" and row["label"] == 1
            and row["baseline_decision"] == "REVIEW_REQUIRED" and row["challenger_decision"] == "PASS"]
    if len(lost) != 17:
        raise ValueError(f"Expected all 17 PCB3 baseline-to-challenger losses; got {len(lost)}")
    add(lost, "pcb3_baseline_detected_challenger_missed_all")
    for category in CATEGORIES:
        both_missed = [row for row in cases if row["category"] == category and row["label"] == 1
                       and row["baseline_decision"] == row["challenger_decision"] == "PASS"]
        add(_stable_subset(both_missed, 8, category + ":both_missed"), category + "_both_missed_path_sample")
        normal_fp = [row for row in cases if row["category"] == category and row["label"] == 0
                     and (row["baseline_decision"] == "REVIEW_REQUIRED"
                          or row["challenger_decision"] == "REVIEW_REQUIRED")]
        add(_stable_subset(normal_fp, 4, category + ":normal_fp"), category + "_normal_fp_path_sample")
    return [selected[name] for name in sorted(selected)]


def _outline(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=np.bool_)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                           & mask[1:-1, :-2] & mask[1:-1, 2:])
    return mask & ~interior


def _crop_native(image: Image.Image, mask: np.ndarray, anomaly_map: np.ndarray) -> tuple[Image.Image, str]:
    if mask.any():
        ys, xs = np.nonzero(mask)
        center_x = (float(xs.min()) + float(xs.max()) + 1) / 2 / 256 * image.width
        center_y = (float(ys.min()) + float(ys.max()) + 1) / 2 / 256 * image.height
        label = "Native-resolution crop near GT (green outline on full view)"
    else:
        y, x = np.unravel_index(int(np.argmax(anomaly_map)), anomaly_map.shape)
        center_x = (x + 0.5) / 256 * image.width
        center_y = (y + 0.5) / 256 * image.height
        label = "Native-resolution crop near hottest map point"
    side = min(256, image.width, image.height)
    left = max(0, min(image.width - side, int(round(center_x - side / 2))))
    top = max(0, min(image.height - side, int(round(center_y - side / 2))))
    crop = image.crop((left, top, left + side, top + side)).resize((256, 256), Image.Resampling.LANCZOS)
    return crop, label


def render_card(case: dict[str, Any], audited: dict[str, Any], evaluation: dict[str, Any],
                archive: tarfile.TarFile, sample_path: Path) -> Image.Image:
    name, category = case["image"], case["category"]
    raw_bytes = _member_bytes(archive, name)
    if hashlib.sha256(raw_bytes).hexdigest() != audited["image_sha256"]:
        raise ValueError(f"Source JPEG checksum differs: {name}")
    image = _jpeg(raw_bytes, name)
    if image.size != (audited["width"], audited["height"]) or _pixel_sha(image) != audited["decoded_pixel_sha256"]:
        raise ValueError(f"Source JPEG decoded pixels differ: {name}")
    if case["label"]:
        mask_bytes = _member_bytes(archive, audited["mask"])
        if hashlib.sha256(mask_bytes).hexdigest() != audited["mask_sha256"]:
            raise ValueError(f"Source mask checksum differs: {name}")
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            if source_mask.format != "PNG" or source_mask.size != image.size:
                raise ValueError(f"Source mask encoding/size differs: {name}")
            mask = _binary_mask(source_mask, 256, 128, "nonzero_before_nearest")
        if int(mask.sum()) != audited["mask_nonzero_256"]:
            raise ValueError(f"Source mask area differs: {name}")
    else:
        mask = np.zeros((256, 256), dtype=np.bool_)
    if file_sha256(sample_path) != case["sample_sha256"]:
        raise ValueError(f"Saved map checksum differs: {sample_path}")
    expected = {
        "category": category, "label": case["label"], "image": name,
        "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
        "plan_sha256": evaluation["plan_sha256"],
        "execution_config_sha256": evaluation["evaluation_config_sha256"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "source_archive_sha256": evaluation["source_archive_sha256"],
        "execution_git_commit": evaluation["git"]["commit"],
    }
    metadata, anomaly_map, saved_mask = _read_sample(sample_path, expected, 256)
    if (
        metadata.get("score") != case["baseline_score"]
        or metadata.get("challenger_score") != case["challenger_score"]
        or not np.array_equal(saved_mask, mask)
    ):
        raise ValueError(f"Saved score or GT mask differs: {sample_path}")
    original = image.resize((256, 256), Image.Resampling.BILINEAR)
    image_pixels = np.asarray(original).copy()
    edge = _outline(mask)
    image_pixels[edge] = (0, 255, 0)
    fixed_scale = np.clip(anomaly_map / evaluation["thresholds"][category]["pixel"], 0, 1)
    heat = colorize_anomaly_values(fixed_scale)
    overlay = np.rint(0.55 * np.asarray(original) + 0.45 * heat).astype(np.uint8)
    overlay[edge] = (0, 255, 0)
    crop, _ = _crop_native(image, mask, anomaly_map)
    card = Image.new("RGB", (768, 338), "white")
    for column, picture in enumerate((Image.fromarray(image_pixels), Image.fromarray(overlay), crop)):
        card.paste(picture, (column * 256, 58))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    short = Path(name).name
    draw.text((4, 4), f"{category} {short} | GT={'anomaly' if case['label'] else 'normal'}", fill="black", font=font)
    draw.text((4, 18), f"baseline {case['baseline_score']:.2f}/{case['baseline_threshold']:.2f} {case['baseline_decision']}", fill="black", font=font)
    draw.text((4, 32), f"challenger {case['challenger_score']:.2f}/{case['challenger_threshold']:.2f} {case['challenger_decision']}", fill="black", font=font)
    draw.text((4, 316), "256px model input + green GT", fill="black", font=font)
    draw.text((260, 316), "Fixed-scale map overlay + GT", fill="black", font=font)
    draw.text((516, 316), "Native crop at GT/map peak", fill="black", font=font)
    return card


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluation, cases, audited = load_evidence(args.handoff, args.plan, args.config)
    selected = select_cases(cases)
    if args.output.exists():
        raise FileExistsError(f"Existing visual diagnostic output was not overwritten: {args.output}")
    if args.handoff_output.exists():
        raise FileExistsError(f"Existing visual diagnostic ZIP was not overwritten: {args.handoff_output}")
    if file_sha256(args.archive) != evaluation["source_archive_sha256"]:
        raise ValueError("Registered VisA source tar SHA-256 differs")
    positions = {row["image"]: index for index, row in enumerate(cases, start=1)}
    summary: dict[str, Any] = {
        "schema_version": 1, "status": "post_test_visual_review_prepared_exploratory",
        "source_evaluation_handoff_sha256": EVALUATION_ZIP_SHA256,
        "source_archive_sha256": evaluation["source_archive_sha256"],
        "selected_count": len(selected), "selected_groups": dict(Counter(
            group for row in selected for group in row["review_groups"]
        )),
        "selection_rule": "All 17 PCB3 baseline-detected/challenger-missed anomalies; per category up to 8 both-missed anomalies and 4 normal false positives, path-hash sampled, no score-based selection.",
        "model_changed": False, "thresholds_changed": False, "inference_performed": False,
        "automatic_pass_allowed": False,
        "interpretation": "Green is VisA ground truth, not a model decision. Red/yellow map colors are display only; a hot spot is not proof of the physical defect. Native crop lets reviewers inspect details lost at 256px.",
        "limitations": [
            "This v2 test was observed after a mask parser correction, so the analysis is descriptive and exploratory.",
            "Path-hash samples are neither random population estimates nor a basis for tuning and retesting on the same holdout.",
            "To claim improvement, preregister a new method and evaluate on a fresh untouched holdout or independent product data.",
            "VisA images are CC BY 4.0; retain dataset attribution and avoid redistributing derived sheets beyond authorized review.",
        ],
        "source_attribution": "VisA visual anomaly dataset (Amazon Science), CC BY 4.0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.handoff_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        stage = Path(temporary)
        with tarfile.open(args.archive, "r:") as archive:
            for order, row in enumerate(selected, start=1):
                sample_path = args.samples / f"sample-{positions[row['image']]:04d}.npz"
                card = render_card(row, audited[row["image"]], evaluation, archive, sample_path)
                filename = f"case-{order:03d}-{row['category']}.png"
                card.save(stage / filename)
                row["card"] = filename
                print(f"visual case {order}/{len(selected)}", flush=True)
        (stage / "selection.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "VisA PCB1-4 v2 post-test visual review (exploratory only).\n"
            "Left: source resized to model input with green GT outline. Middle: saved anomaly map "
            "on fixed category pixel-threshold scale. Right: native-resolution crop near GT or map peak.\n"
            "PCB3 changed-detection cases are all included. Other cases were selected by path hash, not score.\n"
            "Do not use this holdout for tuning, automatic PASS, or production decisions.\n"
            "Source: VisA visual anomaly dataset (Amazon Science), CC BY 4.0.\n",
            encoding="utf-8",
        )
        os.replace(stage, args.output)
    with zipfile.ZipFile(args.handoff_output, mode="x", compression=zipfile.ZIP_DEFLATED) as handoff:
        for path in sorted(args.output.iterdir()):
            handoff.write(path, path.name)
    summary["output"] = str(args.output)
    summary["handoff_zip"] = str(args.handoff_output)
    summary["handoff_sha256"] = file_sha256(args.handoff_output)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handoff-output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
