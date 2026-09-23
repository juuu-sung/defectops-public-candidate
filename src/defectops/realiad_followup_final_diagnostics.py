"""Read-only, post-final Real-IAD error analysis. Never fits, scores or tunes a model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from defectops.manifest import canonical
from defectops.padim_experiment import checksum
from defectops.padim_inspection import colorize_anomaly_values
from defectops.realiad_followup_final import (
    EXPECTED_CANDIDATE,
    EXPECTED_SELECTION_SHA256,
    _archive_member,
    _read_sample,
)


FINAL_ZIP_SHA256 = "82a01ed48b600915c7a86017783d3a38a16a06f3bf3d98e878c3a34ab6a2dd45"
SEEDS = (42, 43, 44)
SHORT = "pcb"
CATEGORY = "realiad_pcb"
GROUND_TRUTH_THRESHOLD = 128


def _entry(archive: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in archive.namelist() if name.replace("\\", "/") == suffix]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one final handoff entry: {suffix}")
    return archive.read(matches[0])


def _json(archive: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    return json.loads(_entry(archive, suffix))


def _checked_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    manifest = _json(archive, "contracts/realiad-pcb-final-sealed-v1.json")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("Final manifest content hash differs")
    records = manifest.get("records", [])
    if len(records) != 360 or len({r["source_image_member"] for r in records}) != 360:
        raise ValueError("Unexpected PCB final manifest count or duplicate source image")
    return manifest


def load_final_rows(handoff: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if checksum(handoff) != FINAL_ZIP_SHA256:
        raise ValueError("Final handoff ZIP SHA-256 differs from the verified final bundle")
    with zipfile.ZipFile(handoff) as archive:
        manifest = _checked_manifest(archive)
        selection = _json(archive, "contracts/selection-report.json")
        if (
            selection.get("selection_report_sha256") != EXPECTED_SELECTION_SHA256
            or selection.get("selected_candidate_id") != EXPECTED_CANDIDATE
        ):
            raise ValueError("Final handoff candidate selection differs")
        records = manifest["records"]
        bundles: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            root = f"{SHORT}-seed{seed}"
            evaluation = _json(archive, f"{root}/evaluation.json")
            fit = _json(archive, f"{root}/fit.json")
            raw = _entry(archive, f"{root}/predictions.jsonl")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if (
                evaluation.get("status") != "evaluated_final_locked"
                or evaluation.get("category") != CATEGORY
                or evaluation.get("seed") != seed
                or evaluation.get("candidate_id") != EXPECTED_CANDIDATE
                or evaluation.get("selection_report_sha256") != EXPECTED_SELECTION_SHA256
                or evaluation.get("final_manifest_sha256") != manifest["manifest_sha256"]
                or evaluation.get("predictions_sha256") != hashlib.sha256(raw).hexdigest()
                or evaluation.get("sample_count") != len(records)
                or len(evaluation.get("sample_artifact_sha256", [])) != len(records)
                or len(rows) != len(records)
                or fit.get("threshold") != evaluation.get("threshold")
                or fit.get("checkpoint_sha256") != evaluation.get("checkpoint_sha256")
            ):
                raise ValueError(f"Final handoff contract differs for seed {seed}")
            threshold = evaluation["threshold"]
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"Invalid image threshold for seed {seed}")
            counts = Counter()
            for record, row in zip(records, rows, strict=True):
                score = row.get("score")
                if (
                    row.get("source_image_member") != record["source_image_member"]
                    or row.get("label") != record["label"]
                    or row.get("defect_type") != record["defect_type"]
                    or row.get("candidate_id") != EXPECTED_CANDIDATE
                    or row.get("threshold") != threshold
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or row.get("decision") != ("REVIEW_REQUIRED" if score >= threshold else "PASS")
                ):
                    raise ValueError(f"Final prediction differs at seed {seed}: {record['source_image_member']}")
                counts[(record["label"], row["decision"])] += 1
            actual = {
                "tp": counts[(1, "REVIEW_REQUIRED")], "fn": counts[(1, "PASS")],
                "fp": counts[(0, "REVIEW_REQUIRED")], "tn": counts[(0, "PASS")],
            }
            if actual != evaluation["metrics"]["fixed_threshold"]["counts"]:
                raise ValueError(f"Final confusion counts differ for seed {seed}")
            bundles[seed] = {"evaluation": evaluation, "fit": fit, "rows": rows}
    return manifest, bundles, records


def build_cases(records: list[dict[str, Any]], bundles: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for index, record in enumerate(records):
        seed_rows = {seed: bundles[seed]["rows"][index] for seed in SEEDS}
        votes = sum(row["decision"] == "REVIEW_REQUIRED" for row in seed_rows.values())
        label = record["label"]
        group = (
            "consensus_false_negative" if label == 1 and votes == 0 else
            "partial_false_negative" if label == 1 and votes < 3 else
            "consensus_true_positive" if label == 1 else
            "consensus_false_positive" if votes == 3 else
            "partial_false_positive" if votes > 0 else
            "consensus_true_negative"
        )
        cases.append({
            "index": index, "category": CATEGORY, "source_image_member": record["source_image_member"],
            "sample_group": record["sample_group"], "defect_type": record["defect_type"],
            "label": label, "group": group, "review_votes": votes,
            "seeds": {
                str(seed): {
                    "score": row["score"], "threshold": row["threshold"],
                    "margin": row["score"] - row["threshold"],
                    "margin_ratio": row["score"] / row["threshold"] - 1,
                    "decision": row["decision"],
                }
                for seed, row in seed_rows.items()
            },
        })
    return cases


def summarize(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    by_defect: dict[str, Counter[str]] = defaultdict(Counter)
    groups: Counter[str] = Counter()
    for case in cases:
        groups[case["group"]] += 1
        if case["label"] == 1:
            by_defect[case["defect_type"]][case["group"]] += 1
    return {
        "schema_version": 1, "status": "analyzed_postfinal_descriptive",
        "purpose": "Describe frozen final prediction errors; not model selection or tuning.",
        "source_handoff_sha256": FINAL_ZIP_SHA256,
        "final_manifest_sha256": manifest["manifest_sha256"],
        "candidate_id": EXPECTED_CANDIDATE, "category": CATEGORY, "seeds": list(SEEDS),
        "total_images": len(cases), "group_counts": dict(sorted(groups.items())),
        "by_defect_type": {
            defect: {"count": sum(counts.values()), **dict(sorted(counts.items()))}
            for defect, counts in sorted(by_defect.items())
        },
        "model_changed": False, "thresholds_changed": False,
        "automatic_pass_allowed": False,
        "limitations": [
            "The final holdout has already been observed; this is descriptive error analysis only.",
            "Do not use these cases to tune a model, threshold, or candidate and then claim a confirmatory final result.",
            "Original source images, saved maps, and checkpoint files are not included in the final handoff ZIP.",
        ],
    }


def _ground_truth_outline(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
        & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    edge = (mask & ~interior).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(edge).filter(ImageFilter.MaxFilter(3))) > 0


def _render_case(
    case: dict[str, Any], record: dict[str, Any], evaluations: dict[int, dict[str, Any]],
    source: zipfile.ZipFile, samples_root: Path,
) -> Image.Image:
    image_bytes = _archive_member(source, record["source_image_member"], record["image_sha256"])
    with Image.open(io.BytesIO(image_bytes)) as raw_image:
        if raw_image.format != "JPEG" or raw_image.mode != "RGB":
            raise ValueError("Unexpected Real-IAD PCB source image encoding")
        original_size = raw_image.size
        original = raw_image.resize((256, 256), Image.Resampling.BILINEAR).copy()
    if record["label"] == 0:
        ground_truth = np.zeros((256, 256), dtype=np.bool_)
    else:
        mask_bytes = _archive_member(source, record["source_mask_member"], record["mask_sha256"])
        with Image.open(io.BytesIO(mask_bytes)) as raw_mask:
            if raw_mask.format != "PNG" or raw_mask.size != original_size:
                raise ValueError("Unexpected Real-IAD PCB ground-truth mask encoding")
            ground_truth = np.asarray(
                raw_mask.convert("L").resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8
            ) >= GROUND_TRUTH_THRESHOLD
    outline = _ground_truth_outline(ground_truth)
    original_pixels = np.asarray(original).copy()
    original_pixels[outline] = (0, 255, 0)
    panel = Image.new("RGB", (1024, 296), "white")
    panel.paste(Image.fromarray(original_pixels), (0, 32))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    draw.text((4, 3), f"{case['index']:03d} {case['defect_type']} {Path(case['source_image_member']).name}", fill="black", font=font)
    draw.text((4, 17), "Original + GT outline (green)", fill="black", font=font)
    for column, seed in enumerate(SEEDS, start=1):
        evaluation = evaluations[seed]
        sample_path = samples_root / f"{SHORT}-seed{seed}" / "samples" / f"sample-{case['index']:04d}.npz"
        if checksum(sample_path) != evaluation["sample_artifact_sha256"][case["index"]]:
            raise ValueError(f"Saved map file checksum differs: {sample_path}")
        expected = {
            "selection_report_sha256": EXPECTED_SELECTION_SHA256,
            "final_manifest_sha256": evaluation["final_manifest_sha256"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "threshold": evaluation["threshold"], "category": CATEGORY, "seed": seed,
            "source_image_member": record["source_image_member"], "image_sha256": record["image_sha256"],
            "label": record["label"], "defect_type": record["defect_type"],
        }
        metadata, anomaly_map, saved_mask = _read_sample(sample_path, expected, 256)
        if metadata["score"] != case["seeds"][str(seed)]["score"] or not np.array_equal(saved_mask, ground_truth):
            raise ValueError(f"Saved map/ground truth differs from final prediction: {sample_path}")
        normalized = np.clip(anomaly_map / evaluation["pixel_threshold"], 0, 1)
        heatmap = colorize_anomaly_values(normalized)
        overlay_pixels = np.rint(0.55 * np.asarray(original) + 0.45 * heatmap).astype(np.uint8)
        overlay_pixels[outline] = (0, 255, 0)
        overlay = Image.fromarray(overlay_pixels)
        panel.paste(overlay, (column * 256, 32))
        outcome = case["seeds"][str(seed)]
        draw.text((column * 256 + 4, 4), f"seed {seed}: {outcome['score']:.2f} / {outcome['threshold']:.2f}", fill="black", font=font)
        draw.text((column * 256 + 4, 18), f"margin {outcome['margin']:+.2f}; heat is relative", fill="black", font=font)
    return panel


def render_sheets(
    cases: list[dict[str, Any]], records: list[dict[str, Any]], bundles: dict[int, dict[str, Any]],
    manifest: dict[str, Any], source_zip: Path, samples_root: Path, output: Path,
) -> list[str]:
    if checksum(source_zip) != manifest["source_archive_sha256"]:
        raise ValueError("PCB source archive SHA-256 differs from sealed manifest")
    selected = [case for case in cases if case["group"] in {"consensus_false_negative", "consensus_false_positive"}]
    evaluations = {seed: bundle["evaluation"] for seed, bundle in bundles.items()}
    paths: list[str] = []
    with zipfile.ZipFile(source_zip) as source:
        for defect in sorted({case["defect_type"] for case in selected}):
            selected_group = [case for case in selected if case["defect_type"] == defect]
            for page_number, start in enumerate(range(0, len(selected_group), 5), start=1):
                batch = selected_group[start:start + 5]
                sheet = Image.new("RGB", (1024, 296 * len(batch)), "#dddddd")
                for row_number, case in enumerate(batch):
                    panel = _render_case(case, records[case["index"]], evaluations, source, samples_root)
                    sheet.paste(panel, (0, row_number * 296))
                name = f"{defect.lower()}-{page_number:02d}.png"
                path = output / name
                if path.exists():
                    raise FileExistsError(f"Existing diagnostic sheet was not overwritten: {path}")
                sheet.save(path)
                paths.append(name)
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, bundles, records = load_final_rows(args.handoff)
    cases = build_cases(records, bundles)
    analysis = summarize(cases, manifest)
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"Existing diagnostic output was not overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.source_zip or args.samples_root:
        if not args.source_zip or not args.samples_root:
            raise ValueError("Both --source-zip and --samples-root are required for contact sheets")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        if args.source_zip:
            analysis["contact_sheets"] = render_sheets(
                cases, records, bundles, manifest, args.source_zip, args.samples_root, stage
            )
        (stage / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / "cases.jsonl").write_text(
            "".join(json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
        )
        os.replace(stage, output)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--samples-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
