"""Read-only post-test MPI-PCB map diagnostics; no inference, fitting or tuning."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
               str(Path(__file__).resolve().parents[1])]

from defectops.padim_experiment import git_state
from defectops.padim_inspection import colorize_anomaly_values
from defectops.realiad_followup_development import candidate_image_score
from scripts.mpi_pcb_external_protocol import REGIONS, decode_image, mask_details, sha256_file
from scripts.mpi_pcb_external_v2_protocol import (
    DAMAGED_SHA256, MASK_MEMBER, MASK_SHA256, REPAIR_MEMBER, REPAIRED_RGB_SHA256, load_config,
)

NAME = "mpi-pcb-map-diagnostics-v1"
EVALUATION_NAME = "mpi-pcb-external-evaluation-v2"
EVALUATION_SHA256 = "768714d947673b1b03f91782de802ce33f8796c21af582a5e5e078d172e2315e"
MODEL_KEYS = ("realiad_pcb", "visa_pcb3_v2")
MODEL_LABELS = ("Real-IAD PCB", "VisA PCB3 v2")
CANDIDATE = {"image_score": "mean_of_top_map_fraction", "top_fraction": 0.01}
TOLERANCE = 1e-5
TOP_K = math.ceil(256 * 256 * 0.01)
SOURCE_ATTRIBUTION = (
    "MPI-PCB Dataset v1.0; Diulhio Candido de Oliveira, Bogdan Tomoyuki Nassu, "
    "Marco Aurelio Wehrmeister; DOI 10.5281/zenodo.8213098; CC BY 4.0. "
    "Derived diagnostic overlays, not original photographs or new production labels."
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_evidence(project: Path, handoff: Path) -> dict:
    """Pin the reviewed 9-JSON bundle, not editable local score copies."""
    if sha256_file(handoff) != EVALUATION_SHA256:
        raise ValueError("Reviewed MPI-PCB v2 handoff SHA-256 differs")
    names = {"input-lock.json", "recovery.json", "test-audit.json", "cohort-amendment.json",
             "calibration-replay.json", "evaluation.json", "handoff-manifest.json",
             *(f"{key}-scores.json" for key in MODEL_KEYS)}
    with zipfile.ZipFile(handoff) as archive:
        if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
            raise ValueError("MPI-PCB v2 bundle members differ")
        reports = {name: json.loads(archive.read(name)) for name in names}
        members = reports["handoff-manifest.json"]["members"]
        if len(members) != 8 or {r["name"] for r in members} != names - {"handoff-manifest.json"}:
            raise ValueError("Incomplete handoff manifest")
        for row in members:
            if digest(archive.read(row["name"])) != row["sha256"]:
                raise ValueError(f'Report checksum differs: {row["name"]}')
    config = load_config(project / "configs/mpi-pcb-external-evaluation-v2.toml")
    evaluation = reports["evaluation.json"]
    if (reports["input-lock.json"]["evaluation_config"] != config
            or evaluation["status"] != "mpi_pcb_external_evaluated_integrity_amended_v2"
            or evaluation["automatic_pass_allowed"] is not False
            or evaluation["confirmatory_claim_allowed"] is not False
            or evaluation["cohort_accounting"] != reports["cohort-amendment.json"]):
        raise ValueError("Expected the reviewed complete integrity-amended evaluation")
    audit = {p["key"]: p for p in reports["test-audit.json"]["photos"]}
    cohort = reports["cohort-amendment.json"]["photo_keys"]
    if len(cohort) != 109 or len(set(cohort)) != 109 or "good/0013" in cohort:
        raise ValueError("Expected the fixed 109-photo cohort")
    for key in MODEL_KEYS:
        model = reports[f"{key}-scores.json"]
        if ([p["key"] for p in model["photos"]] != cohort
                or model["threshold"] != config["thresholds"][key]
                or model["checkpoint_sha256"] != evaluation["models"][key]["checkpoint_sha256"]):
            raise ValueError("Model, threshold, or cohort changed")
        counts = Counter()
        for i, photo in enumerate(model["photos"], 1):
            rows = photo["regions"]
            expected = audit[photo["key"]]
            if (photo["label"] != expected["label"]
                    or [r["region"] for r in rows] != REGIONS
                    or [r["member"] for r in rows] != [r["member"] for r in expected["regions"]]
                    or photo["map_file"] != f"maps/{key}/photo-{i:03d}.npz"
                    or photo["threshold"] != model["threshold"]
                    or not all(math.isfinite(r["score"]) for r in rows)
                    or photo["score"] != max(r["score"] for r in rows)):
                raise ValueError("Photo identity, map path, or score changed")
            flagged = photo["score"] >= model["threshold"]
            expected_decision = "REVIEW_REQUIRED" if flagged else "BELOW_THRESHOLD_NOT_AUTO_PASS"
            if photo["decision"] != expected_decision or photo["automatic_pass_allowed"] is not False:
                raise ValueError("Photo decision changed")
            counts[(photo["label"], flagged)] += 1
        actual = dict(tp=counts[(1, True)], fn=counts[(1, False)],
                      fp=counts[(0, True)], tn=counts[(0, False)])
        if actual != evaluation["models"][key]["metrics"]["fixed_threshold"]["counts"]:
            raise ValueError("Evaluation confusion matrix differs")
    return reports


def load_maps(evaluation_dir: Path, photo: dict) -> np.ndarray:
    root = evaluation_dir.resolve(strict=True)
    path = (root / photo["map_file"]).resolve(strict=True)
    path.relative_to(root)
    if sha256_file(path) != photo["map_sha256"]:
        raise ValueError(f'Saved map checksum differs: {photo["key"]}')
    with np.load(path, allow_pickle=False) as sample:
        if set(sample.files) != {"anomaly_maps", "regions"}:
            raise ValueError("Unexpected map payload")
        maps, regions = sample["anomaly_maps"], sample["regions"]
        if (maps.shape != (12, 256, 256) or maps.dtype != np.float32
                or not np.isfinite(maps).all() or np.any(maps < 0)
                or regions.dtype.kind != "U" or regions.tolist() != REGIONS):
            raise ValueError("Expected 12 ordered finite nonnegative float32 maps")
        return maps


def region_statistics(values: np.ndarray, mask: np.ndarray | None, saved_score: float) -> dict:
    """Tie-aware top-1% attribution; no image threshold used as a pixel threshold."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid native map")
    score = candidate_image_score(0.0, values, CANDIDATE)
    if not math.isfinite(saved_score) or abs(score - saved_score) > TOLERANCE:
        raise ValueError("Saved region score differs from its top-1% map mean")
    flat = values.reshape(-1)
    top = np.partition(flat, len(flat) - TOP_K)[-TOP_K:]
    cutoff = float(top.min())
    above, tied = flat > cutoff, flat == cutoff
    slots = TOP_K - int(above.sum())
    result = {"saved_score": saved_score, "score_from_map": score,
              "absolute_score_difference": abs(score - saved_score),
              "map_mean": float(np.mean(values, dtype=np.float64)), "map_max": float(flat.max()),
              "top1pct_pixels": TOP_K, "top1pct_cutoff": cutoff,
              "top1pct_cutoff_ties": int(tied.sum()), "top1pct_cutoff_slots": slots,
              "gt_available": mask is not None, "gt": None}
    if mask is None:
        return result
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError("Expected native boolean mask")
    gt = mask.reshape(-1)
    n = int(gt.sum())
    gt_above = int(np.count_nonzero(gt & above))
    gt_tied = int(np.count_nonzero(gt & tied))
    bg_tied = int(np.count_nonzero(~gt & tied))
    minimum = gt_above + max(0, slots - bg_tied)
    maximum = gt_above + min(slots, gt_tied)
    strict_sum = float(np.sum(flat[gt & above], dtype=np.float64))
    top_sum = float(np.sum(top, dtype=np.float64))
    inside, outside = flat[gt], flat[~gt]
    rank_auc = None
    if n and len(outside):
        background = np.sort(outside)
        lo = np.searchsorted(background, inside, side="left")
        hi = np.searchsorted(background, inside, side="right")
        rank_auc = float(np.sum((lo + hi) * 0.5, dtype=np.float64) / (n * len(outside)))
    ys, xs = np.where(mask)
    result["gt"] = {
        "pixels": n, "area_fraction": n / len(flat),
        "bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if n else None,
        "mean": float(np.mean(inside, dtype=np.float64)) if n else None,
        "maximum": float(inside.max()) if n else None,
        "non_gt_mean": float(np.mean(outside, dtype=np.float64)) if len(outside) else None,
        "within_region_rank_auc": rank_auc,
        "top1pct_overlap_pixels_min": minimum, "top1pct_overlap_pixels_max": maximum,
        "top1pct_gt_recall_min": minimum / n if n else None,
        "top1pct_gt_recall_max": maximum / n if n else None,
        "top1pct_score_share_min": (strict_sum + cutoff * (minimum - gt_above)) / top_sum if top_sum > 0 else None,
        "top1pct_score_share_max": (strict_sum + cutoff * (maximum - gt_above)) / top_sum if top_sum > 0 else None,
        "smaller_than_top1pct_pool": 0 < n < TOP_K,
    }
    return result


def read_photo(source, audited, recovered: bytes):
    images, masks = [], []
    for row in audited["regions"]:
        raw = source.read(row["member"])
        if row["member"] == REPAIR_MEMBER:
            if digest(raw) != DAMAGED_SHA256:
                raise ValueError("Original damaged tile identity changed")
            raw = recovered
        if digest(raw) != row["member_sha256"]:
            raise ValueError(f'Image checksum differs: {row["member"]}')
        image = decode_image(raw)
        if digest(image.tobytes()) != row["decoded_rgb_sha256"]:
            raise ValueError("Image decoded pixels changed")
        images.append(image)
        if row["mask_member"] is None:
            masks.append(np.zeros((256, 256), dtype=np.bool_))
            continue
        raw_mask = source.read(row["mask_member"])
        if digest(raw_mask) != row["mask_sha256"]:
            raise ValueError("Mask checksum changed")
        if row["mask_member"] == MASK_MEMBER:
            if digest(raw_mask) != MASK_SHA256 or "mask" in row:
                raise ValueError("Known mask-only exception changed")
            masks.append(None)
            continue
        if mask_details(raw_mask) != row["mask"]:
            raise ValueError("Mask decoding differs from the reviewed native nonzero rule")
        with Image.open(io.BytesIO(raw_mask)) as image_mask:
            masks.append(np.asarray(image_mask.convert("L")) > 0)
    return images, masks


def comparison_group(photos: dict) -> str:
    flags = [photos[key]["decision"] == "REVIEW_REQUIRED" for key in MODEL_KEYS]
    if photos[MODEL_KEYS[0]]["label"]:
        return ("both_missed", "visa_only_caught", "realiad_only_caught", "both_caught")[flags[0] * 2 + flags[1]]
    return ("both_below_threshold", "visa_only_false_alarm", "realiad_only_false_alarm", "both_false_alarm")[flags[0] * 2 + flags[1]]


def describe_photo(photo, maps, masks, localization_eligible):
    rows = [{"region": region, **region_statistics(values, mask if localization_eligible else None, saved["score"])}
            for region, values, mask, saved in zip(REGIONS, maps, masks, photo["regions"], strict=True)]
    if abs(max(row["score_from_map"] for row in rows) - photo["score"]) > TOLERANCE:
        raise ValueError("Photo score reconstruction differs")
    maximum = max(row["saved_score"] for row in rows)
    winners = [row for row in rows if row["saved_score"] == maximum]
    gt_rows = [row for row in rows if row["gt"] and row["gt"]["pixels"] > 0]
    return {"score": photo["score"], "threshold": photo["threshold"], "decision": photo["decision"],
            "margin": photo["score"] - photo["threshold"],
            "highest_regions": [row["region"] for row in winners],
            "highest_region_gt_any": any(row["gt"]["pixels"] > 0 for row in winners) if localization_eligible else None,
            "highest_region_gt_all": all(row["gt"]["pixels"] > 0 for row in winners) if localization_eligible else None,
            "best_gt_region_score": max(row["saved_score"] for row in gt_rows) if gt_rows else None,
            "maximum_minus_best_gt_region_score": maximum - max(row["saved_score"] for row in gt_rows) if gt_rows else None,
            "all_gt_regions_have_zero_top1pct_overlap": all(row["gt"]["top1pct_overlap_pixels_max"] == 0 for row in gt_rows) if gt_rows else None,
            "regions": rows}


def outline(mask):
    if mask is None:
        return np.zeros((256, 256), dtype=np.bool_)
    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def render_page(case, images, masks, maps, page):
    """Six tiles per sheet, fixed model-specific display, no per-image min/max."""
    card = Image.new("RGB", (768, 92 + 6 * 294 + 22), "white")
    draw, font = ImageDraw.Draw(card), ImageFont.load_default()
    draw.text((4, 4), f"MPI-PCB {case['key']} | {case['group']} | page {page+1}/2", fill="black", font=font)
    draw.text((4, 18), "Green: official GT. Color: saved map / (1.5 x model IMAGE threshold), NOT pixel decision.", fill="black", font=font)
    draw.text((4, 32), "Fixed display per model; saturation possible. Do not compare models by color intensity.", fill="black", font=font)
    if not case["localization_eligible"]:
        draw.text((4, 46), "Known missing GT tile: exclude this photo from ALL location summaries; no guessed mask.", fill="red", font=font)
    for column, title in enumerate(("Source + GT", *MODEL_LABELS)):
        draw.text((column * 256 + 4, 64), title, fill="black", font=font)
    for slot, index in enumerate(range(page * 6, (page + 1) * 6)):
        y = 92 + slot * 294
        original = np.asarray(images[index])
        edge = outline(masks[index])
        source = original.copy()
        source[edge] = (0, 255, 0)
        panels = [Image.fromarray(source)]
        for key in MODEL_KEYS:
            heat = colorize_anomaly_values(np.clip(maps[key][index] / (1.5 * case["models"][key]["threshold"]), 0, 1))
            overlay = np.rint(original * 0.6 + heat * 0.4).astype(np.uint8)
            overlay[edge] = (0, 255, 0)
            panels.append(Image.fromarray(overlay))
        for column, panel in enumerate(panels):
            card.paste(panel, (column * 256, y))
        gt_text = str(int(masks[index].sum())) if masks[index] is not None else "UNAVAILABLE"
        draw.text((4, y+259), f"{REGIONS[index]} | GT px {gt_text}", fill="black", font=font)
        for column, key in enumerate(MODEL_KEYS, 1):
            model = case["models"][key]
            region = model["regions"][index]
            flag = " MAX REGION" if REGIONS[index] in model["highest_regions"] else ""
            draw.text((column*256+4, y+259), f"tile {region['saved_score']:.3f}{flag}", fill="black", font=font)
            draw.text((column*256+4, y+273), f"photo {model['score']:.3f} / {model['threshold']:.3f}", fill="black", font=font)
    draw.text((4, card.height-18), "MPI-PCB v1.0 (Oliveira et al.), CC BY 4.0 | DOI: 10.5281/zenodo.8213098 | modified overlay", fill="black", font=font)
    return card


def summarize(cases):
    result = {"photos": len(cases), "groups": dict(Counter(c["group"] for c in cases)), "models": {}}
    for key in MODEL_KEYS:
        groups = {}
        for name, rows in (("all_defects", [c for c in cases if c["label"] == 1]),
                           ("missed_defects", [c for c in cases if c["label"] == 1 and c["models"][key]["decision"] != "REVIEW_REQUIRED"]),
                           ("normal_false_alarms", [c for c in cases if c["label"] == 0 and c["models"][key]["decision"] == "REVIEW_REQUIRED"])):
            eligible = [c for c in rows if c["localization_eligible"]]
            gt_photos = [c for c in eligible if c["label"] == 1]
            metrics = ["within_region_rank_auc", "top1pct_gt_recall_min", "top1pct_gt_recall_max",
                       "top1pct_score_share_min", "top1pct_score_share_max", "pixels"]
            # Mean within each photo, then median across photos: not pooled independent tiles.
            medians = {}
            for metric in metrics:
                means = []
                for case in gt_photos:
                    values = [r["gt"][metric] for r in case["models"][key]["regions"]
                              if r["gt"] and r["gt"]["pixels"] and r["gt"][metric] is not None]
                    if values:
                        means.append(float(np.mean(values)))
                medians[metric] = float(np.median(means)) if means else None
            groups[name] = {
                "photos": len(rows), "location_eligible_photos": len(eligible),
                "location_ineligible_photos": [c["key"] for c in rows if not c["localization_eligible"]],
                "gt_photo_denominator": len(gt_photos),
                "highest_region_has_no_gt": sum(not c["models"][key]["highest_region_gt_any"] for c in gt_photos),
                "all_gt_regions_zero_top1pct_overlap": sum(c["models"][key]["all_gt_regions_have_zero_top1pct_overlap"] for c in gt_photos),
                "all_gt_regions_smaller_than_656_pixels": sum(all(r["gt"]["pixels"] < TOP_K for r in c["models"][key]["regions"] if r["gt"] and r["gt"]["pixels"]) for c in gt_photos),
                "median_of_per_photo_means_over_gt_bearing_regions": medians,
            }
        result["models"][key] = groups
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def publish(stage, output, handoff):
    """Package only our new diagnostic files; never archive source maps/checkpoints."""
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = {"members": [{"name": p.relative_to(stage).as_posix(), "sha256": sha256_file(p)} for p in files],
                "content": "Diagnostic JSON, HTML and attributed derived cards; no raw maps or checkpoints"}
    write_json(stage / "handoff-manifest.json", manifest)
    if output.exists() or handoff.exists():
        raise FileExistsError("Existing diagnostic output preserved")
    # mkdir without exist_ok reserves a new destination; no replacement of old result folders.
    output.mkdir()
    try:
        with zipfile.ZipFile(handoff, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(stage).as_posix())
        for path in list(stage.iterdir()):
            path.rename(output / path.name)
    except Exception:
        # Preserve any partial diagnostic output for inspection; never delete or overwrite it.
        raise


def run(project: Path, archive_path: Path) -> dict:
    project = project.resolve(strict=True)
    evaluation_dir = project / "artifacts/evaluations" / EVALUATION_NAME
    handoff_source = project / "artifacts/handoff" / f"{EVALUATION_NAME}.zip"
    output = project / "artifacts/analyses" / NAME
    handoff = project / "artifacts/handoff" / f"{NAME}.zip"
    if output.exists() or handoff.exists():
        raise FileExistsError("Existing diagnostic output preserved; share it before retrying")
    execution_git = git_state(project)
    reports = load_evidence(project, handoff_source)
    if sha256_file(archive_path) != reports["evaluation.json"]["archive_sha256"]:
        raise ValueError("Original MPI-PCB archive checksum differs")
    recovered = (evaluation_dir / "recovery/grid1-3-defect-0054.png").read_bytes()
    if digest(recovered) != reports["recovery.json"]["recovered_png_sha256"]:
        raise ValueError("Saved reconstructed tile checksum differs; no network fallback")
    with decode_image(recovered) as image:
        if digest(image.tobytes()) != REPAIRED_RGB_SHA256:
            raise ValueError("Reconstructed RGB pixels differ")
    audit = {p["key"]: p for p in reports["test-audit.json"]["photos"]}
    indices = {key: {p["key"]: p for p in reports[f"{key}-scores.json"]["photos"]} for key in MODEL_KEYS}
    output.parent.mkdir(parents=True, exist_ok=True)
    handoff.parent.mkdir(parents=True, exist_ok=True)
    cases, maximum_difference = [], 0.0
    with tempfile.TemporaryDirectory(prefix=f".{NAME}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "cards").mkdir()
        with zipfile.ZipFile(archive_path) as source:
            if len(source.namelist()) != len(set(source.namelist())):
                raise ValueError("Duplicate source ZIP member names")
            for index, photo_key in enumerate(reports["cohort-amendment.json"]["photo_keys"], 1):
                photos = {key: indices[key][photo_key] for key in MODEL_KEYS}
                images, masks = read_photo(source, audit[photo_key], recovered)
                eligible = photo_key not in reports["cohort-amendment.json"]["localization_ineligible_photos"]
                maps = {key: load_maps(evaluation_dir, photos[key]) for key in MODEL_KEYS}
                case = {"key": photo_key, "label": audit[photo_key]["label"],
                        "group": comparison_group(photos), "localization_eligible": eligible,
                        "reconstructed_photo": photo_key == "defect/0054", "cards": [],
                        "models": {key: describe_photo(photos[key], maps[key], masks, eligible) for key in MODEL_KEYS}}
                for key in MODEL_KEYS:
                    maximum_difference = max(maximum_difference, *(r["absolute_score_difference"] for r in case["models"][key]["regions"]))
                # All 55 defects (including successes) and all normal false alarms: no cherry-picking.
                if case["label"] == 1 or case["group"] != "both_below_threshold":
                    for page in range(2):
                        name = f"cards/{photo_key.replace('/', '-')}-page{page+1}.jpg"
                        render_page(case, images, masks, maps, page).save(stage / name, format="JPEG", quality=90)
                        case["cards"].append(name)
                cases.append(case)
                if index % 10 == 0 or index == 109:
                    print(f"saved-map diagnostics {index}/109 photos (no GPU inference)", flush=True)
        report = {
            "schema_version": 1, "status": "mpi_pcb_saved_maps_analyzed_post_test_exploratory",
            "git": execution_git, "source_handoff_sha256": EVALUATION_SHA256,
            "source_archive_sha256": reports["evaluation.json"]["archive_sha256"],
            "summary": summarize(cases), "map_files_verified": len(cases) * 2,
            "region_scores_recomputed": len(cases) * 24,
            "maximum_absolute_region_score_difference": maximum_difference,
            "review_card_photos": sum(bool(c["cards"]) for c in cases),
            "review_card_files": sum(len(c["cards"]) for c in cases),
            "selection_rule": "Numerics: all 109 photos; cards: all 55 defects + all 6 normal false alarms, 12 regions each",
            "model_changed": False, "thresholds_changed": False, "inference_performed": False,
            "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
            "resolution_causality_identified": False,
            "pixel_decision_threshold_used": False, "source_attribution": SOURCE_ATTRIBUTION,
            "limitations": [
                "Post-test description, not independent final validation or a new model.",
                "256px inputs/maps cannot establish loss relative to original resolution; no upscaling or synthetic detail.",
                "Top-1% = 656 pixels per tile. GT overlap bounds retain cutoff ties, not arbitrary tie choices.",
                "No pixel pass/fail threshold. Display divides maps by 1.5 times each model's image threshold only.",
                "All location metrics exclude defect/0006 because one GT tile is unreadable; valid GT tiles remain visible on its cards.",
                "Regions overlap and are not independent photos; summary uses a mean per photo then median across photos.",
                "Small footprint, low rank, or normal-region dominance are evidence to inspect, not confirmed physical causes.",
            ],
        }
        if report["review_card_photos"] != 61 or report["review_card_files"] != 122:
            raise ValueError("Unexpected review cohort; expected 55 defects and 6 normal false alarms")
        write_json(stage / "analysis.json", report)
        write_json(stage / "cases.json", cases)
        write_json(stage / "source-evaluation.json", reports["evaluation.json"])
        links = []
        for case in cases:
            if case["cards"]:
                files = " ".join(f'<a href="{html.escape(name)}">{i+1}쪽</a>' for i, name in enumerate(case["cards"]))
                links.append(f'<li>{html.escape(case["key"])} · {html.escape(case["group"])} — {files}</li>')
        page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>MPI-PCB 원인 진단</title>'
                '<style>body{font:16px system-ui;max-width:960px;margin:40px auto;line-height:1.7}li{margin:8px 0}</style>'
                '<h1>MPI-PCB: 어디에 반응했는지 확인하기</h1>'
                '<p>기존 109장 점수는 그대로입니다. 불량 55장과 정상 오경보 6장의 모든 12영역을 두 쪽씩 표시합니다.</p>'
                '<p>왼쪽은 원본+초록 정답 테두리, 가운데는 Real-IAD, 오른쪽은 VisA 이상 맵입니다. '
                '색은 고정 표시 척도이며 픽셀 합격·불량 기준이 아닙니다. 두 모델의 색 농도로 우열을 비교하지 마세요.</p>'
                '<p>사진별 색 범위를 자동 확대하지 않습니다. 높은 값은 포화될 수 있으므로 숫자도 확인하세요. '
                'defect/0006은 정답 일부가 없어 위치 통계에서 제외합니다.</p>'
                '<p>현재 자료는 이미 256×256입니다. 작은 결함 픽셀 수와 점수 기여는 확인하지만 축소가 원인임을 증명하지는 않습니다. '
                '상위 1%는 영역당 656픽셀입니다. 상세 수치는 analysis.json과 cases.json에 있습니다.</p>'
                '<p>아래 결과는 이미 관찰한 시험의 원인 진단이며, 재학습·새 추론·임곗값 변경·자동 통과는 하지 않았습니다.</p>'
                '<ul>' + "\n".join(links) + '</ul><p>' + html.escape(SOURCE_ATTRIBUTION) + '</p></html>')
        with (stage / "index.html").open("x", encoding="utf-8") as stream:
            stream.write(page)
        publish(stage, output, handoff)
    return {"status": report["status"], "summary": report["summary"],
            "map_files_verified": report["map_files_verified"],
            "maximum_absolute_region_score_difference": maximum_difference,
            "review_card_photos": report["review_card_photos"],
            "output": str(output), "handoff": str(handoff), "handoff_sha256": sha256_file(handoff)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.project, args.archive)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
