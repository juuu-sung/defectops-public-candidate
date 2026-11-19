"""Replay the paired transistor test with image-score and spatial-area rules."""

import csv
import json
import math
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from defectops.evaluation_metrics import confusion_counts
from defectops.manifest import checksum, verify_manifest

ART = ROOT / "artifacts"
ANALYSES = ART / "analyses"
TRIALS = ANALYSES / "inspection-threshold-trials"
TRIAL_ID = "transistor-area-100-20261119"
SCORE = 38.70416
FRACTION = 0.025
DEVICES = ("cpu", "gpu")
QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")


def read_json(path):
    return json.loads(path.read_bytes())


def by_path(records):
    rows = list(records)
    result = {r["image_path"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate image path")
    return result


def read_table(path):
    with path.open(newline="") as stream:
        return by_path(csv.DictReader(stream))


def ordered_maps(path, names, expected_hash):
    if checksum(path) != expected_hash:
        raise ValueError(f"Map archive checksum differs: {path}")
    with np.load(path, allow_pickle=False) as contents:
        keys = contents["image_paths"].tolist()
        order = {name: i for i, name in enumerate(keys)}
        if len(keys) != len(order) or set(order) != set(names):
            raise ValueError("Test cohort differs from map archive")
        maps = contents["anomaly_maps"][[order[name] for name in names]]
    if maps.shape != (len(names), 256, 256) or not np.isfinite(maps).all():
        raise ValueError("Invalid transistor anomaly maps")
    return maps


def make_overlay(original_path, mask, output_path):
    with Image.open(original_path) as photo:
        photo = photo.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    pixels = Image.fromarray(mask.astype(np.uint8) * 255).resize(photo.size, Image.Resampling.NEAREST)
    tinted = Image.composite(Image.new("RGB", photo.size, "#f02b90"), photo, pixels.point(lambda x: x // 2))
    edge = ImageChops.subtract(pixels, pixels.filter(ImageFilter.MinFilter(5)))
    tinted.paste("#ff98ce", mask=edge)
    draw = ImageDraw.Draw(tinted)
    for y in range(0, photo.height, 16):
        draw.line((255, y, 255, y + 7), fill="#433a40", width=1)
        draw.line((256, y, 256, y + 7), fill="#fff3f7", width=1)
    tinted.save(output_path)


def review_case(row, identifier, raw_map, directory, evaluation, pixel_threshold):
    if not re.fullmatch("[a-f0-9]{32}", identifier):
        raise ValueError("Invalid inspection identifier")
    source = ART / "inspections" / identifier
    record = read_json(source / "inspection.json")
    assets = record["artifacts"]
    if not (record["inspection_id"] == identifier and record["status"] == "completed" and
            record["environment"]["device"] == "cpu" and record["raw_score"] == row["cpu_score"] and
            record["trace"]["checkpoint_sha256"] == evaluation["fit"]["checkpoint_sha256"] and
            record["trace"]["evaluation_id"] == evaluation["evaluation_id"] and
            record["trace"]["manifest_sha256"] == evaluation["manifest_sha256"] and
            record["input"]["sha256"] == row["input_sha256"] == checksum(source / "input.png") and
            record["pixel"]["threshold"] == pixel_threshold and record["pixel"]["anomalous_map_pixels"] == row["cpu_area"] and
            record["image_threshold"] == evaluation["calibration_contract"]["image"]["threshold"] and
            record["decision"] == "PASS" and assets["heatmap_sha256"] == checksum(source / "heatmap.png") and
            assets["anomaly_map_sha256"] == checksum(source / "anomaly-map.npz")):
        raise ValueError(f"Inspection evidence differs for {row['image_path']}")
    with np.load(source / "anomaly-map.npz", allow_pickle=False) as inspection_map:
        if not np.array_equal(inspection_map["anomaly_map"], raw_map):
            raise ValueError("Served inspection differs from batch map")
    preview = directory / f"{identifier}.png"
    make_overlay(source / "input.png", raw_map >= pixel_threshold, preview)
    return {
        "sample": row["image_path"].removeprefix("transistor/test/"), "inspection_id": identifier,
        **{key: row[key] for key in ("input_sha256", "ground_truth_label", "cpu_score", "gpu_score",
                                     "cpu_area", "gpu_area", "cpu_right", "gpu_right")},
        "area_sha256": checksum(preview), "heatmap_sha256": assets["heatmap_sha256"],
    }


def analyze(staged):
    previous = read_json(TRIALS / "transistor-100-20261119/trial.json")
    eval_dir = ART / "evaluations" / previous["evaluation_id"]
    evaluation = read_json(eval_dir / "evaluation.json")
    manifest = read_json(eval_dir / "manifest.json")
    verify_manifest(ROOT / "data", manifest)
    test = by_path(r for r in manifest["records"] if r["category"] == "transistor" and r["official_split"] == "test")
    names = sorted(test)
    labels = np.array([test[n]["label"] for n in names], dtype=bool)
    parity_dir = ANALYSES / "transistor-patchcore-mac-gpu-test-20261119"
    parity = read_json(parity_dir / "summary.json")
    pairs = read_table(parity_dir / "per-image.csv")
    error_table = read_table(ANALYSES / "transistor-patchcore-test-errors-20261119/per-image.csv")
    assets = evaluation["artifacts"]
    pixel = evaluation["calibration_contract"]["pixel"]["threshold"]
    low = previous["trial_threshold"]
    assert pixel == previous["pixel_threshold"] == parity["pixel_threshold"]
    assert evaluation["fit"]["checkpoint_sha256"] == previous["checkpoint_sha256"] == parity["checkpoint_sha256"]
    assert evaluation["manifest_sha256"] == manifest["manifest_sha256"] == parity["manifest_sha256"] == previous["manifest_sha256"]
    assert checksum(eval_dir / assets["predictions"]) == assets["predictions_sha256"]
    with (eval_dir / assets["predictions"]).open() as stream:
        gpu = by_path(json.loads(line) for line in stream)
    assert set(names) == gpu.keys() == pairs.keys() == error_table.keys() and len(names) == 100 and labels.sum() == 40
    rows = [{"image_path": n, "input_sha256": test[n]["image_sha256"], "ground_truth_label": int(test[n]["label"]),
             "defect_type": test[n]["defect_type"]} for n in names]
    for row in rows:
        n = row["image_path"]
        assert row["input_sha256"] == pairs[n]["input_sha256"] == error_table[n]["input_sha256"]
        assert row["ground_truth_label"] == gpu[n]["label"] == int(pairs[n]["dataset_label"])
        assert float(pairs[n]["gpu_score"]) == gpu[n]["raw_score"] and gpu[n]["pixel_threshold"] == pixel
    map_sources = {"cpu": (parity_dir / "cpu-maps.npz", parity["cpu_maps_sha256"]),
                   "gpu": (eval_dir / assets["test_maps"], assets["test_maps_sha256"])}
    counts = {}
    bounds = {"area": [], "right": []}
    cpu_maps = None
    for d in DEVICES:
        map_path, digest = map_sources[d]
        raw = ordered_maps(map_path, names, digest)
        if d == "cpu":
            cpu_maps = raw
        masks = raw >= pixel
        height, width = masks.shape[1:]
        area = masks.sum(axis=(1, 2))
        right = masks[:, :, width // 2:].sum(axis=(1, 2))
        quadrant_size = height // 2, width // 2
        quadrants = masks.reshape(len(names), 2, quadrant_size[0], 2, quadrant_size[1]).sum(axis=(2, 4)).reshape(len(names), 4)
        scores = np.array([float(pairs[n][d + "_score"]) for n in names])
        assert np.isfinite(scores).all() and np.all(area == quadrants.sum(axis=1)) and np.all(right == quadrants[:, 1::2].sum(axis=1))
        assert area.tolist() == [int(pairs[n][d + "_flagged_pixels"]) for n in names]
        if d == "gpu":
            assert area.tolist() == [gpu[n]["anomalous_map_pixels"] for n in names]
        high_alert = scores >= SCORE
        area_cut = math.ceil(height * width * FRACTION)
        right_cut = math.ceil(height * (width // 2) * FRACTION)
        predictions = {"score": high_alert, "lower_score": scores >= low,
                       "score_or_area": high_alert | (area >= area_cut),
                       "score_or_right": high_alert | (right >= right_cut)}
        for method, alerts in predictions.items():
            counts.setdefault(method, {})[d] = confusion_counts(labels, alerts.astype(int), 1)
        for feature, values in (("area", area), ("right", right)):
            # Inclusive integer thresholds that separate all remaining normals and defects.
            bounds[feature].append((int(values[~high_alert & ~labels].max()) + 1, int(values[~high_alert & labels].min())))
        for i, row in enumerate(rows):
            row[d + "_score"] = float(scores[i])
            row[d + "_area"] = int(area[i])
            row[d + "_right"] = int(right[i])
            row.update({f"{d}_{k}": int(v) for k, v in zip(QUADRANTS, quadrants[i])})
            for method, alerts in predictions.items():
                row[f"{d}_{method}_outcome"] = ("TP" if labels[i] else "FP") if alerts[i] else ("FN" if labels[i] else "TN")
    disagreements = {k: [r["image_path"] for r in rows if r[f"cpu_{k}_outcome"] != r[f"gpu_{k}_outcome"]] for k in counts}
    if any(disagreements.values()):
        raise ValueError("CPU/GPU rule decisions differ; report device-specific cohorts")
    anchors = [r for r in rows if r["cpu_score_outcome"] == "FN" and r["cpu_lower_score_outcome"] == "TP"]
    normals = [r for r in rows if r["cpu_score_outcome"] == "TN" and r["cpu_lower_score_outcome"] == "FP"]
    if len(anchors) != 1 or len(normals) != 14:
        raise ValueError("Input cohort differs from the requested missed defect and fourteen normals")
    normal_ids = {item["sample"]: item["inspection_id"] for item in previous["cases"]}
    anchor_row, = anchors
    paths = {n: i for i, n in enumerate(names)}
    images = staged / "areas"
    images.mkdir()

    def case(row, identifier):
        return review_case(row, identifier, cpu_maps[paths[row["image_path"]]], images, evaluation, pixel)

    for r in rows:
        r["comparison_url"] = ""
    anchor = case(anchor_row, error_table[anchor_row["image_path"]]["inspection_id"])
    cases = []
    for r in sorted(normals, key=lambda x: (-x["cpu_area"], x["image_path"])):
        photo = case(r, normal_ids[r["image_path"].removeprefix("transistor/test/")])
        cases.append(photo)
        r["comparison_url"] = f"http://127.0.0.1:8791/?mode=live&inspection={photo['inspection_id']}&area={TRIAL_ID}#live-comparison"
    anchor_row["comparison_url"] = next(r["comparison_url"] for r in normals if r["image_path"].endswith("/" + cases[0]["sample"]))
    with (staged / "per-image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1, "type": "map_area", "trial_id": TRIAL_ID,
        "evaluation_id": evaluation["evaluation_id"], "checkpoint_sha256": evaluation["fit"]["checkpoint_sha256"],
        "manifest_sha256": manifest["manifest_sha256"], "map_sha256": {k: v[1] for k, v in map_sources.items()},
        "sample_count": len(rows), "map_size_hw": [height, width], "pixel_threshold": pixel,
        "model_image_threshold": previous["baseline_threshold"], "score_threshold": SCORE, "lower_score_threshold": low,
        "area_fraction": FRACTION, "area_cut": area_cut, "right_area_cut": right_cut,
        "joint_inclusive_cut_range": {k: [max(p[0] for p in pairs), min(p[1] for p in pairs)] for k, pairs in bounds.items()},
        "rule": "score >= score_threshold OR count(raw_map >= pixel_threshold) >= area_cut",
        "right_rule": "score >= score_threshold OR count(raw_map[:, width//2:] >= pixel_threshold) >= right_area_cut",
        "selection": "Exploratory rule chosen and assessed on the same 100 test images; ground-truth masks are not rule inputs.",
        "counts": counts, "device_disagreements": disagreements, "anchor": anchor, "cases": cases,
        "remaining_false_alarms": [r["image_path"] for r in rows if r["cpu_score_or_area_outcome"] == "FP"],
    }
    (staged / "trial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    destination = TRIALS / TRIAL_ID
    with TemporaryDirectory(dir=TRIALS, prefix=".area-") as temporary:
        staged = Path(temporary) / TRIAL_ID
        staged.mkdir()
        report = analyze(staged)
        if destination.exists():
            for file in (f for f in staged.rglob("*") if f.is_file()):
                if checksum(file) != checksum(destination / file.relative_to(staged)):
                    raise ValueError("Existing result differs from this replay")
        else:
            staged.rename(destination)
    print(json.dumps(report["counts"], indent=2))
    print(destination)


if __name__ == "__main__":
    main()
