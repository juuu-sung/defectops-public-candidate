"""Score a separately acquired, labeled batch. Acquisition provenance must establish
independence; byte or pixel checks alone cannot identify all derived photographs.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from defectops.manifest import canonical, checksum, verify_manifest

ART = ROOT / "artifacts"
TRIALS = ART / "analyses/inspection-threshold-trials"
RULE = ROOT / "reports/transistor-inspection-100/frozen-rule.json"
LABELS = {"0": 0, "good": 0, "normal": 0, "정상": 0, "1": 1, "defect": 1, "bad": 1, "불량": 1}


def pixel_hash(photo):
    rgb = photo.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + b":" + rgb.tobytes()).hexdigest()


def inventory(image_root, labels_file, manifest):
    """Reject ambiguous labels and exclude exact files and losslessly re-encoded copies."""
    verify_manifest(ROOT / "data", manifest)
    known = {}
    for record in manifest["records"]:
        if record["category"] != "transistor":
            continue
        with Image.open(ROOT / "data" / record["image_path"]) as im:
            known[pixel_hash(im)] = record["image_path"]
    seen, batch, excluded = {}, [], []
    with labels_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Labels need image_path and label columns")
        for row in reader:
            path = (image_root / row["image_path"]).resolve(strict=True)
            sample = path.relative_to(image_root).as_posix()
            label = LABELS.get(row["label"].strip().lower())
            if label is None:
                raise ValueError(f"Unknown ground truth for {sample}: {row['label']}")
            with Image.open(path) as image:
                pixels = pixel_hash(image)
                size = image.size
            if pixels in seen and seen[pixels][1] != label:
                raise ValueError(f"Conflicting ground truth for the same photo: {sample}")
            if pixels in known or pixels in seen:
                match = known[pixels] if pixels in known else seen[pixels][0]
                excluded.append({"image_path": sample, "duplicate_of": match})
            else:
                batch.append({"image_path": sample, "label": label, "source_sha256": checksum(path),
                              "rgb_sha256": pixels, "size": size})
            seen[pixels] = (sample, label)
    return batch, excluded


def api_png(path, item):
    if checksum(path) != item["source_sha256"]:
        raise ValueError(f"Photo changed after inventory: {item['image_path']}")
    with Image.open(path) as photo:
        if photo.size != (1024, 1024):
            raise ValueError(f"Native dimensions {photo.size} require an explicit input policy for this API")
        if photo.format == "PNG" and photo.mode == "RGB":
            return path.read_bytes()
        converted = io.BytesIO()
        photo.convert("RGB").save(converted, format="PNG")
        return converted.getvalue()


def rule_decision(score, raw_map, rule):
    if not math.isfinite(score) or list(raw_map.shape) != rule["map_size_hw"] or not np.isfinite(raw_map).all():
        raise ValueError("Invalid score or raw anomaly map")
    area = int(np.count_nonzero(raw_map >= rule["pixel_threshold"]))
    return score >= rule["score_threshold"] or area >= rule["area_cut"], area


def existing_inspections(rule):
    matches = {}
    for path in sorted((ART / "inspections").glob("*/inspection.json")):
        r = json.loads(path.read_bytes())
        trace = r.get("trace", {})
        if (r.get("status") == "completed" and trace.get("checkpoint_sha256") == rule["checkpoint_sha256"]
                and trace.get("evaluation_id") == rule["evaluation_id"] and r.get("environment", {}).get("device") == "cpu"):
            matches.setdefault(r["input"]["sha256"], r["inspection_id"])
    return matches


def inspect(http, png, rule, known_ids):
    digest = hashlib.sha256(png).hexdigest()
    identifier = known_ids.get(digest)
    if identifier is None:
        response = http.post("/v1/inspections", content=png, headers={"Content-Type": "image/png"})
        identifier = response.raise_for_status().json()["inspection_id"]
    if not re.fullmatch("[0-9a-f]{32}", identifier):
        raise ValueError("Invalid inspection ID")
    prefix = f"/v1/inspections/{identifier}"
    record = http.get(prefix).raise_for_status().json()
    if (record["status"] != "completed" or record["inspection_id"] != identifier or record["model"] != "patchcore"
            or record["input"]["sha256"] != digest or record["trace"]["checkpoint_sha256"] != rule["checkpoint_sha256"]
            or record["trace"]["manifest_sha256"] != rule["manifest_sha256"] or record["environment"]["device"] != "cpu"
            or record["trace"]["evaluation_id"] != rule["evaluation_id"] or record["pixel"]["threshold"] != rule["pixel_threshold"]):
        raise ValueError(f"Image or model differs from the frozen experiment: {identifier}")
    raw = http.get(prefix + "/artifacts/anomaly-map.npz").raise_for_status().content
    if hashlib.sha256(raw).hexdigest() != record["artifacts"]["anomaly_map_sha256"]:
        raise ValueError("Raw anomaly map digest differs from the inspection")
    with np.load(io.BytesIO(raw), allow_pickle=False) as result:
        alerted, area = rule_decision(record["raw_score"], result["anomaly_map"], rule)
    if area != record["pixel"]["anomalous_map_pixels"]:
        raise ValueError("Area on the native model map differs from the API")
    return identifier, record["raw_score"], area, alerted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("labels", type=Path, help="CSV with image_path and label (0/1 or normal/defect)")
    parser.add_argument("--check-only", action="store_true", help="Inventory inputs without creating inspections")
    args = parser.parse_args()
    root = args.image_root.resolve(strict=True)
    rule = json.loads(RULE.read_bytes())
    if (rule["score_threshold"], rule["area_fraction"], rule["area_cut"], rule["map_size_hw"]) != (38.70416, 0.025, 1639, [256, 256]):
        raise ValueError("Stored rule does not match the requested frozen rule")
    eval_dir = ART / "evaluations" / rule["evaluation_id"]
    manifest = json.loads((eval_dir / "manifest.json").read_bytes())
    evaluation = json.loads((eval_dir / "evaluation.json").read_bytes())
    if (manifest["manifest_sha256"] != rule["manifest_sha256"] or
            evaluation["fit"]["checkpoint_sha256"] != rule["checkpoint_sha256"] or
            evaluation["calibration_contract"]["pixel"]["threshold"] != rule["pixel_threshold"]):
        raise ValueError("Frozen model provenance differs")
    batch, excluded = inventory(root, args.labels, manifest)
    print(f"{len(batch)} distinct candidate images, {len(excluded)} exact duplicates excluded.")
    if {x["label"] for x in batch} != {0, 1}:
        parser.exit(2, "A new labeled normal and defective cohort is required; no holdout metrics were produced.\n")
    if args.check_only:
        return
    batch.sort(key=lambda r: r["image_path"])
    if any(x["size"] != (1024, 1024) for x in batch):
        raise ValueError("Some input dimensions differ from the live API contract; review preprocessing before inference")
    batch_id = hashlib.sha256(canonical([checksum(RULE), batch, excluded])).hexdigest()[:20]
    destination = TRIALS / f"transistor-holdout-{batch_id}"
    known_ids, results = existing_inspections(rule), []
    with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=120) as http:
        health = http.get("/healthz").raise_for_status().json()
        if (health["status"] != "ready" or health["device"] != "cpu" or health["model"] != "patchcore" or
                health["fit_run_id"] != evaluation["fit"]["run_id"] or health["review_registration"]["enabled"]):
            raise ValueError("Check the resident Mac model and local study registration settings")
        for row in batch:
            pid, score, area, alert = inspect(http, api_png(root / row["image_path"], row), rule, known_ids)
            outcome = ("TP" if row["label"] else "FP") if alert else ("FN" if row["label"] else "TN")
            url = f"http://127.0.0.1:8791/?mode=live&inspection={pid}#live-comparison" if outcome in {"FN", "FP"} else ""
            results.append({**row, "score": score, "flagged_pixels": area, "area_fraction": area / math.prod(rule["map_size_hw"]),
                            "trial_alert": alert, "outcome": outcome, "inspection_id": pid, "error_map_url": url})
    counts = Counter(row["outcome"] for row in results)
    summary = {"rule_sha256": checksum(RULE), "labels_sha256": checksum(args.labels), "sample_count": len(results),
               "score_threshold": rule["score_threshold"], "area_fraction": rule["area_fraction"], "area_cut": rule["area_cut"],
               "pixel_threshold": rule["pixel_threshold"], "map_size_hw": rule["map_size_hw"], "checkpoint_sha256": rule["checkpoint_sha256"],
               "device": "cpu", "evaluation_id": rule["evaluation_id"], "training_manifest_sha256": rule["manifest_sha256"],
               "counts": {key: counts[key] for key in ("TP", "FN", "TN", "FP")}, "excluded_duplicates": excluded,
               "errors": [r for r in results if r["error_map_url"]]}
    TRIALS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TRIALS) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        with (stage / "per-image.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if not destination.exists():
            stage.rename(destination)
        elif any(checksum(f) != checksum(destination / f.name) for f in stage.iterdir()):
            raise ValueError("Existing holdout result differs from this replay")
    print(summary["counts"], destination)


if __name__ == "__main__":
    main()
