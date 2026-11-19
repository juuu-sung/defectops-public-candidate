"""Evaluate one shared image threshold on the stored 100-image CPU/GPU test pair."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from defectops.evaluation_metrics import confusion_counts
from defectops.manifest import checksum, verify_manifest

ART = ROOT / "artifacts"
ANALYSES = ART / "analyses"
EVAL = ART / "evaluations/patchcore-transistor-seed42-eval-v1"
PARITY = ANALYSES / "transistor-patchcore-mac-gpu-test-20261119"
ERRORS = ANALYSES / "transistor-patchcore-test-errors-20261119"
TRIAL_ID = "transistor-100-20261119"
DEST = ANALYSES / "inspection-threshold-trials" / TRIAL_ID
DEVICES = ("cpu", "gpu")


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def keyed(records):
    by_path = {r["image_path"]: r for r in records}
    assert len(by_path) == len(records), "duplicate test image"
    return by_path


def csv_file(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def outcome(label, score, threshold):
    return ("TP" if label else "FP") if score >= threshold else ("FN" if label else "TN")


def case_url(identifier):
    return f"http://127.0.0.1:8791/?mode=live&inspection={identifier}&trial={TRIAL_ID}#live-comparison"


def main():
    evaluation = read_json(EVAL / "evaluation.json")
    manifest = read_json(EVAL / "manifest.json")
    verify_manifest(ROOT / "data", manifest)
    records = keyed([r for r in manifest["records"] if r["official_split"] == "test"])
    errors = keyed(read_csv(ERRORS / "per-image.csv"))
    paired = keyed(read_csv(PARITY / "per-image.csv"))
    parity_meta = read_json(PARITY / "summary.json")
    error_meta = read_json(ERRORS / "summary.json")
    artifacts = evaluation["artifacts"]
    for key in ("predictions", "calibration_contract", "test_maps"):
        assert checksum(EVAL / artifacts[key]) == artifacts[key + "_sha256"]
    with (EVAL / artifacts["predictions"]).open() as stream:
        gpu_records = keyed([json.loads(line) for line in stream])
    assert records.keys() == paired.keys() == errors.keys() == gpu_records.keys()
    assert len(records) == len(evaluation["test_sample_ids"]) == 100
    checkpoint = evaluation["fit"]["checkpoint_sha256"]
    manifest_hash = manifest["manifest_sha256"]
    assert checkpoint == error_meta["checkpoint_sha256"] == parity_meta["checkpoint_sha256"]
    assert manifest_hash == evaluation["manifest_sha256"] == parity_meta["manifest_sha256"] == error_meta["manifest_sha256"]
    contract = read_json(EVAL / artifacts["calibration_contract"])
    assert contract == evaluation["calibration_contract"] and contract["image"]["decision_rule"] == "score >= threshold"
    baseline, pixel = contract["image"]["threshold"], contract["pixel"]["threshold"]
    cpu_map_file = PARITY / "cpu-maps.npz"
    assert checksum(cpu_map_file) == parity_meta["cpu_maps_sha256"] == error_meta["cpu_maps_sha256"]
    with np.load(cpu_map_file, allow_pickle=False) as archive:
        paths = archive["image_paths"].tolist()
        map_stack = archive["anomaly_maps"]
    assert len(set(paths)) == len(paths) == len(records) and set(paths) == records.keys()
    assert map_stack.shape == (len(paths), 256, 256) and np.isfinite(map_stack).all()
    map_indices = {name: i for i, name in enumerate(paths)}
    names = sorted(records)
    rows = []
    for name in names:
        ref, pair, audit, pred = records[name], paired[name], errors[name], gpu_records[name]
        assert ref["image_sha256"] == pair["input_sha256"] == audit["input_sha256"]
        assert ref["label"] == int(pair["dataset_label"]) == int(audit["ground_truth_label"]) == pred["label"]
        assert ref["defect_type"] == pred["defect_type"] == audit["defect_type"] == pair["defect_type"]
        assert baseline == float(audit["image_threshold"]) == float(pair["image_threshold"]) == pred["image_threshold"]
        assert pixel == float(audit["pixel_threshold"]) == float(pair["pixel_threshold"]) == pred["pixel_threshold"]
        row = {"image_path": name, "ground_truth_label": ref["label"], "defect_type": ref["defect_type"],
               "input_sha256": ref["image_sha256"]}
        for device in DEVICES:
            key = device + "_score"
            score = float(pair[key])
            assert math.isfinite(score) and score == float(audit[key])
            if device == "gpu":
                assert score == pred["raw_score"] and pred["decision"] == audit["gpu_decision"]
            assert outcome(ref["label"], score, baseline) == audit[device + "_outcome"]
            row[key] = score
        rows.append(row)
    labels = np.array([r["ground_truth_label"] for r in rows])
    scores = {d: np.array([r[d + "_score"] for r in rows]) for d in DEVICES}
    assert int(labels.sum()) == 40


    def counts(threshold):
        return {d: confusion_counts(labels, scores[d], threshold) for d in DEVICES}


    limits = {d: float(scores[d][labels == 1].min()) for d in DEVICES}
    trial = min(limits.values())
    before, after = counts(baseline), counts(trial)
    assert all(limits[d] < baseline and counts(limits[d])[d]["fn"] == 0 and
               counts(math.nextafter(limits[d], math.inf))[d]["fn"] > 0 for d in DEVICES)
    assert all(after[d]["fn"] == 0 for d in DEVICES) and any(
        c["fn"] > 0 for c in counts(math.nextafter(trial, math.inf)).values()
    )
    changed_normals = []
    recovered = []
    for row in rows:
        row.update(baseline_threshold=baseline, trial_threshold=trial)
        for d in DEVICES:
            row[d + "_baseline"] = outcome(row["ground_truth_label"], row[d + "_score"], baseline)
            row[d + "_trial"] = outcome(row["ground_truth_label"], row[d + "_score"], trial)
        # In this paired evaluation, both devices agree for each individual photo at both cuts.
        assert row["cpu_baseline"] == row["gpu_baseline"] and row["cpu_trial"] == row["gpu_trial"]
        row["change"] = "unchanged"
        row["trial_gallery_url"] = ""
        if row["cpu_baseline"] == "TN" and row["cpu_trial"] == "FP":
            row["change"] = "new_false_alarm"
            changed_normals.append(row)
        if row["cpu_baseline"] == "FN" and row["cpu_trial"] == "TP":
            row["change"] = "recovered_defect"
            recovered.append(row)
    assert len(changed_normals) == after["cpu"]["fp"] - before["cpu"]["fp"] and len(recovered) == before["cpu"]["fn"]
    cases = []
    local = {}
    for path in sorted((ART / "inspections").glob("*/inspection.json")):
        r = read_json(path)
        if (r.get("status") == "completed" and r.get("environment", {}).get("device") == "cpu"
                and r["trace"]["checkpoint_sha256"] == checkpoint and r["trace"]["evaluation_id"] == evaluation["evaluation_id"]):
            local.setdefault((r["input"]["sha256"], r["raw_score"]), r["inspection_id"])
    with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=90.0) as http:
        health = http.get("/healthz").raise_for_status().json()
        assert health["status"] == "ready" and health["device"] == "cpu" and health["fit_run_id"] == evaluation["fit"]["run_id"]
        assert health["review_registration"]["enabled"] is False
        for row in sorted(changed_normals, key=lambda r: (-r["cpu_score"], r["image_path"])):
            name = row["image_path"]
            original = (ROOT / "data" / name).read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            assert digest == row["input_sha256"]
            identifier = local.get((digest, row["cpu_score"]))
            if identifier is None:
                r = http.post("/v1/inspections", content=original, headers={"Content-Type": "image/png"}).raise_for_status().json()
                identifier = r["inspection_id"]
            assert re.fullmatch("[a-f0-9]{32}", identifier)
            prefix = f"/v1/inspections/{identifier}"
            r = http.get(prefix).raise_for_status().json()
            assert r["inspection_id"] == identifier and r["status"] == "completed" and r["environment"]["device"] == "cpu"
            assert r["input"]["sha256"] == digest and r["trace"]["checkpoint_sha256"] == checkpoint
            assert r["trace"]["evaluation_id"] == evaluation["evaluation_id"] and r["trace"]["manifest_sha256"] == manifest_hash
            assert r["raw_score"] == row["cpu_score"] and r["decision"] == "PASS" and r["automatic_pass_allowed"] is False
            assert r["image_threshold"] == baseline and r["pixel"]["threshold"] == pixel
            for key in ("input", "heatmap", "overlay", "anomaly_map"):
                blob = http.get(f"{prefix}/artifacts/" + r["artifacts"][key]).raise_for_status().content
                assert hashlib.sha256(blob).hexdigest() == r["artifacts"][key + "_sha256"]
                if key == "input":
                    assert blob == original
                if key == "anomaly_map":
                    with np.load(io.BytesIO(blob), allow_pickle=False) as raw:
                        assert np.array_equal(raw["anomaly_map"], map_stack[map_indices[name]])
                        assert float(raw["image_threshold"]) == baseline and float(raw["pixel_threshold"]) == pixel
            cases.append({"sample": name.removeprefix("transistor/test/"), "inspection_id": identifier,
                          "ground_truth_label": 0, "input_sha256": digest, "cpu_score": row["cpu_score"], "gpu_score": row["gpu_score"]})
            row["trial_gallery_url"] = case_url(identifier)
            print("verified", cases[-1]["sample"], identifier, flush=True)
    boundaries = sorted({baseline, *(float(s) for d in DEVICES for s in scores[d])}, reverse=True)
    sweep = []
    for threshold in boundaries:
        values = counts(threshold)
        sweep.append({"threshold": threshold, **{f"{d}_{k}": v for d in DEVICES for k,v in values[d].items()}})
    assert max(r["threshold"] for r in sweep if r["cpu_fn"] == r["gpu_fn"] == 0) == trial
    summary = {
        "schema_version": 1, "trial_id": TRIAL_ID, "category": "transistor", "sample_count": len(rows),
        "fit_run_id": evaluation["fit"]["run_id"], "evaluation_id": evaluation["evaluation_id"],
        "checkpoint_sha256": checkpoint, "manifest_sha256": manifest_hash,
        "decision_rule": "score >= threshold", "selection": "highest shared threshold with zero image-level false negatives on these same test photos",
        "baseline_threshold": baseline, "trial_threshold": trial, "pixel_threshold": pixel,
        "maximum_by_device": limits, "baseline": before, "trial": after,
        "device_disagreements": {phase: sum(r["cpu_" + phase] != r["gpu_" + phase] for r in rows) for phase in ("baseline", "trial")},
        "milestones": [{"threshold": t, "counts": counts(t)} for t in sorted(
            {baseline, *(min(r["cpu_score"], r["gpu_score"]) for r in recovered)}, reverse=True)],
        "recovered_defects": [{k: r[k] for k in ("image_path", "cpu_score", "gpu_score")} for r in recovered],
        "existing_false_alarms": [r["image_path"] for r in rows if r["cpu_baseline"] == "FP"],
        "cases": cases,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DEST.parent, prefix=".prepare-") as temporary:
        staged = Path(temporary) / TRIAL_ID
        staged.mkdir()
        csv_file(staged / "per-image.csv", rows)
        csv_file(staged / "thresholds.csv", sweep)
        (staged / "trial.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if DEST.exists():
            assert all((DEST / file.name).read_bytes() == file.read_bytes() for file in staged.iterdir())
        else:
            staged.rename(DEST)
    print("saved", DEST, "milestones", summary["milestones"])


if __name__ == "__main__":
    main()
