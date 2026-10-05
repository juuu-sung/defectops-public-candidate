"""CPU-only attribution of frozen SuperSimpleNet maps; no model fitting or new inference."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from defectops.evaluation_metrics import binary_auroc
from scripts import mpi_supersimplenet_evaluation_protocol as e
from scripts import mpi_supersimplenet_fit_protocol as p
from scripts import run_mpi_supersimplenet_evaluation as evaluation

fit, ev = e.fit, e.inputs.ev
NAME = "mpi-supersimplenet-map-diagnostics-v1"
EVALUATION_SHA = "ac568747b3986d9a91c596d821e55110a43d198f7b590121829f4ad53010b3aa"
POLICY = {"cases": 20, "normal_high_per_group": 3,
          "normal_controls": "calibration threshold anchor and median; score then photo_id order",
          "top_pixels": 656, "tie_policy": "stable raster order for contributions; all cutoff ties on display",
          "display_range": [-1., 1.], "display_zero": "white", "display_negative": "blue",
          "location_excluded_photos": ["defect/0006"], "inference_performed": False,
          "training_performed": False, "threshold_changed": False, "automatic_pass_allowed": False,
          "unselected_test_image_or_mask_decode_allowed": False, "automatic_cause_assigned": False}


def selections(plan, normal):
    contract, groups = plan["contract"], plan["cohorts"]
    entries = []
    for key in e.POLICY["acceptance"]["regression_keys"]:
        i = next(i for i, s in enumerate(groups["test"]) if s["key"] == key)
        entries.append({"group": "test", "index": i, "role": "fixed_miss", "photo": groups["test"][i]})
    for group in ("development", "calibration"):
        scores = p.photo_scores(normal["tile_scores"][group], contract)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], groups[group][i]["photo_id"]))[:3]
        for i in order:
            entries.append({"group": group, "index": i, "role": "normal_high", "photo": groups[group][i]})
    group = "calibration"
    scores = p.photo_scores(normal["tile_scores"][group], contract)
    order = sorted(range(len(scores)), key=lambda i: (scores[i], groups[group][i]["photo_id"]))
    anchor = next(i for i in order if scores[i] == contract["threshold"])
    for role, i in (("normal_threshold_anchor", anchor), ("normal_median", order[len(order)//2])):
        entries.append({"group": group, "index": i, "role": role, "photo": groups[group][i]})
    if len(entries) != POLICY["cases"] or len({(s["group"], s["index"]) for s in entries}) != POLICY["cases"]:
        raise ValueError("Diagnostic selection overlaps or differs; no replacement")
    return entries


def prepare(project, handoffs, fit_root, evaluation_root):
    data = ev.checked_bundle(handoffs / (e.NAME + ".zip"), EVALUATION_SHA)
    plan = data["plan.json"]
    signature = fit.digest(plan)
    if (plan["name"] != e.NAME or plan["policy"] != e.POLICY
            or e.value(evaluation_root, "identity", signature) != plan
            or e.value(evaluation_root, "result", signature) != data["report.json"]
            or e.value(evaluation_root, "screen", signature) != data["screen.json"]):
        raise ValueError("Reviewed evaluation and local records differ")
    report = data["report.json"]
    if (report["status"] != "supersimplenet_observed_evaluation_completed" or report["cached_photos"] != 12
            or report["screen_passed"] or report["full_comparison_run"] or report["warm_evaluated"]
            or report["stability_evaluated"] or data["screen.json"]["retained"] != 0):
        raise ValueError("Expected the reviewed failed 12-case screen only")
    for name, sha in plan["sources_lf_sha256"].items():
        if hashlib.sha256((project / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Frozen evaluation source changed: {name}")
    records = e.read_fit(project, handoffs, fit_root)
    if (plan["contract"] != records["normal-results.json"]["contract"]
            or plan["fit_signature"] != fit.digest(records["plan.json"])
            or plan["checkpoint_sha256"] != records["checkpoint-receipts.json"][-1]["files"]["model.pt"]):
        raise ValueError("Final model or calibration contract changed")
    source = e.inputs.prepare(project, handoffs)
    if plan["cohorts"] != source["cohorts"] or plan["source"] != source["identity"]:
        raise ValueError("Original image/mask provenance changed")
    if e.screen(data["screen.json"]["predictions"]) != data["screen.json"]:
        raise ValueError("Screen cannot be reproduced")
    entries = selections(plan, records["normal-results.json"])
    from scripts.mpi_score_probe_protocol import EVALUATION_SHA256 as baseline_sha
    baseline = ev.checked_bundle(handoffs / "mpi-pcb-normal-bank-evaluation-v1.zip", baseline_sha)
    if fit.digest(baseline["input-lock.json"]["identity"]) != baseline["evaluation.json"]["execution_sha256"]:
        raise ValueError("Baseline identity differs")
    source["baseline_bundle"] = baseline
    baseline_root = project / "artifacts/evaluations/mpi-pcb-normal-bank-evaluation-v1"
    for entry in entries:
        entry["baseline"] = baseline_record(entry, source, baseline, baseline_root)
    paths = ["scripts/mpi_supersimplenet_map_diagnostic.py", "scripts/mpi_supersimplenet_map_cards.py",
             "scripts/review_mpi_supersimplenet_maps.py", "scripts/run_mpi_supersimplenet_map_diagnostics.ps1",
             "docs/mpi-supersimplenet-map-diagnostics-v1.md"]
    identity = {"name": NAME, "policy": POLICY, "evaluation_zip_sha256": EVALUATION_SHA,
                "fit_zip_sha256": e.FIT_SHA, "evaluation_identity": signature,
                "fit_identity": plan["fit_signature"], "checkpoint_sha256": plan["checkpoint_sha256"],
                "contract": plan["contract"], "entries": entries,
                "sources_lf_sha256": {s: hashlib.sha256((project / s).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                      for s in paths},
                "baseline_zip_sha256": baseline_sha, "numpy_version": np.__version__}
    return data, records, source, identity


def load_maps(entry, data, records, fit_root, evaluation_root):
    group, i = entry["group"], entry["index"]
    photo = entry["photo"]
    plan, contract = data["plan.json"], data["plan.json"]["contract"]
    name = f"photo-{i:03d}"
    if group == "test":
        expected = next(s for s in data["screen.json"]["predictions"] if s["key"] == photo["key"])
        actual = evaluation.read_cache(evaluation_root, group, i, fit.digest(plan), photo, contract)
        if actual != expected:
            raise ValueError("Cached test sample differs from reviewed evaluation")
        root = evaluation_root
        receipt = fit.read_phase(root / group, name, fit.digest(plan), ["predictions.npy", "sample.json"])
        if receipt != data["sample-receipts.json"][f"test/{name}/receipt.json"]:
            raise ValueError("Test map receipt differs from received ZIP")
    else:
        root = fit_root
        receipt = fit.read_phase(root / group, name, plan["fit_signature"], ["predictions.npy"])
        if receipt not in records["prediction-receipts.json"]:
            raise ValueError("Normal map receipt differs from received fit")
        expected = e.sample(photo, records["normal-results.json"]["tile_scores"][group][i], contract)
    raw = p.validate_predictions(np.load(root / group / name / "predictions.npy", allow_pickle=False))
    if raw.shape != (12, 65537):
        raise ValueError("Expected ordered maps for all 12 regions")
    replay = e.sample(photo, p.tile_scores(raw), contract)
    if any(replay[k] != expected[k] for k in replay):
        raise ValueError("Saved maps do not exactly reproduce frozen scores")
    return raw, replay, receipt


def map_stats(values, mask):
    flat = values.ravel()
    cutoff = float(np.partition(flat, -656)[-656])
    strict, tied = values > cutoff, values == cutoff
    slots = 656 - int(strict.sum())
    stats = {"minimum": float(flat.min()), "maximum": float(flat.max()), "mean": float(flat.mean()),
             "top656_mean": float(np.partition(flat, -656)[-656:].mean()), "top656_cutoff": cutoff,
             "display_below_range_fraction": float(np.mean(values < POLICY["display_range"][0])),
             "display_above_range_fraction": float(np.mean(values > POLICY["display_range"][1])),
             "support_including_ties_pixels": int((strict | tied).sum()), "location": None}
    if mask is not None:
        positives = int(mask.sum())
        lo = int((strict & mask).sum()) + max(0, slots - int((tied & ~mask).sum()))
        hi = int((strict & mask).sum()) + min(slots, int((tied & mask).sum()))
        stats["location"] = {"gt_pixels": positives, "within_tile_auroc": binary_auroc(mask, values),
            "gt_mean": float(values[mask].mean()) if positives else None,
            "background_mean": float(values[~mask].mean()) if positives < mask.size else None,
            "gt_maximum": float(values[mask].max()) if positives else None,
            "top656_gt_overlap_bounds": [lo, hi],
            "top656_gt_recall_bounds": [lo/positives, hi/positives] if positives else None}
    return stats


def describe(entry, raw, masks, sample, contract, baseline=None):
    raw = p.validate_predictions(raw)
    if raw.shape != (12, 65537) or len(masks) != 12:
        raise ValueError("Expected 12 predictions and masks")
    replay = e.sample(entry["photo"], p.tile_scores(raw), contract)
    if any(sample[k] != v for k, v in replay.items()):
        raise ValueError("Diagnostic would change classification scores")
    maps = raw[:, 1:].reshape(12, 256, 256).astype(np.float64)
    eligible = entry["photo"]["key"] not in POLICY["location_excluded_photos"]
    if baseline is not None and (baseline.shape != maps.shape or not np.isfinite(baseline).all()):
        raise ValueError("Invalid baseline map shape or values")
    rows = []
    for i, region in enumerate(fit.REGIONS):
        mask = masks[i] if eligible and entry["group"] == "test" else None
        if mask is not None and (mask.shape != (256, 256) or mask.dtype != np.bool_):
            raise ValueError("Expected native boolean mask")
        tile = sample["tile_scores"][i]
        boundary = contract["centers"][i] + contract["scales"][i] * contract["threshold"]
        stats = map_stats(maps[i], mask)
        prior = map_stats(baseline[i], mask) if baseline is not None else None
        if prior is not None:
            prior.pop("display_below_range_fraction")
            prior.pop("display_above_range_fraction")
        rows.append({"region": region, "tile_score": tile,
            "standardized_score": (tile-contract["centers"][i])/contract["scales"][i],
            "raw_equivalent_photo_boundary": boundary, "tile_margin": tile-boundary,
            "segmentation": stats, "q90_map": prior})
    return {"key": entry["photo"]["key"], "role": entry["role"], "group": entry["group"],
            "sample": sample, "threshold": contract["threshold"], "regions": rows,
            "location_eligible": eligible and entry["group"] == "test",
            "location_is_within_tile_not_dataset_metric": True,
            "baseline_available": baseline is not None,
            "classification_score_is_not_map_maximum": True,
            "recovered_image_used": any(r["member"] == ev.REPAIR_MEMBER for r in entry["photo"]["regions"]),
            "automatic_cause_assigned": False}, (maps, baseline)


def baseline_record(entry, source, baseline, root):
    if entry["group"] != "test":
        return {"available": False, "reason": "normal baseline maps not selected; no new inference"}
    ordinal = next(i+1 for i, item in enumerate(baseline["scores.json"]) if item["key"] == entry["photo"]["key"])
    name = f"photo-{ordinal:03d}"
    directory = root / "samples" / name
    if not directory.exists():
        return {"available": False, "reason": "saved baseline folder absent; no new inference"}
    # A partial/corrupt folder is an error, never silently treated as absent.
    photo = next(p for p in source["previous"]["test-audit.json"]["photos"] if p["key"] == entry["photo"]["key"])
    if photo != entry["photo"]:
        raise ValueError("Baseline and candidate photo provenance differ")
    before, receipt = ev.read_sample(root / "samples", name, fit.digest(baseline["input-lock.json"]["identity"]), photo)
    if before["photo"] != baseline["scores.json"][ordinal-1]:
        raise ValueError("Baseline sample differs from pinned handoff")
    reference = next(p for p in source["reference"]["test"] if p["key"] == photo["key"])
    if any(abs(a["score"]-b["top1"]) > 1e-4 for a,b in zip(before["photo"]["regions"],reference["regions"],strict=True)):
        raise ValueError("Stored baseline maps belong to another scoring model")
    return {"available": True, "phase": name, "receipt": receipt}


def load_baseline(entry, source, root):
    expected = entry.get("baseline", {"available": False})
    if not expected["available"]:
        return None
    actual = baseline_record(entry, source, source["baseline_bundle"], root)
    if actual != expected:
        raise ValueError("Baseline receipt changed")
    raw = (root / "samples" / actual["phase"] / "maps.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != actual["receipt"]["files"]["maps.npz"]:
        raise ValueError("Baseline map changed after receipt verification")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return z["anomaly_maps"].copy()
