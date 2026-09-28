"""Read-only, post-test score analysis. Does not update a model or service gate.

Run from any directory with the repository's Python/NumPy environment.
Requires the previously received, Git-ignored score handoff on this machine.
Writes only the adjacent, dated analysis JSON.
"""

import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/analyses/mpi-pcb-score-probe-v1-received-5784bf5d"
EXPECTED_RAW_SHA = "6471e598103e63c17259110907647cc9f976d8488b2c2c9493322baffa93c9cd"
SPECS = [("top1", "none"), ("top1", "mad"), ("native_raw", "none"),
         ("native_raw", "mad"), ("top1", "q90"), ("top1", "q95"), ("top1", "std")]


def parameters(dev, scale):
    center = np.median(dev, axis=0)
    if scale == "none":
        return np.zeros(12), np.ones(12)
    if scale == "mad":
        width = np.median(np.abs(dev - center), axis=0)
    elif scale in {"q90", "q95"}:
        width = np.quantile(dev, float(scale[1:]) / 100, axis=0, method="higher") - center
    else:
        center, width = dev.mean(axis=0), dev.std(axis=0, ddof=1)
    assert np.isfinite(width).all() and (width > 1e-6).all()
    return center, width


def quantify(values, labels, threshold):
    pred = values >= threshold
    pos, neg = values[labels == 1], values[labels == 0]
    return {"tp": int((pred & (labels == 1)).sum()),
            "fn": int((~pred & (labels == 1)).sum()),
            "fp": int((pred & (labels == 0)).sum()),
            "tn": int((~pred & (labels == 0)).sum()),
            "auroc": float(((pos[:, None] > neg) + .5 * (pos[:, None] == neg)).mean())}


def higher(values, q):
    return sorted(values)[math.ceil(q * (len(values) - 1))]


def main():
    manifest = json.loads((SOURCE / "handoff-manifest.json").read_text())
    hashes = {}
    for name in ["raw-scores.json", "scores.json", "calibration.json"]:
        hashes[name] = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert hashes[name] == manifest[name]
    assert hashes["raw-scores.json"] == EXPECTED_RAW_SHA
    raw = json.loads((SOURCE / "raw-scores.json").read_text())
    old_scores = json.loads((SOURCE / "scores.json").read_text())
    regions = [r["region"] for r in raw["development"][0]["regions"]]
    for split, count in [("development", 128), ("calibration", 169), ("test", 109)]:
        assert len(raw[split]) == count
        assert len({p["key"] for p in raw[split]}) == count
        assert all([r["region"] for r in p["regions"]] == regions for p in raw[split])
        if split != "test":
            assert all(p["label"] == 0 and p["key"].startswith(split + "/") for p in raw[split])
    labels = np.array([p["label"] for p in raw["test"]])
    assert (labels == 0).sum() == 54 and (labels == 1).sum() == 55
    results, cache = {}, {}
    for field, scale in SPECS:
        arrays = {s: np.array([[r[field] for r in p["regions"]] for p in rows])
                  for s, rows in raw.items()}
        assert all(np.isfinite(a).all() for a in arrays.values())
        center, width = parameters(arrays["development"], scale)
        z = {s: (a - center) / width for s, a in arrays.items()}
        scores = {s: a.max(axis=1) for s, a in z.items()}
        threshold = float(np.quantile(scores["calibration"], .95, method="higher"))
        key = field + "_" + scale
        record = quantify(scores["test"], labels, threshold)
        record.update(threshold=threshold, center=center.tolist(), width=width.tolist(),
                      raw_region_boundaries=(center + width * threshold).tolist(),
                      calibration_alerts=int((scores["calibration"] >= threshold).sum()),
                      calibration_boundary_keys=[p["key"] for p, s in zip(raw["calibration"], scores["calibration"])
                                                 if s == threshold],
                      false_negatives=[p["key"] for p, s in zip(raw["test"], scores["test"])
                                       if p["label"] and s < threshold],
                      false_positives=[p["key"] for p, s in zip(raw["test"], scores["test"])
                                       if not p["label"] and s >= threshold])
        results[key] = record
        cache[key] = (arrays, z, scores)
    for key, previous in [("top1_none", "baseline_top1"), ("top1_mad", "a_region_mad_top1"),
                          ("native_raw_none", "b_native_raw")]:
        rows = old_scores[previous]
        assert [p["key"] for p in rows] == [p["key"] for p in raw["test"]]
        assert np.allclose(cache[key][2]["test"], [r["score"] for r in rows], rtol=0, atol=1e-12)
        assert all(math.isclose(results[key]["threshold"], p["threshold"], abs_tol=1e-12) for p in rows)

    # Recompute the highlighted candidate using independent standard-library arithmetic.
    center, width = [], []
    for j in range(12):
        v = [p["regions"][j]["top1"] for p in raw["development"]]
        med = statistics.median(v)
        center.append(med)
        width.append(higher(v, .9) - med)
    independent = {s: [max((r["top1"] - center[j]) / width[j]
                           for j, r in enumerate(p["regions"])) for p in rows]
                   for s, rows in raw.items()}
    assert math.isclose(higher(independent["calibration"], .95), results["top1_q90"]["threshold"], abs_tol=1e-12)
    assert np.allclose(independent["test"], cache["top1_q90"][2]["test"], rtol=0, atol=1e-12)

    # Same resampled PHOTO rows for both methods; retain their 12-region dependence.
    # These are normal-pool perturbations on a FIXED, REUSED test, not generalization CIs.
    rng = np.random.default_rng(20260929)
    samples = {k: [] for k in ["top1_mad", "top1_q90"]}
    arrays = cache["top1_q90"][0]
    for _ in range(1000):
        di = rng.integers(0, 128, 128)
        ci = rng.integers(0, 169, 169)
        for key, scale in [("top1_mad", "mad"), ("top1_q90", "q90")]:
            m, w = parameters(arrays["development"][di], scale)
            cal = ((arrays["calibration"][ci] - m) / w).max(axis=1)
            t = float(np.quantile(cal, .95, method="higher"))
            values = ((arrays["test"] - m) / w).max(axis=1)
            metric = quantify(values, labels, t)
            samples[key].append([metric["tp"], metric["fp"]])
    sensitivity = {k: {"tp_fp_percentiles_2_5_10_50_90_97_5": np.quantile(v, [.025, .1, .5, .9, .975], axis=0).tolist()}
                   for k, v in samples.items()}
    separate_sensitivity = {}
    for varied in ["development", "calibration"]:
        trials = []
        rng = np.random.default_rng(20260929)
        for _ in range(1000):
            dev, cal = arrays["development"], arrays["calibration"]
            if varied == "development":
                dev = dev[rng.integers(0, len(dev), len(dev))]
            else:
                cal = cal[rng.integers(0, len(cal), len(cal))]
            m, w = parameters(dev, "q90")
            t = float(np.quantile(((cal - m) / w).max(axis=1), .95, method="higher"))
            values = ((arrays["test"] - m) / w).max(axis=1)
            v = quantify(values, labels, t)
            trials.append([v["tp"], v["fp"]])
        separate_sensitivity[varied + "_only"] = np.quantile(trials, [.025, .1, .5, .9, .975], axis=0).tolist()

    oracle = {}
    for key in ["top1_mad", "top1_q90"]:
        values = cache[key][2]["test"]
        candidates = [quantify(values, labels, t) for t in [math.inf, *np.unique(values)]]
        oracle[key] = [{"fp_budget": b, **max((v for v in candidates if v["fp"] <= b),
                                              key=lambda v: (v["tp"], -v["fp"]))}
                       for b in [0, 1, 2, 3, 5]]
    q = results["top1_q90"]
    q_scores = cache["top1_q90"][2]["test"]
    cal_values = cache["top1_q90"][2]["calibration"]
    rank = math.ceil((len(cal_values) + 1) * .95)
    rank_threshold = float(sorted(cal_values)[rank - 1])
    rank_diagnostic = {"status": "post_test_diagnostic_not_a_replacement_gate",
                       "rank_one_based": rank, "threshold": rank_threshold, "comparison": ">",
                       "tp": int(((q_scores > rank_threshold) & (labels == 1)).sum()),
                       "fp": int(((q_scores > rank_threshold) & (labels == 0)).sum()),
                       "limit": "Any split-conformal coverage claim requires a fixed scorer and exchangeable normal calibration/future data; not established here."}
    base = cache["top1_mad"][2]["test"] >= results["top1_mad"]["threshold"]
    paired = {"gained_tp": int(((q_scores >= q["threshold"]) & ~base & (labels == 1)).sum()),
              "lost_tp": int(((q_scores < q["threshold"]) & base & (labels == 1)).sum())}
    test_rows = [{"key": p["key"], "label": p["label"], "score": float(s),
                  "alert": bool(s >= q["threshold"]),
                  "winning_region": regions[int(cache["top1_q90"][1]["test"][i].argmax())]}
                 for i, (p, s) in enumerate(zip(raw["test"], q_scores))]
    localization = {"status": "optional_source_zip_not_available"}
    loc_zip = Path.home() / "Downloads/mpi-pcb-normal-bank-evaluation-v1.zip"
    if loc_zip.exists():
        loc_sha = hashlib.sha256(loc_zip.read_bytes()).hexdigest()
        assert loc_sha == "070a15bfed987802201868dee759fc5c24445bf12c6c6d3643feb3b25798e857"
        with zipfile.ZipFile(loc_zip) as archive:
            loc_manifest = json.loads(archive.read("handoff-manifest.json"))
            payload = archive.read("cases.json")
            assert hashlib.sha256(payload).hexdigest() == loc_manifest["cases.json"]
            cases = {p["key"]: p for p in json.loads(payload)}
        items = []
        for p in test_rows:
            case = cases[p["key"]]
            if not p["label"] or not case["localization_eligible"]:
                continue
            r = next(r for r in case["models"]["mpi_pcb_target_v1"]["regions"]
                     if r["region"] == p["winning_region"])
            items.append({"key": p["key"], "alert": p["alert"], "region": p["winning_region"],
                          "gt_pixels": r["gt"]["pixels"],
                          "top1_gt_overlap_min": r["gt"]["top1pct_overlap_pixels_min"],
                          "top1_gt_recall_min": r["gt"]["top1pct_gt_recall_min"],
                          "top1_gt_score_share_min": r["gt"]["top1pct_score_share_min"]})
        localization = {"status": "received_statistics_only_no_raw_map_recomputation",
                        "source_zip_sha256": loc_sha, "eligible_photos": len(items),
                        "winner_contains_gt": sum(p["gt_pixels"] > 0 for p in items),
                        "winner_top1_overlaps_gt": sum(p["top1_gt_overlap_min"] > 0 for p in items),
                        "grid3_2_winners": sum(p["region"] == "grid3-2_s1024" for p in items),
                        "missed_photos": [p for p in items if not p["alert"]],
                        "winner_without_gt": [p for p in items if not p["gt_pixels"]]}
    result = {"date": "2026-09-29", "status": "post_test_exploratory_analysis_only",
              "source_payload_sha256": hashes,
              "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu_inference_performed": False, "model_or_service_gate_modified": False,
              "test_previously_observed": True, "candidate_selection_used_test_results": True,
              "normalization_fit": "128 normal development photos only",
              "threshold_fit": "169 distinct normal calibration photos; 95% higher quantile; >= comparison",
              "sensitivity_scope": "1000 photo-row normal-pool bootstrap perturbations, fixed reused test. Not independent-board or deployment confidence intervals.",
              "regions": regions, "results": results, "paired_q90_vs_mad": paired,
              "q90_finite_sample_rank_diagnostic": rank_diagnostic,
              "normal_pool_sensitivity": sensitivity,
              "q90_separate_sensitivity_percentiles_2_5_10_50_90_97_5": separate_sensitivity,
              "oracle_sweep_not_deployable": oracle,
              "q90_test_predictions": test_rows, "independent_stdlib_replay_passed": True}
    result["q90_localization_check"] = localization
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"results": {k: {m: v[m] for m in ["tp", "fn", "fp", "tn", "auroc"]}
                                  for k, v in results.items()}, "paired": paired,
                      "sensitivity": sensitivity, "separate_sensitivity": separate_sensitivity,
                      "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
