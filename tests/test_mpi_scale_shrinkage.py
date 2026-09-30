import copy
import json
import math
from pathlib import Path
import statistics
import zipfile

import numpy as np
import pytest

from scripts import run_mpi_scale_shrinkage as p


def fixture_data():
    rng = np.random.default_rng(19)
    sizes = {"development": 16, "calibration": 19, "test": 109, "validation": 8}
    raw = {}
    for g, n in sizes.items():
        raw[g] = []
        for i in range(n):
            label = int(g == "test" and i >= 54)
            pid = f"{g}-{i}"
            key = f"{'defect' if label else 'good'}/{pid}" if g == "test" else f"{g}/{pid}"
            raw[g].append({"key": key, "photo_id": pid, "label": label,
                "regions": [{"region": r, "top1": float(10 + rng.uniform(0, 5) + label * 7)} for r in p.REGIONS]})
    arrays = p.matrices(raw, sizes)
    m = p.fit_method(arrays["development"], arrays["calibration"], 0)
    stored = {"field": "top1", "threshold": m["threshold"], "regions": [
        {"region": r, "center": c, "scale": s} for r, c, s in zip(p.REGIONS, m["center"], m["scale"])]}
    preds = p.predictions(raw, arrays, m)
    c = p.counts(arrays, np.array([r["label"] for r in raw["test"]]), m)
    reference = {"observed_test": {"control_q90": {
        "counts": {k: c[k] for k in ("tp", "fn", "fp", "tn")},
        "false_negatives": [r["key"] for r in preds["test"] if r["label"] and not r["alert"]],
        "false_positives": [r["key"] for r in preds["test"] if not r["label"] and r["alert"]]}},
        "observed_normals": {"control_q90": {"alert_keys": [r["key"] for r in preds["validation"] if r["alert"]]}}}
    return raw, arrays, stored, reference


def test_fixed_config_rejects_variant(tmp_path):
    cfg = p.load_config()
    assert cfg["pool_weight"] == .5
    cfg["pool_weight"] = .25
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="configuration changed"):
        p.load_config(path)


def test_hand_calculation_and_threshold_ties():
    dev = np.outer(np.arange(10), np.arange(1, 13)).astype(float)
    cal = np.outer(np.arange(20), np.arange(1, 13)).astype(float)
    got = p.fit_method(dev, cal, .5)
    centers = [statistics.median(v) for v in dev.T.tolist()]
    widths = [sorted(v)[math.ceil(.9 * (len(v)-1))] - c for v, c in zip(dev.T.tolist(), centers)]
    pool = statistics.median(widths)
    mixed = [.5*w + .5*pool for w in widths]
    scores = [max((v-c)/s for v,c,s in zip(row,centers,mixed)) for row in cal.tolist()]
    assert got["center"] == centers and got["scale"] == mixed
    assert got["threshold"] == sorted(scores)[math.ceil(.95*(len(scores)-1))]
    assert np.any(p.normalized(cal, got).max(1) >= got["threshold"])
    equal = np.tile(np.arange(10)[:, None], (1, 12))
    assert p.fit_method(equal, equal, 0) == p.fit_method(equal, equal, .5)


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_invalid_scale_or_nonfinite_stops(value):
    dev = np.tile(np.arange(10.)[:, None], (1, 12))
    dev[:, 3] = value
    with pytest.raises(ValueError):
        p.fit_method(dev, dev, .5)


def test_role_order_count_and_finite_validation():
    raw, *_ = fixture_data()
    sizes = {g: len(v) for g, v in raw.items()}
    for change in ("order", "label", "duplicate", "nan", "overlap", "count"):
        x = copy.deepcopy(raw)
        if change == "order":
            x["development"][0]["regions"].reverse()
        elif change == "label":
            x["calibration"][0]["label"] = 1
        elif change == "duplicate":
            x["test"][1] = x["test"][0]
        elif change == "nan":
            x["validation"][0]["regions"][0]["top1"] = float("nan")
        elif change == "overlap":
            x["calibration"][0]["photo_id"] = x["development"][0]["photo_id"]
            x["calibration"][0]["key"] = "calibration/" + x["development"][0]["photo_id"]
        else:
            x["test"].pop()
        with pytest.raises(ValueError):
            p.matrices(x, sizes)


def test_labels_cannot_affect_normal_fit_and_region_permutation_equivariant():
    raw, arrays, *_ = fixture_data()
    before = p.fit_method(arrays["development"], arrays["calibration"], .5)
    for r in raw["test"]:
        r["label"] = 1-r["label"]
    assert p.fit_method(arrays["development"], arrays["calibration"], .5) == before
    perm = np.arange(11, -1, -1)
    permuted = p.fit_method(arrays["development"][:, perm], arrays["calibration"][:, perm], .5)
    np.testing.assert_array_equal(permuted["scale"], np.array(before["scale"])[perm])
    assert permuted["threshold"] == before["threshold"]


def test_replay_and_tampered_source():
    raw, arrays, stored, ref = fixture_data()
    _, result = p.replay(arrays, raw, stored, ref, p.load_config())
    assert result["maximum_absolute_difference"] == 0
    stored["threshold"] += .01
    with pytest.raises(ValueError):
        p.replay(arrays, raw, stored, ref, p.load_config())


def test_zip_identity_and_manifest(tmp_path):
    raw, _, stored, ref = fixture_data()
    payloads = {"raw-scores.json": raw, "control.json": {"candidate_top1_q90": stored}, "report.json": {"comparison": ref}}
    blobs = {n: p.canonical(v) for n, v in payloads.items()}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, blob in blobs.items():
            z.writestr(name, blob)
        z.writestr("handoff-manifest.json", p.canonical({n: p.sha(b) for n,b in blobs.items()}))
    cfg = p.load_config()
    cfg.update(source_zip_sha256=p.sha(archive.read_bytes()),
        source_raw_scores_sha256=p.sha(blobs["raw-scores.json"]),
        source_control_sha256=p.sha(blobs["control.json"]), cohort_counts={g:len(v) for g,v in raw.items()})
    assert len(p.load_source(archive, cfg)[0]["test"]) == 109
    cfg["source_raw_scores_sha256"] = "bad"
    with pytest.raises(ValueError, match="Pinned payload"):
        p.load_source(archive, cfg)
    cfg["source_zip_sha256"] = "bad"
    with pytest.raises(ValueError, match="ZIP identity"):
        p.load_source(archive, cfg)


def test_rng_pairing_draw_order_and_invalid_not_dropped():
    b = p.load_config()["bootstrap"]
    i, di, ci = next(p.trial_indices("both", b, 16, 19))
    rng = np.random.Generator(np.random.PCG64(b["scope_seeds"]["both"]))
    np.testing.assert_array_equal(di, rng.integers(0, 16, 16))
    np.testing.assert_array_equal(ci, rng.integers(0, 19, 19))
    _, arrays, *_ = fixture_data()
    labels = np.array([0]*54 + [1]*55)
    a = p.trial(arrays, labels, di, ci)
    assert a == p.trial(arrays, labels, di, ci)
    for name, w in (("control", 0), ("candidate", .5)):
        assert a["methods"][name]["parameters"] == p.fit_method(arrays["development"][di], arrays["calibration"][ci], w)
    bad = p.trial(arrays, labels, np.zeros(16, dtype=int), ci)
    summary = p.summarize_trials([a, bad], p.load_config()["development_screening"])
    assert not summary["passed"] and summary["methods"]["candidate"]["quantiles"] is None


def test_stability_gate_not_low_constant_performance():
    def record(tp):
        return {"methods": {n:{"valid": True, "counts": {"tp":tp, "fp":0, "fp_validation":0}} for n in ("control", "candidate")}}
    gate = p.load_config()["development_screening"]
    assert not p.summarize_trials([record(3)]*4, gate)["passed"]
    assert p.summarize_trials([record(53)]*4, gate)["passed"]


def test_sealed_publish_and_corruption(tmp_path):
    path = tmp_path / "a.json"
    p.publish(path, "id", {"value": 1})
    original = path.read_bytes()
    p.publish(path, "id", {"value": 1})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        p.publish(path, "other", {"value": 1})
    with pytest.raises(ValueError):
        p.publish(path, "id", {"value": 2})
    obj = json.loads(original); obj["payload"]["value"] = 2
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        p.read_sealed(path, "id")


def test_budget_enforced(monkeypatch):
    monkeypatch.setattr(p, "peak_rss_bytes", lambda: 3)
    with pytest.raises(RuntimeError, match="budget"):
        p.check_budget(p.time.monotonic(), 0, {"maximum_elapsed_seconds":10, "maximum_process_peak_rss_bytes":2})
    with pytest.raises(RuntimeError, match="budget"):
        p.check_budget(p.time.monotonic(), 11, {"maximum_elapsed_seconds":10, "maximum_process_peak_rss_bytes":20})


def test_execute_interruption_resume_and_completed_replay(tmp_path, monkeypatch):
    cfg = p.load_config(); cfg["bootstrap"]["trials_per_scope"] = 4
    monkeypatch.setattr(p, "load_config", lambda: cfg)
    monkeypatch.setattr(p, "load_source", lambda *a: fixture_data())
    original_trial = p.trial
    calls = 0
    def interrupted(*a):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return original_trial(*a)
    monkeypatch.setattr(p, "trial", interrupted)
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        p.execute(Path("unused"), out)
    first = (out / "trials/development_only/0000.json").read_bytes()
    assert len(list((out / "failures").glob("*.json"))) == 1
    monkeypatch.setattr(p, "trial", original_trial)
    report = p.execute(Path("unused"), out)
    assert report == p.execute(Path("unused"), out)
    assert (out / "trials/development_only/0000.json").read_bytes() == first
    assert len(list((out / "trials").glob("*/*.json"))) == 12
    (out / "trials/development_only/0000.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        p.execute(Path("unused"), out)
