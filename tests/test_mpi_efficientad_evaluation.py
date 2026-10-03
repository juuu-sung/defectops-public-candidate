"""CPU regression and orchestration tests, not GPU accuracy evidence."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import mpi_efficientad_evaluation_protocol as e
from scripts import run_mpi_efficientad_evaluation as r


def contract():
    return {"centers": [0.]*12, "scales": [1.]*12, "threshold": 1.,
            "map_quantiles": [{"q90": 0., "q995": 1.}]*2}


def photo(key, label=1):
    return {"key": key, "photo_id": key.split("/")[-1], "label": label, "regions": []}


def test_signed_scores_inclusive_boundary_and_inventory():
    c = contract()
    assert e.decision([-2.]*12, c)["score"] == -2.
    assert not e.decision([-2.]*12, c)["alert"]
    assert e.decision([1.]*12, c)["alert"]
    for x in ([1]*11, [float("nan")]*12, [float("inf")]*12):
        with pytest.raises(ValueError): e.decision(x, c)
    values = [e.sample(photo(k), [2.]*12, c) for k in e.POLICY["acceptance"]["regression_keys"]]
    assert e.screen(values)["retained"] == 12
    values[0]["alert"] = False
    assert not e.screen(values)["passed"] and e.screen(values)["retained"] == 11
    with pytest.raises(ValueError): e.screen(values[::-1])


def test_draws_fixed_paired_whole_photos():
    a = list(e.draws())
    assert len(a) == 600 and [x[0] for x in a[::200]] == ["development_only", "calibration_only", "both"]
    for x, y in zip(a, e.draws(), strict=True):
        assert x[:2] == y[:2] and np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3])
        assert x[2].shape == (128,) and x[3].shape == (169,)
    assert np.array_equal(a[0][3], np.arange(169))
    assert np.array_equal(a[200][2], np.arange(128))
    assert len(set(a[0][2])) < 128


def test_stability_additive_width_rule_and_invalid_no_reroll():
    trials = [{"valid": True, **{m: {"counts": {"tp": 53, "fp54": 2, "fp256": 12}} for m in ("baseline", "candidate")}} for _ in range(200)]
    assert e.summarize(trials)["passed"]
    trials[0]["valid"] = False
    assert not e.summarize(trials)["passed"]
    assert not e.summarize(trials[1:])["passed"]


class Monitor:
    runtime = None
    def __init__(self, *args): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake": True}


class Runtime:
    calls = 0
    def __init__(self, *args): Runtime.calls += 1
    def release(self): pass


def setup_run(tmp_path, monkeypatch, pass_screen=True, pass_full=True, pass_warm=True):
    keys = e.POLICY["acceptance"]["regression_keys"]
    tests = [photo(k) for k in keys] + [photo(f"defect/{i:04d}") for i in range(55) if f"defect/{i:04d}" not in keys]
    tests += [photo(f"good/{i:04d}", 0) for i in range(54)]
    cohorts = {"test": tests, "validation": [photo(f"validation/{i:04d}", 0) for i in range(256)],
               "development": [], "calibration": []}
    identity = {"contract": contract(), "fit_signature": "fit", "checkpoint_sha256": "checkpoint"}
    source = {"cohorts": cohorts, "previous": {}, "config": {}, "records": {"fit.json": {"checkpoint_context": {}}}}
    fit_root = tmp_path / "fit"
    fit_root.mkdir()
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w"): pass
    monkeypatch.setattr(r, "normal_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(r.ev, "recovered_bytes", lambda *args: b"recovered")
    original_sha = r.fit.sha256_file
    monkeypatch.setattr(r.fit, "sha256_file", lambda f: ("checkpoint" if str(f).endswith("step-070000/model.pt") else
        r.ev.CHECKPOINT_SHA256 if str(f).endswith("pilot-v1/fit/model.pt") else original_sha(f)))
    accessed = []
    def score(archive, item, runtime, recovered, monitor, c):
        accessed.append(item["key"])
        high = bool(item["label"]) and (pass_screen or item["key"] != keys[0])
        raw = np.full((12, 2, 256, 256), 20. if high else 0., dtype=np.float32)
        s = e.sample(item, r.p.tile_scores(raw, c["map_quantiles"]), c)
        s["photo_seconds"] = .1
        return raw, s
    monkeypatch.setattr(r, "score_photo", score)
    monkeypatch.setattr(r.e, "compare", lambda *args: {"passed": pass_full})
    monkeypatch.setattr(r, "benchmark", lambda *args: {"candidate_budget_passed": pass_warm})
    def sensitive(*args):
        accessed.append("STABILITY")
        return {"passed": True}
    monkeypatch.setattr(r, "stability", sensitive)
    class Baseline(Runtime):
        def load(self, *args): pass
    args = [tmp_path / "out", archive, fit_root, tmp_path / "recovered", {}, source, identity, tmp_path / "handoff.zip"]
    kwargs = {"runtime_factory": Runtime, "monitor_factory": Monitor, "baseline_factory": Baseline}
    return args, kwargs, accessed


def test_screen_failure_stops_before_other_photos_and_exports(tmp_path, monkeypatch):
    args, kw, accessed = setup_run(tmp_path, monkeypatch, pass_screen=False)
    def forbidden(*args): raise AssertionError("Unselected 0054 recovery must remain unopened")
    monkeypatch.setattr(r.ev, "recovered_bytes", forbidden)
    result = r.execute(*args, **kw)
    assert accessed == e.POLICY["acceptance"]["regression_keys"]
    assert not result["screen_passed"] and not result["full_comparison_run"]
    assert not result["warm_evaluated"] and not result["stability_evaluated"]
    with zipfile.ZipFile(args[-1]) as z:
        assert all(n.endswith(".json") for n in z.namelist())
        assert json.loads(z.read("screen.json"))["retained"] == 11
    calls = Runtime.calls
    assert r.execute(*args, **kw) == result and Runtime.calls == calls


@pytest.mark.parametrize("field", ["maps", "sample", "gate", "identity"])
def test_cache_corruption_refused(tmp_path, monkeypatch, field):
    args, kw, _ = setup_run(tmp_path, monkeypatch, pass_screen=False)
    r.execute(*args, **kw)
    root = args[0]
    paths = {"maps": root / "test/photo-000/maps.npy", "sample": root / "test/photo-000/sample.json",
             "gate": next(root.glob("attempt-*/normal-gate.json")), "identity": root / "identity/value.json"}
    paths[field].write_bytes(b"corrupted")
    with pytest.raises((ValueError, KeyError)): r.execute(*args, **kw)


def test_normal_gate_failure_prevents_test_access(tmp_path, monkeypatch):
    args, kw, accessed = setup_run(tmp_path, monkeypatch)
    def fail(*args): raise ValueError("normal replay failed")
    monkeypatch.setattr(r, "normal_gate", fail)
    with pytest.raises(ValueError, match="normal replay"): r.execute(*args, **kw)
    assert accessed == []
    with pytest.raises(ValueError, match="Prior failed"): r.execute(*args, **kw)


def test_ctrl_c_resume_reuses_completed_screen_samples(tmp_path, monkeypatch):
    args, kw, accessed = setup_run(tmp_path, monkeypatch, pass_screen=False)
    real_score = r.score_photo
    def interrupt(*a):
        if len(accessed) == 4: raise KeyboardInterrupt()
        return real_score(*a)
    monkeypatch.setattr(r, "score_photo", interrupt)
    with pytest.raises(KeyboardInterrupt): r.execute(*args, **kw)
    monkeypatch.setattr(r, "score_photo", real_score)
    assert not r.execute(*args, **kw)["screen_passed"]
    assert accessed == e.POLICY["acceptance"]["regression_keys"]


def test_full_flow_and_conditional_stability_without_large_maps(tmp_path, monkeypatch):
    args, kw, accessed = setup_run(tmp_path, monkeypatch)
    # Smaller test arrays only inside this orchestration fake; production map validator remains separate.
    monkeypatch.setattr(r.p, "tile_scores", lambda raw, q: np.full(12, float(raw.flat[0]) / 10))
    real_score = r.score_photo
    def small(*a):
        raw, sample = real_score(*a)
        return raw[:, :, :1, :1], sample
    monkeypatch.setattr(r, "score_photo", small)
    result = r.execute(*args, **kw)
    assert result["all_gates_passed"] and result["cached_photos"] == 365
    assert accessed[-1] == "STABILITY" and len(accessed) == 366
    assert len(set(accessed)) == 366


@pytest.mark.parametrize("full,warm", [(False, True), (True, False)])
def test_full_or_warm_failure_skips_bootstrap(tmp_path, monkeypatch, full, warm):
    args, kw, accessed = setup_run(tmp_path, monkeypatch, pass_full=full, pass_warm=warm)
    monkeypatch.setattr(r.p, "tile_scores", lambda raw, q: np.full(12, float(raw.flat[0]) / 10))
    real_score = r.score_photo
    def small(*a):
        raw, sample = real_score(*a)
        return raw[:, :, :1, :1], sample
    monkeypatch.setattr(r, "score_photo", small)
    result = r.execute(*args, **kw)
    assert result["full_comparison_run"] and result["warm_evaluated"] and not result["stability_evaluated"]
    assert "STABILITY" not in accessed


def test_map_resampling_recomputes_quantiles_not_only_threshold():
    a = np.arange(2*256*256, dtype=np.float32).reshape(1, 2, 256, 256)
    b = a * 3
    q1 = r.p.map_quantiles([a, b])
    q2 = r.p.map_quantiles([b, b])
    assert q1 != q2
    for c in range(2):
        # Repeated observations change finite-sample linear interpolation positions.
        expected = np.quantile(np.concatenate([b[:, c].ravel()]*2), [.9, .995], method="linear")
        assert q2[c]["q90"] == expected[0]
        assert q2[c]["q995"] == expected[1]


def test_actual_bootstrap_refits_all_paths_and_resumes(tmp_path, monkeypatch):
    policy = copy.deepcopy(e.POLICY)
    policy["acceptance"]["stability"]["repetitions_per_scope"] = 1
    monkeypatch.setattr(e, "POLICY", policy)
    counts = {"development": 128, "calibration": 169, "test": 109, "validation": 256}
    groups = {g: [photo(f"{g}/{i:04d}", int(g == "test" and i < 55)) for i in range(n)] for g, n in counts.items()}
    base = {g: np.repeat(np.arange(n)[:, None], 12, axis=1).astype(float) for g, n in counts.items()}
    monkeypatch.setattr(r.comparison, "matrix", lambda raw: base)
    monkeypatch.setattr(r.np, "load", lambda path, **kw: np.full((12, 2, 1, 1), int(path.parent.name.split("-")[-1]), np.float32))
    quantile_calls = []
    def quantiles(arrays, check):
        quantile_calls.append([int(a.flat[0]) for a in arrays])
        return [{"q90": float(np.mean(quantile_calls[-1])), "q995": 999.}]*2
    monkeypatch.setattr(r.p, "map_quantiles", quantiles)
    monkeypatch.setattr(r.p, "tile_scores", lambda a, q: np.full(12, float(a.flat[0])-q[0]["q90"]))
    records = {"normal-results.json": {"contract": {"map_quantiles": [{"q90": 0., "q995": 999.}]*2},
                                        "tile_scores": {g: base[g] for g in ("development", "calibration")}}}
    source = {"cohorts": groups, "reference": {}}
    result = r.stability(tmp_path, "identity", tmp_path, source, records, Monitor())
    assert len(quantile_calls) == 2  # development_only and both, not calibration_only
    generated = list(e.draws())
    assert quantile_calls[0] == generated[0][2].tolist()
    assert quantile_calls[1] == generated[2][2].tolist()
    trials = result["trials"]
    assert all(trials[g][0]["valid"] for g in trials)
    before = copy.deepcopy(result)
    assert r.stability(tmp_path, "identity", tmp_path, source, records, Monitor()) == before
    assert len(quantile_calls) == 2


def test_completed_cache_cannot_bypass_failed_screen(tmp_path, monkeypatch):
    args, kw, _ = setup_run(tmp_path, monkeypatch, pass_screen=False)
    r.execute(*args, **kw)
    target = args[0] / "validation/photo-000"
    target.mkdir()
    with pytest.raises(ValueError): r.execute(*args, **kw)


def test_full_compare_checks_identity_and_lost_detections():
    c = contract()
    photos = [photo(f"defect/{i:04d}") for i in range(55)] + [photo(f"good/{i:04d}", 0) for i in range(54)]
    method = {"field": "top1", "threshold": 1., "regions": [
        {"region": reg, "center": 0., "scale": 1.} for reg in e.fit.REGIONS]}
    reference = [{**s, "regions": [{"region": reg, "top1": 2. if i < 53 else 0.} for reg in e.fit.REGIONS]}
                 for i, s in enumerate(photos)]
    source = {"contract": {"candidate_top1_q90": method}, "reference": {"test": reference}}
    samples = {"test": [e.sample(s, [2. if i < 54 else 0.]*12, c) for i, s in enumerate(photos)],
               "validation": [e.sample(photo(f"validation/{i:04d}", 0), [0.]*12, c) for i in range(256)]}
    result = e.compare(samples, source)
    assert result["passed"] and result["gained_keys"] == ["defect/0053"]
    samples["test"][0] = e.sample(photos[0], [0.]*12, c)
    assert not e.compare(samples, source)["passed"]
    samples["test"] = samples["test"][::-1]
    with pytest.raises(ValueError, match="order"): e.compare(samples, source)


def test_policy_reuses_frozen_acceptance_and_never_updates_model():
    assert e.POLICY["acceptance"] == r.p.POLICY["future_evaluation"]
    assert e.POLICY["warm_photos"] * e.POLICY["warm_repeats"] == 60
    source = (Path(r.__file__).parent / "mpi_efficientad_evaluation_runtime.py").read_text()
    assert "optimizer.step" not in source and "normalize=False" in source
    assert 'state["step"] != 70000' in source and "weights_only=True" in source


def test_candidate_warm_fixed_order_and_scope(monkeypatch):
    c = contract()
    photos = [photo(f"calibration/{i:04d}", 0) for i in range(169)]
    source = {"cohorts": {"calibration": photos}, "reference": {"calibration": []},
              "contract": {"candidate_top1_q90": {}}}
    records = {"normal-results.json": {"tile_scores": {"calibration": [[0.]*12]*169}}}
    called = []
    def score(archive, item, runtime, recovered, monitor, contract, **kwargs):
        called.append(item["key"])
        assert kwargs == {"check_tiles": False}
        return None, e.sample(item, [0.]*12, contract) | {"photo_seconds": .25}
    monkeypatch.setattr(r, "score_photo", score)
    runtime = type("Warm", (), {"memory": lambda self: {"fake": True}})()
    report = r.benchmark("candidate", runtime, None, source, records, c, Monitor())
    assert called == [p["key"] for p in photos[:5]] + [p["key"] for p in photos[:20]]*3
    assert len(report["seconds"]) == 60 and report["p95_higher_seconds"] == .25
    assert report["candidate_budget_passed"]


def test_resource_failure_is_preserved_and_not_retried(tmp_path, monkeypatch):
    args, kw, accessed = setup_run(tmp_path, monkeypatch)
    class Cap(Monitor):
        def check(self): raise RuntimeError("time or memory cap exceeded")
    kw["monitor_factory"] = Cap
    with pytest.raises(RuntimeError, match="cap exceeded"): r.execute(*args, **kw)
    assert accessed == []
    record = json.loads(next(args[0].glob("attempt-*/finished.json")).read_bytes())
    assert record["status"] == "failed" and "cap exceeded" in record["error"]
    with pytest.raises(ValueError, match="Prior failed"): r.execute(*args, **kw)


def test_unaccounted_kill_requires_review(tmp_path, monkeypatch):
    args, kw, _ = setup_run(tmp_path, monkeypatch, pass_screen=False)
    r.execute(*args, **kw)
    folder = args[0] / "attempt-unaccounted"
    folder.mkdir()
    (folder / "started.json").write_text('{}')
    with pytest.raises(ValueError, match="Unfinished attempt"): r.execute(*args, **kw)
