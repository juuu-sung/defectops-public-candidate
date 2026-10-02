import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import mpi_normal_diversity_evaluation_protocol as p
from scripts import evaluate_mpi_normal_diversity as r

PROJECT = Path(__file__).resolve().parents[1]


def data():
    raw = {}
    for group, count in p.COUNTS.items():
        raw[group] = []
        for i in range(count):
            label = int(group == "test" and i >= 54)
            prefix = ("defect" if label else "good") if group == "test" else group
            offset = {"development": 0, "calibration": 128, "test": 500, "validation": 800}[group]
            pid = f"{i+offset:04d}"
            value = float(i+1) if group in ("development", "calibration") else (200. if label else 1.)
            raw[group].append({"key": prefix+"/"+pid, "photo_id": pid, "label": label,
                "regions": [{"region": region, "member": f"{group}/{region}/{pid}.png",
                             "top1": value+j, "native_raw": value+j}
                            for j, region in enumerate(p.fit.REGIONS)]})
    return raw


def policy():
    result = json.loads((PROJECT/"configs/mpi-normal-diversity-v1.json").read_bytes())
    result["acceptance"]["bootstrap"]["repetitions_per_scope"] = 3
    return result


def method(raw):
    x = p.matrix(raw)
    return p.fit_method(x["development"], x["calibration"])


def test_contract_matches_scalar_and_q95_boundary():
    raw = data()
    actual = method(raw)
    assert actual == p.c.calibrate(raw["development"], raw["calibration"])
    assert p.contract_replay(raw, actual)["maximum_absolute_difference"] == 0
    assert p.c.scoring.score(raw["calibration"][160], actual)["alert"]
    assert not p.c.scoring.score(raw["calibration"][159], actual)["alert"]
    with pytest.raises(ValueError):
        p.contract_replay(raw, actual | {"threshold": actual["threshold"]+.1})


@pytest.mark.parametrize("problem", ["order", "count", "label", "duplicate", "nan", "negative", "region", "path"])
def test_frozen_cohort_validation(problem):
    expected, actual = data(), data()
    if problem == "order": actual["validation"].reverse()
    if problem == "count": actual["test"].pop()
    if problem == "label": actual["validation"][0]["label"] = 1
    if problem == "duplicate": actual["test"][1] = actual["test"][0]
    if problem == "nan": actual["test"][0]["regions"][0]["top1"] = float("nan")
    if problem == "negative": actual["test"][0]["regions"][0]["native_raw"] = -1
    if problem == "region": actual["test"][0]["regions"].reverse()
    if problem == "path": actual["test"][0]["regions"][0]["member"] = "wrong.png"
    with pytest.raises(ValueError): p.validate_raw(actual, expected)


def test_loss_not_hidden_by_gained_defect():
    baseline, candidate = data(), data()
    for row in baseline["test"][-1]["regions"]: row["top1"] = 1.
    for row in candidate["test"][-2]["regions"]: row["top1"] = 1.
    result, _ = p.compare(baseline, candidate, {"baseline": method(baseline), "candidate": method(candidate)})
    assert result["counts"]["baseline"]["tp"] == result["counts"]["candidate"]["tp"] == 54
    assert result["lost_baseline_detections"] == [candidate["test"][-2]["key"]]
    assert result["gained_detections"] == [candidate["test"][-1]["key"]]
    assert not result["passed"]


def test_bootstrap_paired_deterministic_draw_order():
    raw, cfg = data(), policy()
    result = p.bootstrap(raw, raw, cfg)
    for scope, seed in zip(cfg["acceptance"]["bootstrap"]["scopes"], cfg["acceptance"]["bootstrap"]["seeds"]):
        rng = np.random.Generator(np.random.PCG64(seed))
        for trial in result["scopes"][scope]:
            di = rng.integers(0, 128, 128) if scope in ("development_only", "both") else np.arange(128)
            ci = rng.integers(0, 169, 169) if scope in ("calibration_only", "both") else np.arange(169)
            assert trial["development_indices"] == di.tolist()
            assert trial["calibration_indices"] == ci.tolist()
            assert trial["models"]["baseline"] == trial["models"]["candidate"]
    assert result == p.bootstrap(raw, raw, cfg)


def test_bootstrap_invalid_is_not_dropped_or_redrawn():
    baseline, candidate = data(), data()
    for photo in candidate["development"]:
        for row in photo["regions"]: row["top1"] = 1.
    result = p.bootstrap(baseline, candidate, policy())
    assert not result["passed"]
    for value in result["summary"].values():
        assert value["invalid_repetitions"] == 3
        assert value["quantiles"] is None


def test_bootstrap_does_not_fit_using_test(monkeypatch):
    raw = data()
    original = p.fit_method
    def fit(dev, cal):
        assert dev.shape == (128, 12) and cal.shape == (169, 12)
        assert dev.max() <= 139 and cal.max() <= 180
        return original(dev, cal)
    monkeypatch.setattr(p, "fit_method", fit)
    p.bootstrap(raw, raw, policy())


@pytest.mark.parametrize("kind", ["scale", "threshold", "nan"])
def test_bad_method_rejected(kind):
    dev = np.ones((128, 12)) if kind == "scale" else np.tile(np.arange(128, dtype=float)[:, None], (1, 12))
    cal = np.ones((169, 12))
    if kind == "threshold": cal[:] = float("nan")
    if kind == "nan": dev[0, 0] = float("nan")
    with pytest.raises(ValueError): p.fit_method(dev, cal)


def test_width_gate_requires_half_not_merely_smaller():
    cfg = policy()["acceptance"]["bootstrap"]
    cfg["repetitions_per_scope"] = 1000
    def trials(a, b):
        return [{"valid": True, "models": {m: {"counts": {"tp": x, "fp54": 0, "fp256": 0}}
                                          for m, x in (("baseline", a[i % 2]), ("candidate", b[i % 2]))}} for i in range(1000)]
    assert p.summarize(trials([51, 55], [53, 55]), cfg)["gates"]["tp_width_halved"]
    assert not p.summarize(trials([51, 55], [52, 55]), cfg)["gates"]["tp_width_halved"]
    assert p.summarize(trials([53, 53], [53, 53]), cfg)["gates"]["tp_width_halved"]


def test_replay_rejects_decision_flip_even_below_raw_tolerance():
    raw = data()
    m = method(raw)
    expected = raw["calibration"][160]
    actual = copy.deepcopy(expected)
    for row in actual["regions"]: row["top1"] -= 1e-5
    with pytest.raises(ValueError, match="decision_changed=True"): r.replay(actual, expected, m)


def test_fresh_process_commands_are_distinct():
    args = SimpleNamespace(project=Path("C:/project"), archive=Path("C:/Downloads/a.zip"), handoffs=Path("C:/handoff"))
    assert r.worker_command(args, "baseline-warm")[-2:] == ["--job", "baseline-warm"]
    assert r.worker_command(args, "candidate-warm")[-2:] == ["--job", "candidate-warm"]
    assert r.JOBS == ("score", "baseline-warm", "candidate-warm", "bootstrap")


def test_unaccounted_attempt_blocks_resume(tmp_path):
    (tmp_path/"attempts/attempt-crashed").mkdir(parents=True)
    with pytest.raises(ValueError, match="without a cost receipt"): r.attempts(tmp_path)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    raw = data()
    methods = {m: method(raw) for m in ("baseline", "candidate")}
    source = {"config": {}, "cohorts": raw, "reference": raw, "previous": {},
              "records": {"fit.json": {"checkpoint_context": {}}}}
    env = {"kind": "FAKE_CPU_NOT_GPU_EVIDENCE"}
    records = {"first.json": {g: raw[g] for g in ("development", "calibration")},
               "training.json": {"context": {}}, "environment.json": env,
               "report.json": {"resource_review_required": True}, "attempts.json": [{"error": "C-stage 7 GiB violation"}]}
    identity = {"policy": policy(), "methods": methods}
    state = {"calls": 0, "fail": None, "drift": False, "budget": False, "loads": 0}
    archive = tmp_path/"source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for photos in raw.values():
            for photo in photos:
                for row in photo["regions"]: z.writestr(row["member"], b"synthetic")
    class Runtime:
        def __init__(self, config): pass
        def load(self, path, context): state["loads"] += 1
        def release(self): pass
    class Monitor:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def check(self):
            if state["budget"]: raise RuntimeError("synthetic resource cap")
        def snapshot(self): return {"fake": True}
    def score(z, photo, runtime, method, recovered):
        state["calls"] += 1
        if state["calls"] == state["fail"]: raise RuntimeError("synthetic interruption")
        result = copy.deepcopy(photo)
        if state["drift"]:
            result["regions"][0]["top1"] += 1
        return result, p.c.scoring.score(result, method), .01
    monkeypatch.setattr(r, "score_photo", score)
    monkeypatch.setattr(p.c.selection.spatial.ev, "recovered_bytes", lambda *args: b"recovered")
    def run(job):
        return r.run_job(job, tmp_path, archive, source, records, identity, {"baseline": archive, "candidate": archive}, archive,
                         Runtime, Monitor, lambda runtime: env)
    return tmp_path, run, state, records, identity, source


def test_complete_resume_and_export_preserves_C_failure(worker):
    root, run, state, records, identity, source = worker
    values = {job: run(job) for job in r.JOBS}
    calls = state["calls"]
    assert run("score") == values["score"] and calls == state["calls"]
    assert state["loads"] == 3
    for job in ("baseline-warm", "candidate-warm"):
        assert values[job]["count"] == 96
        assert len(values[job]["measurements"]) == 96
    report = r.export(root, records, identity, root/"handoff.zip")
    assert report["D_passed"] and report["C_resource_review_required"]
    assert not report["overall_candidate_accepted"] and not report["automatic_pass_allowed"]
    assert not report["training_performed"] and not report["new_independent_evaluation"]
    assert r.export(root, records, identity, root/"handoff.zip") == report


def test_interruption_reuses_completed_photos_and_keeps_cost(worker):
    root, run, state, *_ = worker
    state["fail"] = 19  # gate16 + two saved test photos
    with pytest.raises(RuntimeError, match="interruption"): run("score")
    assert len(list((root/"scores").glob("test-*"))) == 2
    state["fail"] = None
    before = state["calls"]
    run("score")
    assert state["calls"]-before == 16+365-2
    costs = r.attempts(root)
    assert len(costs) == 2 and sum(a["outcome"] == "failed" for a in costs) == 1


def test_gate_drift_stops_before_any_test(worker):
    root, run, state, *_ = worker
    state["drift"] = True
    with pytest.raises(ValueError, match="replay failed"): run("score")
    assert not (root/"scores").exists()
    assert r.attempts(root)[0]["outcome"] == "failed"


def test_budget_stops_before_test(worker):
    root, run, state, *_ = worker
    state["budget"] = True
    with pytest.raises(RuntimeError, match="resource cap"): run("score")
    assert state["calls"] == 0
    assert r.attempts(root)[0]["error"]


def test_warm_retry_restarts_96_measurements(worker):
    root, run, state, *_ = worker
    run("score")
    state["fail"] = state["calls"]+16+5+10
    with pytest.raises(RuntimeError): run("baseline-warm")
    state["fail"] = None
    before = state["calls"]
    result = run("baseline-warm")
    assert state["calls"]-before == 16+5+96
    assert result["count"] == 96
    assert len(r.attempts(root)) == 3


def test_cache_corruption_preserved_and_rejected(worker):
    root, run, state, *_ = worker
    state["fail"] = 19
    with pytest.raises(RuntimeError): run("score")
    path = root/"scores/test-000/sample.json"
    path.write_bytes(b"tampered")
    state["fail"] = None
    with pytest.raises(ValueError, match="incomplete or changed"): run("score")
    assert path.read_bytes() == b"tampered"


@pytest.mark.parametrize("field", ["seconds", "memory"])
def test_monitor_enforces_caps(field):
    monitor = object.__new__(r.Monitor)
    monitor.sample = lambda: None
    monitor.runtime = None
    monitor.error = None
    monitor.rss = 21*1024**3 if field == "memory" else 0
    monitor.reserved = 0
    monitor.prior, monitor.started = 7201, r.time.perf_counter()
    if field == "memory": monitor.prior = 0
    monitor.limits = policy()["resource_plan"]
    with pytest.raises(RuntimeError, match="cap exceeded"): monitor.check()


@pytest.mark.parametrize("group", ["test", "validation"])
def test_photo_inference_uses_fixed_reader_and_all_12_regions(monkeypatch, group):
    raw = data()
    photo = raw[group][0]
    calls = []
    class Image:
        def __init__(self, row): self.row = row
        def close(self): calls.append("closed")
    def normal(z, row):
        assert group == "validation"
        calls.append(row["region"])
        return Image(row)
    def test(z, row, recovered):
        assert group == "test" and recovered == b"verified"
        calls.append(row["region"])
        return Image(row)
    class Runtime:
        def synchronize(self): pass
        def predict(self, image): return 10., np.full((256, 256), 8., dtype=np.float32)
    monkeypatch.setattr(r.fit, "checked_image", normal)
    monkeypatch.setattr(p.c.selection.spatial.ev, "read_tile", test)
    sample, decision, seconds = r.score_photo(None, photo, Runtime(), method(raw), b"verified")
    assert calls[::2] == list(p.fit.REGIONS)
    assert calls[1::2] == ["closed"]*12
    assert all(x["top1"] == 8. and x["native_raw"] == 10. for x in sample["regions"])
    assert seconds >= 0 and not decision["alert"]


@pytest.mark.parametrize("bad", ["nan", "shape", "negative"])
def test_inference_bad_map_stops_and_closes_image(monkeypatch, bad):
    image = SimpleNamespace(close=lambda: closed.append(True))
    closed = []
    values = np.ones((256, 256), dtype=np.float32)
    if bad == "nan": values[0, 0] = np.nan
    if bad == "negative": values[0, 0] = -1
    if bad == "shape": values = values[:128]
    runtime = SimpleNamespace(synchronize=lambda: None, predict=lambda im: (10., values))
    monkeypatch.setattr(r.fit, "checked_image", lambda *args: image)
    raw = data()
    with pytest.raises(ValueError, match="Invalid anomaly map"):
        r.score_photo(None, raw["validation"][0], runtime, method(raw), None)
    assert closed == [True]


def test_failed_warm_time_is_reported_not_aborted(worker, monkeypatch):
    root, run, state, *_ = worker
    run("score")
    original = r.score_photo
    def slow(*args):
        sample, decision, seconds = original(*args)
        return sample, decision, 2.1
    monkeypatch.setattr(r, "score_photo", slow)
    result = run("candidate-warm")
    assert not result["passed"] and result["p95_higher_seconds"] == 2.1
    assert result["count"] == 96


def test_model_load_peak_survives_loader_reset(monkeypatch):
    runtime = object.__new__(r.EvaluationRuntime)
    runtime.load_allocated = runtime.load_reserved = 0
    runtime.synchronize = lambda: None
    runtime.torch = SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200))
    monkeypatch.setattr(r.fit.TorchRuntime, "reset_peak", lambda self: None)
    monkeypatch.setattr(r.fit.TorchRuntime, "memory", lambda self: {"torch_cuda_peak_allocated_bytes": 50,
                                                                  "torch_cuda_peak_reserved_bytes": 80})
    runtime.reset_peak()
    assert runtime.memory()["torch_cuda_peak_allocated_bytes"] == 100
    assert runtime.memory()["torch_cuda_peak_reserved_bytes"] == 200
