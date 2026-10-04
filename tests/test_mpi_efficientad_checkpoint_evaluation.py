"""CPU workflow checks with fake inference; not checkpoint accuracy evidence."""
import copy
import hashlib
import json
import zipfile
import numpy as np
import pytest
from scripts import run_mpi_efficientad_checkpoint_evaluation as r
from scripts import mpi_efficientad_checkpoint_protocol as c


def contract(threshold=1.):
    return {"map_quantiles": [{"q90": 0., "q995": 1.}]*2,
            "centers": [0.]*12, "scales": [1.]*12, "threshold": threshold}


class Monitor:
    runtime = None
    def __init__(self, *args): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake": True}


class Runtime:
    calls = []
    def __init__(self, checkpoint, signature, step): self.calls.append(step)
    def release(self): pass


def fixture(tmp_path, monkeypatch, passes=False, full_passes=False, warm_passes=True):
    def photo(key, label):
        return {"key": key, "photo_id": key.split("/")[-1], "label": label, "regions": []}
    keys = r.e.POLICY["acceptance"]["regression_keys"]
    tests = [photo(k, 1) for k in keys] + [photo(f"defect/{i:04d}", 1) for i in range(55) if f"defect/{i:04d}" not in keys]
    tests += [photo(f"good/{i:04d}", 0) for i in range(54)]
    tests[4]["regions"] = [{"member": r.old.ev.REPAIR_MEMBER}]
    cohorts = {"development": [], "calibration": [], "test": tests,
               "validation": [photo(f"validation/{i:04d}", 0) for i in range(256)]}
    source = {"cohorts": cohorts, "previous": {}, "config": {}, "records": {"fit.json": {"checkpoint_context": {}}}}
    checkpoints = {str(s): {"receipt": {"files": {"model.pt": "model"}}} for s in c.STEPS}
    identity = {"fit_signature": "fit", "checkpoints": checkpoints, "cohorts": cohorts}
    archive = tmp_path/"source.zip"
    with zipfile.ZipFile(archive, "w"): pass
    access = []
    def normal(root, archive, runtime, monitor, source, sig):
        access.append("normals")
        # Intentionally different contract per checkpoint; no final threshold reuse.
        value = {"contract": contract(1. if root.name.endswith("010000") else 1.5), "tile_scores": {}}
        r.publish(root, "normal-results", sig, value)
        return value
    monkeypatch.setattr(r, "normals", normal)
    def gate(*args):
        access.append("reload")
        return {"passed": True}
    monkeypatch.setattr(r.old, "normal_gate", gate)
    monkeypatch.setattr(r.old.ev, "recovered_bytes", lambda *args: b"approved")
    original_sha = r.fit.sha256_file
    monkeypatch.setattr(r.fit, "sha256_file", lambda f: "model" if str(f).endswith("model.pt") and "pilot-v1" not in str(f) else
                        r.old.ev.CHECKPOINT_SHA256 if str(f).endswith("pilot-v1/fit/model.pt") else original_sha(f))
    # Small map storage for orchestration; independent cache-shape tests use full maps below.
    def score(archive, item, runtime, recovered, monitor, method):
        if item["key"] == "defect/0054": assert recovered == b"approved"
        access.append(item["key"])
        hit = bool(item["label"]) and (passes or item["key"] != keys[0])
        raw = np.full((12, 2, 2, 2), 20. if hit else 0., dtype=np.float32)
        sample = r.e.sample(item, [2. if hit else 0.]*12, method)
        return raw, {**sample, "photo_seconds": .1}
    monkeypatch.setattr(r.old, "score_photo", score)
    def cache(root, group, i, sig, photo, method):
        phase = f"photo-{i:03d}"
        rec = r.fit.read_phase(root/group, phase, sig, ["maps.npy", "sample.json"])
        assert rec["photo_sha256"] == r.fit.digest(photo)
        gate_path = root/rec["gate_path"]
        if r.fit.sha256_file(gate_path) != rec["gate_sha256"]: raise ValueError("gate changed")
        return json.loads((root/group/phase/"sample.json").read_bytes())
    monkeypatch.setattr(r.old, "read_cache", cache)
    monkeypatch.setattr(r.e, "compare", lambda *args: {"passed": full_passes})
    monkeypatch.setattr(r.old, "benchmark", lambda *args: {"candidate_budget_passed": warm_passes})
    def stability(*args):
        access.append("stability")
        return {"passed": True}
    monkeypatch.setattr(r.old, "stability", stability)
    class Baseline:
        def __init__(self, *args): pass
        def load(self, *args): pass
        def release(self): pass
    args = [tmp_path/"out/step-010000", archive, tmp_path/"fit", tmp_path/"recovery", source, identity, 10000]
    kwargs = {"runtime_factory": Runtime, "monitor_factory": Monitor, "baseline_factory": Baseline}
    return args, kwargs, access


def test_fixed_policy_no_training_and_no_final_threshold():
    assert c.STEPS == (10000, 30000)
    assert c.POLICY["acceptance"] == r.e.POLICY["acceptance"]
    assert c.POLICY["threshold_refitted_for_main_comparison"]
    assert not c.POLICY["training_performed"]
    assert not c.POLICY["automatic_pass_allowed"]
    before = copy.deepcopy(r.e.POLICY)
    c.POLICY["acceptance"]["_test"] = True
    try: assert r.e.POLICY == before
    finally: del c.POLICY["acceptance"]["_test"]


def test_failure_skips_full_but_other_checkpoint_runs_and_exports(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    first = r.evaluate_step(*args, **kw)
    assert access == ["normals", "reload"]+r.e.POLICY["acceptance"]["regression_keys"]
    assert first["retained12"] == 11 and not first["full_comparison_run"]
    assert not first["warm_evaluated"] and not first["stability_evaluated"]
    args[0] = tmp_path/"out/step-030000"; args[-1] = 30000
    second = r.evaluate_step(*args, **kw)
    assert first["contract_sha256"] != second["contract_sha256"]
    result = r.finish(tmp_path/"out", tmp_path/"handoff.zip", args[5], [first, second])
    assert result["selected_model"] is None
    with zipfile.ZipFile(tmp_path/"handoff.zip") as z:
        assert all(n.endswith(".json") for n in z.namelist())
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == {*manifest, "handoff-manifest.json"}
    r.finish(tmp_path/"out", tmp_path/"handoff.zip", args[5], [first, second])


@pytest.mark.parametrize("accuracy,warm,expected", [(False, True, False), (True, False, False), (True, True, True)])
def test_full_warm_and_stability_gates(tmp_path, monkeypatch, accuracy, warm, expected):
    args, kw, access = fixture(tmp_path, monkeypatch, passes=True, full_passes=accuracy, warm_passes=warm)
    result = r.evaluate_step(*args, **kw)
    assert result["full_comparison_run"] and result["cached_test_photos"] == 365
    assert result["warm_evaluated"]
    assert result["stability_evaluated"] == expected
    assert ("stability" in access) == expected


def test_completed_resume_no_gpu_and_tamper_refused(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    first = r.evaluate_step(*args, **kw)
    prior = list(access)
    assert r.evaluate_step(*args, **kw) == first and access == prior
    file = args[0]/"test/photo-000/sample.json"
    file.write_bytes(file.read_bytes()+b" ")
    with pytest.raises(ValueError): r.evaluate_step(*args, **kw)


def test_failed_reload_opens_no_test(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    def bad(*args): raise ValueError("reload mismatch")
    monkeypatch.setattr(r.old, "normal_gate", bad)
    with pytest.raises(ValueError, match="reload mismatch"): r.evaluate_step(*args, **kw)
    assert access == ["normals"]
    with pytest.raises(ValueError, match="needs review"): r.evaluate_step(*args, **kw)


def test_interrupt_resumes_but_forced_termination_does_not(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    score = r.old.score_photo
    def stop(archive, item, *rest):
        if item["key"] == "defect/0048": raise KeyboardInterrupt()
        return score(archive, item, *rest)
    monkeypatch.setattr(r.old, "score_photo", stop)
    with pytest.raises(KeyboardInterrupt): r.evaluate_step(*args, **kw)
    assert len(list((args[0]/"test").glob("photo-*"))) == 2
    monkeypatch.setattr(r.old, "score_photo", score)
    result = r.evaluate_step(*args, **kw)
    assert result["retained12"] == 11
    assert access.count("defect/0006") == 1
    (args[0]/"attempt-forced").mkdir()
    with pytest.raises(ValueError, match="Unfinished"): r.evaluate_step(*args, **kw)


def test_illegal_step_rejected_before_io(tmp_path):
    with pytest.raises(ValueError, match="Only preselected"):
        r.evaluate_step(tmp_path, None, None, None, {}, {}, 70000)


def test_normal_cache_identity_shape_and_digest(tmp_path):
    photo = {"key": "calibration/0000"}
    (tmp_path/"calibration").mkdir()
    with r.fit.phase_stage(tmp_path/"calibration", "photo-000") as stage:
        np.save(stage/"maps.npy", np.zeros((12, 2, 256, 256), np.float32), allow_pickle=False)
        r.fit.seal_phase(stage, "photo-000", "sig", {"photo_sha256": r.fit.digest(photo)}, ["maps.npy"])
    assert r.normal_maps(tmp_path, "calibration", 0, photo, "sig").shape == (12, 2, 256, 256)
    with pytest.raises(ValueError): r.normal_maps(tmp_path, "calibration", 0, {"key": "other"}, "sig")
    with pytest.raises(ValueError): r.normal_maps(tmp_path, "calibration", 0, photo, "other")


def test_normal_calibration_uses_own_maps_and_existing_value_is_checked(tmp_path, monkeypatch):
    source = {"cohorts": {g: [{"key": g}] for g in ("development", "calibration")}}
    raw = np.ones((12, 2, 256, 256), np.float32)
    monkeypatch.setattr(r, "raw_normal", lambda *args: raw)
    seen = []
    def quantiles(arrays, check):
        seen.append(float(arrays[0].mean()))
        return contract()["map_quantiles"]
    monkeypatch.setattr(r.p, "map_quantiles", quantiles)
    monkeypatch.setattr(r.p, "calibrate", lambda **kw: {**contract(), "normal_inputs": kw})
    value = r.normals(tmp_path, None, None, Monitor(), source, "sig")
    assert seen == [1.] and value["tile_scores"]["calibration"][0] == pytest.approx([.1]*12)
    assert r.normals(tmp_path, None, None, Monitor(), source, "sig") == value


def test_resource_failure_releases_and_blocks_automatic_retry(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    class Capped(Monitor):
        def check(self):
            if self.runtime is not None: raise RuntimeError("resource cap")
    kw["monitor_factory"] = Capped
    with pytest.raises(RuntimeError, match="resource cap"): r.evaluate_step(*args, **kw)
    with pytest.raises(ValueError, match="needs review"): r.evaluate_step(*args, **kw)


def test_reviewed_trajectory_requires_inventory_hashes_and_matching_checkpoint(tmp_path, monkeypatch):
    records = {"plan.json": {"name": "historical"}}
    checkpoints = {"10000": {"receipt": "frozen"}}
    plan = {"policy": c.t.POLICY, "checkpoints": checkpoints, "fit_identity": c.fit.digest(records["plan.json"])}
    raw = {"plan.json": c.fit.canonical(plan), "report.json": c.fit.canonical({"status": "checkpoint_trajectory_completed_exploratory"})}
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    path = tmp_path/"trajectory.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, value in raw.items(): z.writestr(name, value)
        z.writestr("handoff-manifest.json", c.fit.canonical(manifest))
    monkeypatch.setattr(c, "TRAJECTORY_SHA", c.fit.sha256_file(path))
    assert c.reviewed_trajectory(path, records, checkpoints) == plan
    with pytest.raises(ValueError, match="provenance"):
        c.reviewed_trajectory(path, records, {})
    with zipfile.ZipFile(path, "a") as z: z.writestr("extra.json", "{}")
    with pytest.raises(ValueError, match="reviewed trajectory"):
        c.reviewed_trajectory(path, records, checkpoints)
    monkeypatch.setattr(c, "TRAJECTORY_SHA", c.fit.sha256_file(path))
    with pytest.raises(ValueError, match="inventory"):
        c.reviewed_trajectory(path, records, checkpoints)


def test_cli_main_runs_both_steps_with_real_paths_and_no_auto_selection(tmp_path, monkeypatch):
    args, kw, access = fixture(tmp_path, monkeypatch)
    (tmp_path/"artifacts"/r.p.NAME).mkdir(parents=True)
    monkeypatch.setattr(r.c, "prepare", lambda *a: (args[4], {**args[5], "environment": {}}))
    monkeypatch.setattr(r, "environment", lambda: {})
    original_sha = r.fit.sha256_file
    monkeypatch.setattr(r.fit, "sha256_file", lambda f: r.fit.EXPECTED_ARCHIVE_SHA256 if f == args[1] else original_sha(f))
    class Disk: free = 100*1024**3
    monkeypatch.setattr(r.shutil, "disk_usage", lambda *a: Disk())
    calls = []
    def run(*a):
        calls.append(a[-1])
        assert a[0].name == f"step-{a[-1]:06d}"
        return {"screen_passed": False}
    monkeypatch.setattr(r, "evaluate_step", run)
    monkeypatch.setattr(r, "finish", lambda *a: {"selected_model": None})
    assert r.main(["--project", str(tmp_path), "--archive", str(args[1])]) == 0
    assert calls == [10000, 30000]


def test_normal_contract_changes_refuse_cached_phase(tmp_path, monkeypatch):
    args, kw, _ = fixture(tmp_path, monkeypatch)
    r.evaluate_step(*args, **kw)
    path = args[0]/"normal-results/value.json"
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError): r.evaluate_step(*args, **kw)


def test_cap_cleanup_still_releases_runtime():
    released = []
    class M(Monitor):
        def check(self): raise RuntimeError("cap")
    class R:
        def release(self): released.append(True)
    m = M()
    m.runtime = R()
    with pytest.raises(RuntimeError, match="cap"): r.release(m.runtime, m)
    assert released == [True] and m.runtime is None
