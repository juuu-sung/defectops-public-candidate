"""CPU orchestration, persistence and arithmetic; explicitly not real CUDA validation."""
import copy
from contextlib import nullcontext
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import mpi_patchcore_scaling_protocol as p
from scripts import mpi_patchcore_scaling_runtime as gpu
from scripts import run_mpi_patchcore_scaling as run
from scripts import mpi_patchcore_scaling_evaluation as evaluation
from tests.test_mpi_normal_diversity_evaluation import data, method


class Tensor(np.ndarray):
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def to(self, *args): return self
    def copy_(self, value): np.copyto(self, value); return self


class NumpyTorch:
    float32 = np.float32
    cuda = SimpleNamespace(empty_cache=lambda: None)
    linalg = SimpleNamespace(norm=lambda x, ord, dim, keepdim: np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim))
    minimum = staticmethod(np.minimum)
    max = staticmethod(lambda x, dim: SimpleNamespace(indices=np.argmax(x, axis=dim)))
    empty = staticmethod(lambda shape, dtype, device: np.empty(shape, dtype=dtype).view(Tensor))
    full = staticmethod(lambda shape, fill, dtype, device: np.full(shape, fill, dtype=dtype).view(Tensor))
    from_numpy = staticmethod(lambda values: values.view(Tensor))
    no_grad = staticmethod(nullcontext)


class Monitor:
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake": True}


def test_locked_design_and_no_time_cap():
    assert p.POLICY["candidates"] == 256*12*1024
    assert p.POLICY["representatives"] == p.probe.budget(64)["representatives"] == 78643
    assert {k: v for k, v in p.POLICY["stability"].items() if k != "refit"} == {
        k: v for k, v in p.evaluation.POLICY["acceptance"]["stability"].items() if k != "refit"}
    assert p.POLICY["time_limit_seconds"] is None
    assert p.POLICY["resource_plan"]["comparison_seconds"] is None
    assert not p.POLICY["automatic_pass_allowed"] and not p.POLICY["service_model_changed"]


def test_raw_blocks_cross_photo_boundaries_without_dropping_rows(tmp_path):
    raw = np.arange(90, dtype=np.float32).reshape(30, 3)
    paths = []
    for i in range(3):
        path = tmp_path/f"{i}.npy"; np.save(path, raw[i*10:(i+1)*10]); paths.append(path)
    chunks = [p.raw_block(paths, a, b, 10, 3) for a, b in p.probe.slices(30, 7)]
    np.testing.assert_array_equal(np.concatenate(chunks), raw)
    with pytest.raises(ValueError): p.raw_block(paths, 29, 31, 10, 3)
    np.save(paths[1], np.full((10, 3), np.nan, np.float32))
    with pytest.raises(ValueError, match="Nonfinite"): p.raw_block(paths, 9, 12, 10, 3)


@pytest.fixture
def selector(tmp_path, monkeypatch):
    cfg = dict(p.POLICY, candidates=129, projected_dimension=17, representatives=23,
               chunk_rows=11, selection_checkpoint_interval=5)
    monkeypatch.setattr(p, "POLICY", cfg)
    raw = np.random.default_rng(12).normal(size=(129, 17)).astype(np.float32)
    paths = []
    for i, (a, b) in enumerate(p.probe.slices(129, 11)):
        path = tmp_path/f"chunk{i}.npy"; np.save(path, raw[a:b]); paths.append(path)
    runtime = object.__new__(gpu.Runtime)
    runtime.torch = NumpyTorch
    return runtime, paths, raw


def test_real_selector_loop_cpu_surrogate_resume_exact(selector, tmp_path):
    runtime, paths, _ = selector
    a = tmp_path/"a"; b = tmp_path/"b"
    expected, receipts = runtime.select(a, "sig", paths, 3, lambda: None)
    def interrupt():
        if (b/"step-000010").exists(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): runtime.select(b, "sig", paths, 3, interrupt)
    assert (b/"step-000010").exists() and not (b/"step-000015").exists()
    actual, resumed = runtime.select(b, "sig", paths, 3, lambda: None)
    assert actual == expected and resumed == receipts
    np.testing.assert_array_equal(np.load(a/"step-000023/minimum.npy"), np.load(b/"step-000023/minimum.npy"))
    # Finished selection does not reread projected arrays or allocate on GPU.
    assert runtime.select(b, "sig", [], 3, lambda: pytest.fail("No computation"))[0] == expected


def test_selector_corruption_gap_and_identity_fail(selector, tmp_path):
    runtime, paths, _ = selector
    root = tmp_path/"states"
    runtime.select(root, "sig", paths, 3, lambda: None)
    with pytest.raises(ValueError): runtime.select(root, "changed", paths, 3, lambda: None)
    (root/"step-000005").rename(root/".preserved-five")
    with pytest.raises(ValueError, match="gap"): runtime.select(root, "sig", paths, 3, lambda: None)
    (root/".preserved-five").rename(root/"step-000005")
    (root/"step-000010/minimum.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incomplete or changed"): runtime.select(root, "sig", paths, 3, lambda: None)


@pytest.mark.parametrize("kind", ["duplicate", "pending", "step", "negative", "inf", "nonzero"])
def test_selection_state_rejects_invalid(kind):
    progress = {"step": 2, "selected": [1, 3], "pending_center": 3}
    minimum = np.ones((8, 1), np.float32); minimum[[1, 3]] = 0
    if kind == "duplicate": progress["selected"] = [1, 1]
    if kind == "pending": progress["pending_center"] = 0
    if kind == "step": progress["step"] = 1
    if kind == "negative": minimum[4] = -1
    if kind == "inf": minimum[4] = np.inf
    if kind == "nonzero": minimum[1] = 1
    with pytest.raises(ValueError): p.validate_selection(progress, minimum, 8, 3)


def test_attempt_interrupt_resume_but_failures_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "Monitor", Monitor)
    root = tmp_path/"root"; root.mkdir()
    with pytest.raises(KeyboardInterrupt):
        with run.attempt(root, "s"): raise KeyboardInterrupt()
    with run.attempt(root, "s"): pass
    assert sorted(a["status"] for a in p.attempts(root)) == ["completed", "interrupted"]
    with pytest.raises(RuntimeError):
        with run.attempt(root, "s"): raise RuntimeError("resource exceeded")
    with pytest.raises(ValueError, match="failed/resource"):
        with run.attempt(root, "s"): pytest.fail("No work after failure")


def test_forced_exit_and_changed_identity_rejected(tmp_path):
    root = tmp_path/"root"; root.mkdir()
    run.initialize(root, {"id": 1})
    with pytest.raises(ValueError): run.initialize(root, {"id": 2})
    (root/"attempt-killed").mkdir()
    with pytest.raises(ValueError, match="Unfinished"): run.initialize(root, {"id": 1})


def test_reload_contract_and_boundary_decision():
    raw = data(); contract = method(raw)
    normal = {g: raw[g] for g in ("development", "calibration")}
    assert p.reload_check(normal, copy.deepcopy(normal), contract)["maximum_absolute_difference"] == 0
    changed = copy.deepcopy(normal)
    changed["calibration"][160]["regions"][0]["top1"] -= .1
    with pytest.raises(ValueError): p.reload_check(normal, changed, contract)
    changed = copy.deepcopy(normal); changed["calibration"].reverse()
    with pytest.raises(ValueError, match="identity"): p.reload_check(normal, changed, contract)


def test_full_comparison_runs_with_zero_detections_and_tracks_losses():
    baseline, candidate = data(), data()
    for sample in candidate["test"]:
        for row in sample["regions"]: row["top1"] = 0.
    source = {"reference": baseline, "contract": {"candidate_top1_q90": method(baseline)}}
    result = evaluation.compare(source, candidate, method(candidate))
    assert result["counts"]["candidate"]["tp"] == 0
    assert len(result["lost_baseline_detections"]) == 55
    assert not result["passed_accuracy_targets"] and not result["miss_improvement_observed"]
    assert len(result["predictions"]["candidate"]["test"]) == 109


def test_all600_paired_draws_kept_invalid_not_rerolled(tmp_path):
    baseline, candidate = data(), data()
    for s in candidate["development"]:
        for r in s["regions"]: r["top1"] = 1.
    result = evaluation.stability(tmp_path, "s", baseline, candidate, lambda: None)
    assert sum(len(t) for t in result["trials"].values()) == 600
    assert not result["passed"]
    for scope, i, di, ci in p.evaluation.draws():
        trial = result["trials"][scope][i]
        assert trial["development_indices"] == di.tolist() and trial["calibration_indices"] == ci.tolist()
        assert not trial["candidate"]["valid"]
    assert evaluation.stability(tmp_path, "s", baseline, candidate, lambda: None) == result


def test_array_refuses_pickle_wrong_dtype_shape(tmp_path):
    path = tmp_path/"a.npy"
    np.save(path, np.zeros((3, 4), np.float64))
    with pytest.raises(ValueError): p.array(path, (3, 4))
    np.save(path, np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError): p.array(path, (4, 3))
    np.save(path, np.array([{}], dtype=object))
    with pytest.raises(ValueError): p.array(path, (1,))


class FakeModelRuntime:
    """No image decode, tensor library or real model; tests persistence only."""
    def __init__(self, config): self.config = config
    def release(self): pass
    def load(self, *args): pass
    def project(self, *args): return [], 0, []
    def select(self, *args): return list(range(78643)), []
    def assemble(self, *args): pass
    def save(self, path, context): path.write_bytes(b"FAKE_NOT_A_MODEL")


def fake_score(archive, photo, runtime, recovered=None):
    value = float(int(photo["photo_id"])+1)
    sample = {k: photo[k] for k in ("key", "photo_id", "label")}
    sample["regions"] = [{"region": r["region"], "member": r["member"], "top1": value, "native_raw": value}
                         for r in photo["regions"]]
    sample["photo_seconds"] = 1.
    return sample, np.full((12, 256, 256), value, np.float32)


def test_fit_save_interrupt_resume_reload_export_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "Monitor", Monitor)
    monkeypatch.setattr(run, "score_photo", fake_score)
    # Tiny fixture changes only the required normal count; production count checks are tested separately.
    monkeypatch.setattr(p.comparison.c, "calibrate", lambda development, calibration:
        method({"development": development, "calibration": calibration}))
    raw = data()
    cohorts = {g: raw[g][:3] for g in ("development", "calibration")}
    reference = {g: [fake_score(None, photo, None)[0] for photo in photos] for g, photos in cohorts.items()}
    source = {"cohorts": cohorts, "reference": reference, "config": {},
              "records": {"fit.json": {"checkpoint_context": {}}},
              "contract": {"candidate_top1_q90": method(reference)}}
    identity = {"fake": True, "training": {}, "cohorts": cohorts}
    root = tmp_path/"fit"
    archive = tmp_path/"empty.zip"
    with zipfile.ZipFile(archive, "w"): pass
    def extract(folder, signature, *args):
        p.publish(folder, "feature-receipts", signature, [])
        return []
    monkeypatch.setattr(run, "extract", extract)
    collect = run.collect
    def interrupted(*args, **kwargs):
        result = collect(*args, **kwargs)
        if args[1] == "first-calibration": raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(run, "collect", interrupted)
    with pytest.raises(KeyboardInterrupt): run.fit_run(tmp_path, root, archive, source, identity, FakeModelRuntime)
    sha = p.fit.sha256_file(root/"model/model.pt")
    monkeypatch.setattr(run, "collect", collect)
    report = run.fit_run(tmp_path, root, archive, source, identity, FakeModelRuntime)
    assert report["checkpoint_sha256"] == sha and report["reload"]["photos"] == 6
    assert report["test_images_decoded"] == 0 and report["reload"]["decision_changes"] == 0
    output = tmp_path/"handoff.zip"
    run.export(root, output, identity, report)
    before = output.read_bytes()
    class Never(FakeModelRuntime):
        def __init__(self, *args): pytest.fail("Completed fit should not create runtime")
    assert run.fit_run(tmp_path, root, archive, source, identity, Never) == report
    run.export(root, output, identity, report)
    assert output.read_bytes() == before
    with zipfile.ZipFile(output) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        assert all(run.hashlib.sha256(z.read(k)).hexdigest() == v for k, v in manifest.items())
        assert all(k.endswith(".json") for k in z.namelist())
    (root/"model/model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError): run.fit_run(tmp_path, root, archive, source, identity, Never)


def test_evaluation_orchestration_does_not_stop_for_zero_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "Monitor", Monitor)
    raw, candidate = data(), data()
    for s in candidate["test"]:
        for r in s["regions"]: r["top1"] = 0.
    cohorts = {g: raw[g] for g in raw}
    identity, root = {"fake": True}, tmp_path/"fit"
    signature = p.fit.digest(identity)
    root.mkdir()
    with p.fit.phase_stage(root, "model") as stage:
        (stage/"model.pt").write_bytes(b"fake")
        p.fit.write_json(stage/"context.json", {})
        receipt = p.fit.seal_phase(stage, "model", signature, {}, ["model.pt", "context.json"])
    normal = {g: raw[g] for g in ("development", "calibration")}
    contract = method(raw)
    p.publish(root, "contract", signature, contract)
    p.publish(root, "reload-check", signature, p.reload_check(normal, normal, contract))
    report = {"checkpoint_sha256": receipt["files"]["model.pt"], "contract": contract, "context": {}}
    source = {"cohorts": cohorts, "reference": raw, "contract": {"candidate_top1_q90": contract},
        "config": {}, "previous": {}, "records": {"fit.json": {"checkpoint_context": {}}}}
    stages = []
    monkeypatch.setattr(run, "read_normals", lambda *a, **k: normal)
    monkeypatch.setattr(run, "normal_gate", lambda *a, **k: stages.append("gate") or {"passed": True})
    def collect(root, group, *a, **k):
        assert stages[0] == "gate"
        stages.append(group)
        return candidate[group]
    monkeypatch.setattr(run, "collect", collect)
    monkeypatch.setattr(evaluation, "warm", lambda *a: stages.append("warm") or {"fake": True})
    monkeypatch.setattr(evaluation, "location", lambda *a: stages.append("location") or {"fake": True})
    archive = tmp_path/"empty.zip"
    with zipfile.ZipFile(archive, "w"): pass
    result = evaluation.evaluate(tmp_path, root, tmp_path/"eval", archive, source, identity, report, FakeModelRuntime)
    assert stages == ["gate", "test", "validation", "warm", "warm", "location"]
    assert result["counts"]["candidate"]["tp"] == 0 and not result["accuracy_targets_passed"]
    assert result["stability_draws"] == 600 and result["warm_evaluated"] and result["location_evaluated"]
    assert not result["service_model_changed"] and not result["automatic_pass_allowed"]
