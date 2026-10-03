import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mpi_unit_feature_runtime as runtime
from scripts import run_mpi_unit_feature as runner
from tests.test_mpi_normal_diversity_fit import photos, scored


FAKE_TORCH = SimpleNamespace(float32=np.dtype("float32"), isfinite=np.isfinite, empty_like=np.empty_like,
    linalg=SimpleNamespace(vector_norm=lambda x, dim, keepdim: np.linalg.norm(x, axis=dim, keepdims=keepdim)))


def test_unit_metric_scale_invariance_and_cosine_identity():
    x = np.array([[3, 4], [6, 8], [-4, 3]], dtype=np.float32)
    unit = runtime.unit_rows(x, FAKE_TORCH)
    np.testing.assert_array_equal(unit[0], unit[1])
    np.testing.assert_allclose(np.sum((unit[0]-unit[2])**2), 2-2*np.dot(unit[0], unit[2]), atol=5e-7)
    np.testing.assert_array_equal(x[0], [3, 4])


@pytest.mark.parametrize("value", [np.zeros((2, 3), np.float32), np.full((2, 3), np.inf, np.float32),
    np.full((2, 3), np.nan, np.float32), np.ones((2, 3), np.float64), np.ones(3, np.float32)])
def test_invalid_features_fail_closed(value):
    with pytest.raises(ValueError): runtime.unit_rows(value, FAKE_TORCH)


def test_both_bank_and_query_transformed_once_without_mutating_original():
    original = np.array([[3, 4], [6, 8]], np.float32)
    core = SimpleNamespace(memory_bank=original, reshape_embedding=lambda x: x.reshape(-1, 2))
    runtime.transform_core(core, FAKE_TORCH)
    np.testing.assert_array_equal(original, [[3, 4], [6, 8]])
    np.testing.assert_allclose(core.memory_bank, core.reshape_embedding(original))
    assert np.linalg.norm(core.memory_bank[0]-core.reshape_embedding(np.array([[9,12]],np.float32))[0]) == 0
    with pytest.raises(ValueError, match="already"): runtime.transform_core(core, FAKE_TORCH)


def test_policy_pin_and_cli():
    policy = json.loads((runner.ROOT / "configs/mpi-unit-feature-v1.json").read_bytes())
    assert runner.fit.digest(policy) == runner.POLICY_SHA256
    assert policy["resource_plan"]["comparison_seconds"] == 7200
    with pytest.raises(SystemExit) as exc: runner.main(["--help"])
    assert exc.value.code == 0


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    normal = {"development": scored("development", 64, 128), "calibration": scored("calibration", 192, 169)}
    reference = copy.deepcopy(normal)
    reference["test"] = scored("test", 0, 109)
    for i, p in enumerate(reference["test"]):
        p["label"] = int(i < 55)
        p["key"] = ("defect/" if p["label"] else "good/") + p["photo_id"]
    reference["validation"] = scored("validation", 400, 256)
    cohorts = copy.deepcopy(reference)
    contract = runner.comparison.c.calibrate(**normal)
    checkpoint = tmp_path / "original.pt"
    checkpoint.write_bytes(b"FAKE-not-a-torch-model")
    env = {"kind": "fake_cpu_not_cuda_evidence"}
    source = {"cohorts": cohorts, "reference": reference, "config": {}, "preparation": {}, "previous": {},
              "contract": {"candidate_top1_q90": contract},
              "records": {"fit.json": {"checkpoint_context": {}}, "reload.json": {"receipt": {"runtime": env}, "photos": []}}}
    policy = json.loads((runner.ROOT / "configs/mpi-unit-feature-v1.json").read_bytes())
    identity = {"policy": policy, "checkpoint_sha256": runner.fit.sha256_file(checkpoint)}
    state = {"loads": 0, "gate": 0, "scores": [], "fail_at": None, "budget": False, "drift": False}

    class Runtime:
        def __init__(self, cfg): self.active = False
        def load(self, *args): self.active = False; state["loads"] += 1
        def environment(self): return env
        def activate(self): self.active = True; return {"unit_bank": "synthetic-fixed"}
        def release(self): pass

    class Monitor:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def check(self):
            if state["budget"]: raise RuntimeError("cumulative time or memory cap exceeded")
        def snapshot(self): return {"kind": "fake"}

    def gate(*args):
        assert not args[-1].active
        state["gate"] += 1
        return {"passed": True}

    def score(archive, photo, rt, recovered, mon):
        assert rt.active
        mon.check()
        if len(state["scores"]) == state["fail_at"]: raise RuntimeError("simulated interruption")
        if photo["key"].startswith(("good/", "defect/")):
            assert (tmp_path / "run/reload-check/check.json").is_file()
        state["scores"].append(photo["key"])
        result = copy.deepcopy(photo)
        if state["drift"] and state["loads"] >= 2:
            result["regions"][0]["top1"] += 1
        return result

    monkeypatch.setattr(runner.ev, "normal_probe", gate)
    monkeypatch.setattr(runner.ev, "recovered_bytes", lambda *a: {})
    monkeypatch.setattr(runner, "score_photo", score)

    def execute():
        return runner.execute(tmp_path / "run", None, checkpoint, tmp_path / "recovery.png", source,
                              policy, identity, tmp_path / "handoff.zip", Runtime, Monitor)
    return tmp_path, state, execute


def test_full_order_early_stop_manifest_and_no_gpu_rerun(pipeline):
    root, state, execute = pipeline
    report = execute()
    assert len(state["scores"]) == 959
    assert state["loads"] == 2
    assert report["stability"]["status"] == "not_run_fixed_gate_failed"
    assert report["decision"] == "not_adopted" and not report["automatic_pass_allowed"]
    before = copy.deepcopy(state)
    assert execute() == report
    assert state == before
    with runner.zipfile.ZipFile(root / "handoff.zip") as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        for name, sha in manifest.items(): assert runner.hashlib.sha256(z.read(name)).hexdigest() == sha


def test_photo_cache_resume_no_repeat_completed_photos(pipeline):
    root, state, execute = pipeline
    state["fail_at"] = 3
    with pytest.raises(RuntimeError, match="interruption"): execute()
    assert len(state["scores"]) == 3
    state["fail_at"] = None
    result = execute()
    assert len(state["scores"]) == 959
    assert result["attempts_with_errors"] == 1


def test_reload_drift_blocks_observed_test(pipeline):
    root, state, execute = pipeline
    state["drift"] = True
    with pytest.raises(ValueError, match="Reload|reload"): execute()
    assert not any(key.startswith(("good/", "defect/")) for key in state["scores"])
    assert not (root / "handoff.zip").exists()


def test_resource_failure_preserved_not_automatically_retried(pipeline):
    root, state, execute = pipeline
    state["budget"] = True
    with pytest.raises(RuntimeError, match="cap exceeded"): execute()
    state["budget"] = False
    with pytest.raises(ValueError, match="resource violation"): execute()
    assert not state["scores"]


def test_tampered_photo_cache_blocks_even_completed_result(pipeline):
    root, state, execute = pipeline
    execute()
    (root / "run/development/photo-000/sample.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete or changed"): execute()


def test_unaccounted_forced_termination_blocks(pipeline):
    root, state, execute = pipeline
    orphan = root / "run/attempts/attempt-orphan"
    orphan.mkdir(parents=True)
    with pytest.raises(ValueError, match="without a cost receipt"): execute()
    assert not state["scores"]


def test_passing_point_gate_runs_stability_but_never_promotes(pipeline, monkeypatch):
    root, state, execute = pipeline
    monkeypatch.setattr(runner.comparison, "compare", lambda *a: ({"passed": True}, {"synthetic": True}))
    called = []
    def bootstrap(baseline, candidate, policy, check):
        check(); called.append(len(candidate["test"]))
        return {"passed": True, "summary": {"synthetic": "not scientific evidence"}}
    monkeypatch.setattr(runner.comparison, "bootstrap", bootstrap)
    result = execute()
    assert called == [109]
    assert result["decision"] == "needs_cost_review"
    assert not result["service_model_changed"] and not result["independent_performance_claim_allowed"]
