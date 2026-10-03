import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mpi_knn3_screen_runtime as rt
from scripts import run_mpi_knn3_screen as r
from tests.test_mpi_normal_diversity_fit import scored


class Tensor:
    """Small CPU stand-in. Not evidence of CUDA performance or library compatibility."""
    def __init__(self, x): self.x = np.asarray(x)
    def min(self, dim): return self.x.min(axis=dim), self.x.argmin(axis=dim)
    def topk(self, k, largest, dim):
        assert not largest
        return SimpleNamespace(values=Tensor(np.sort(self.x, axis=dim)[:, :k]))
    def mean(self, dim): return self.x.mean(axis=dim)


def test_distance_mean3_preserves_magnitude_and_native_tie_index():
    distances = np.array([[2, 2, 5, 8], [0, 4, 5, 9]], np.float32)
    nearest, mean = rt.dual_distances(Tensor(distances))
    np.testing.assert_array_equal(nearest[0], [2, 0])
    np.testing.assert_array_equal(nearest[1], [0, 0])
    np.testing.assert_array_equal(mean, [3, 3])
    _, scaled = rt.dual_distances(Tensor(distances * 7))
    np.testing.assert_allclose(scaled, mean * 7)
    np.testing.assert_array_equal(distances, [[2, 2, 5, 8], [0, 4, 5, 9]])


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_hook_only_intercepts_patch_search_restores_even_on_error(fail, override):
    class Core:
        memory_bank = object()
        def nearest_neighbors(self, embedding, n_neighbors): return "weighted9"
        def euclidean_dist(self, embedding, bank):
            assert bank is self.memory_bank
            return Tensor([[1, 2, 3]])
    core = Core()
    if override: core.nearest_neighbors = lambda *a: "override9"
    original = core.nearest_neighbors
    captured = []
    try:
        with rt.capture_mean_three(core, captured):
            assert core.nearest_neighbors(None, 9) == ("override9" if override else "weighted9")
            core.nearest_neighbors(SimpleNamespace(shape=(4096, 768)), 1)
            np.testing.assert_array_equal(captured[0], [2])
            if fail: raise RuntimeError("simulate")
    except RuntimeError:
        assert fail
    assert core.nearest_neighbors == original
    assert ("nearest_neighbors" in vars(core)) == override
    with pytest.raises(ValueError, match="query grid"):
        with rt.capture_mean_three(core, []):
            core.nearest_neighbors(SimpleNamespace(shape=(1, 768)), 1)


def test_activate_preserves_raw_bank_and_never_installs_unit_normalizer():
    runtime = object.__new__(rt.MeanThreeRuntime)
    core = SimpleNamespace(training=False, feature_extractor=SimpleNamespace(training=False), tiler=None,
                           memory_bank=object())
    runtime.model = SimpleNamespace(model=core)
    runtime.bank_fingerprint = lambda: {"shape": [78643, 768], "float32_le_sha256": rt.BANK_SHA}
    before = vars(core).copy()
    result = runtime.activate()
    assert result["query_and_bank_normalized"] is False and vars(core) == before
    runtime.bank_fingerprint = lambda: {"float32_le_sha256": "wrong"}
    with pytest.raises(ValueError, match="bank changed"): runtime.activate()


def test_policy_pin_cli_and_selection_is_order_independent():
    policy = json.loads((r.ROOT / "configs/mpi-knn3-screen-v1.json").read_bytes())
    assert r.fit.digest(policy) == r.POLICY_SHA256
    normal = {"development": scored("development", 64, 128), "calibration": scored("calibration", 192, 169)}
    contract = r.comparison.c.calibrate(**normal)
    test = scored("defect", 0, 55)
    for p in test:
        p["label"] = 1
        for row in p["regions"]: row["top1"] = 0 if p["photo_id"] in ("0019", "0024") else 1000
    source = {"reference": {"test": test}, "cohorts": {"test": test}, "contract": {"candidate_top1_q90": contract}}
    assert [p["key"] for p in r.select_regression(source, policy)] == policy["regression_keys"]
    source["reference"]["test"] = list(reversed(test))
    assert [p["key"] for p in r.select_regression(source, policy)] == policy["regression_keys"]
    policy["regression_keys"] = list(reversed(policy["regression_keys"]))
    with pytest.raises(ValueError, match="selection"): r.select_regression(source, policy)
    with pytest.raises(SystemExit) as exc: r.main(["--help"])
    assert exc.value.code == 0


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    raw = {"development": scored("development", 64, 128), "calibration": scored("calibration", 192, 169)}
    raw["test"] = scored("defect", 0, 12)
    for p in raw["test"]:
        p["label"] = 1
        for row in p["regions"]: row["top1"] = 1000
    groups = {g: raw[g] for g in ("development", "calibration")}
    groups["regression"] = raw["test"]
    groups["reload"] = raw["calibration"][:8] + raw["test"]
    contract = r.comparison.c.calibrate(raw["development"], raw["calibration"])
    env = {"kind": "fake_cpu_not_cuda_evidence"}
    source = {"reference": raw, "config": {}, "preparation": {}, "previous": {},
              "contract": {"candidate_top1_q90": contract},
              "records": {"fit.json": {"checkpoint_context": {}}, "reload.json": {"receipt": {"runtime": env}, "photos": []}}}
    policy = json.loads((r.ROOT / "configs/mpi-knn3-screen-v1.json").read_bytes())
    policy["regression_keys"] = [p["key"] for p in raw["test"]]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake-no-torch")
    identity = {"groups": groups, "policy": policy, "checkpoint_sha256": r.fit.sha256_file(checkpoint)}
    state = {"scores": [], "loads": 0, "lost": False, "drift": False, "fail_at": None, "budget": False,
             "baseline_drift": False, "env_drift": False, "probe_failure": False}

    class Runtime:
        def __init__(self, *a): self.loads = 0
        def load(self, *a): state["loads"] += 1; self.loads += 1
        def environment(self): return {} if state["env_drift"] else env
        def release(self): pass
        def activate(self): return {"bank": "unchanged", "mean3": True}

    class Monitor:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def check(self):
            if state["budget"]: raise RuntimeError("memory cap exceeded")
        def snapshot(self): return {"kind": "fake"}

    def score(archive, photo, expected, runtime, recovery, monitor, method):
        monitor.check()
        if len(state["scores"]) == state["fail_at"]: raise RuntimeError("interrupted")
        if photo["key"].startswith("defect/"):
            assert (tmp_path / "run/calibrated/contract.json").is_file()
        if runtime.loads >= 2: assert (tmp_path / "run/screen-gate/gate.json").is_file()
        sample = copy.deepcopy(photo)
        sample["baseline_replay"] = copy.deepcopy(expected)
        if state["baseline_drift"]: sample["baseline_replay"]["regions"][0]["top1"] += 1
        r.replay_check(sample["baseline_replay"], expected, method)
        if state["lost"] and photo["key"] == "defect/0000":
            for row in sample["regions"]: row["top1"] = 0
        if state["drift"] and runtime.loads >= 2: sample["regions"][0]["top1"] += 1
        state["scores"].append(photo["key"])
        return sample

    monkeypatch.setattr(r, "score_photo", score)
    monkeypatch.setattr(r.ev, "normal_probe", lambda *a: {"passed": not state["probe_failure"]})
    monkeypatch.setattr(r.ev, "recovered_bytes", lambda *a: {})
    monkeypatch.setattr(r.ev, "read_masks", lambda *a: pytest.fail("Masks must not be read"))

    def execute():
        return r.execute(tmp_path / "run", None, checkpoint, tmp_path / "recovery.png", source, policy,
                         identity, tmp_path / "handoff.zip", Runtime, Monitor)
    return tmp_path, state, execute


def test_success_small_scope_manifest_and_completed_rerun_without_gpu(pipeline):
    root, state, execute = pipeline
    report = execute()
    assert len(state["scores"]) == report["scored_photos"] == 329
    assert state["loads"] == 2
    assert report["screen"]["retained"] == 12 and report["reload"]["photos"] == 20
    assert not report["false_positive_rate_evaluated"] and not report["full_evaluation_run"]
    before = copy.deepcopy(state)
    assert execute() == report and before == state
    with r.zipfile.ZipFile(root / "handoff.zip") as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        for name, sha in manifest.items(): assert r.hashlib.sha256(z.read(name)).hexdigest() == sha


def test_one_lost_detection_stops_without_reload_or_full_evaluation(pipeline):
    root, state, execute = pipeline
    state["lost"] = True
    report = execute()
    assert report["decision"] == "stop_candidate" and report["screen"]["lost_keys"] == ["defect/0000"]
    assert len(state["scores"]) == 309 and state["loads"] == 1
    assert report["reload"]["status"] == "not_run_screen_failed"


def test_resume_reuses_completed_photo_but_rechecks_environment(pipeline):
    root, state, execute = pipeline
    state["fail_at"] = 3
    with pytest.raises(RuntimeError, match="interrupted"): execute()
    assert len(state["scores"]) == 3
    state["fail_at"] = None
    report = execute()
    assert len(state["scores"]) == 329 and report["attempts_with_errors"] == 1


@pytest.mark.parametrize("kind", ["env_drift", "probe_failure", "baseline_drift", "drift"])
def test_runtime_or_replay_errors_fail_closed(pipeline, kind):
    root, state, execute = pipeline
    state[kind] = True
    with pytest.raises(ValueError): execute()
    assert not (root / "handoff.zip").exists()
    if kind != "drift": assert not state["scores"]


def test_resource_failure_not_automatically_retried(pipeline):
    root, state, execute = pipeline
    state["budget"] = True
    with pytest.raises(RuntimeError, match="cap exceeded"): execute()
    state["budget"] = False
    with pytest.raises(ValueError, match="resource violation"): execute()
    assert not state["scores"]


@pytest.mark.parametrize("target", ["sample", "gate"])
def test_cache_tampering_fails_even_after_completed_result(pipeline, target):
    root, state, execute = pipeline
    execute()
    path = (root / "run/development/photo-000/sample.json" if target == "sample"
            else next((root / "run/attempts").glob("*/gate-*.json")))
    path.write_text("{}")
    with pytest.raises(ValueError): execute()


def test_real_scoring_checks_baseline_and_closes_images(monkeypatch):
    photo = scored("development", 64, 1)[0]
    closed = []
    class Image:
        def close(self): closed.append(True)
    monkeypatch.setattr(r.fit, "checked_image", lambda *a: Image())
    monkeypatch.setattr(r.fit, "candidate_image_score", lambda raw, values, c: float(values[0, 0]))
    class Runtime:
        n = 0
        def synchronize(self): pass
        def predict_pair(self, image):
            self.n += 1
            return float(self.n), np.full((256,256), self.n), np.full((256,256), self.n+2)
    method = {"field": "top1", "threshold": 20, "regions": [{"region": x, "center": 0, "scale": 1} for x in r.fit.REGIONS]}
    result = r.score_photo(None, photo, photo, Runtime(), None, SimpleNamespace(check=lambda: None), method)
    assert result["regions"][0]["top1"] == 3 and result["baseline_replay"]["regions"][0]["top1"] == 1
    assert len(closed) == 12
