"""CPU arithmetic/control tests are not CUDA feasibility or model accuracy tests."""
import copy
import json
import os
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import mpi_patchcore_scaling_probe as p
from scripts import mpi_patchcore_scaling_probe_runtime as runtime
from scripts import run_mpi_patchcore_scaling_probe as runner


def test_locked_budget_no_candidate_reduction():
    a, b = p.budget(64), p.budget(256)
    assert a["candidates"] == 786432 and b["candidates"] == 3145728
    assert b["raw_float32_bytes"] == 9*1024**3
    assert a["raw_float32_bytes"] == int(2.25*1024**3)
    assert a["representatives"] == b["representatives"] == 78643
    assert b["bank_float32_bytes"] == 78643*768*4
    assert a["projected_dimension"] == 335 and b["projected_dimension"] == 369
    assert p.POLICY["time_limit_seconds"] is None
    assert p.POLICY["model_training_performed"] is False
    assert p.POLICY["resource_plan"]["selection_and_inference_reserved_GiB"] == 6
    with pytest.raises(ValueError): p.budget(128)


def test_chunks_cover_once_including_tail():
    for n in (1, 10, 2048, 3145728):
        chunks = p.slices(n, 127)
        assert chunks[0][0] == 0 and chunks[-1][1] == n
        assert sum(z-a for a,z in chunks) == n
        assert all(z == c for (_,z),(c,_) in zip(chunks, chunks[1:]))
    with pytest.raises(ValueError): p.slices(0, 10)
    with pytest.raises(ValueError): p.slices(1, 0)


class NumpyTorch:
    linalg = SimpleNamespace(norm=lambda x, ord, dim, keepdim: np.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim))
    minimum = staticmethod(np.minimum)
    max = staticmethod(lambda x, dim: SimpleNamespace(indices=np.argmax(x, axis=dim)))


@pytest.mark.parametrize("chunk", [1, 7, 127, 1024])
def test_chunk_recurrence_matches_full_distance_and_first_tie(chunk):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(129, 17)).astype(np.float32)
    features[7] = features[3]
    a = np.full((129,1), np.inf, np.float32); b = a.copy()
    x = y = 3
    for _ in range(60):
        a = np.minimum(a, np.linalg.norm(features-features[x], axis=1, keepdims=True))
        x = int(np.argmax(a[:,0])); a[x,0] = 0
        y = runtime.update_chunked(NumpyTorch, features, b, y, chunk)
        assert x == y
        np.testing.assert_array_equal(a, b)
    ties = np.array([[0.,0.],[1.,0.],[-1.,0.]], np.float32)
    assert runtime.update_chunked(NumpyTorch, ties, np.full((3,1),np.inf,np.float32), 0, 1) == 1


def test_changed_selection_is_not_tolerated():
    with pytest.raises(ValueError, match="differs"):
        p.validate_indices([1,2], [2,1])


class Monitor:
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake": True}


class FakeRuntime:
    calls = []
    def parity(self, check): self.calls.append("parity"); return {"passed": True, "fake": True}
    def scale(self, check): self.calls.append("scale"); return {"passed": True, "fake": True}
    def release(self): pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    FakeRuntime.calls = []
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=100*1024**3))
    root, handoff = tmp_path/"probe", tmp_path/"handoff.zip"
    identity = {"name": p.NAME, "policy": copy.deepcopy(p.POLICY)}
    def run(cls=FakeRuntime, env=lambda: {"fake": True}):
        return runner.execute(root, handoff, identity, cls, env, Monitor)
    return root, handoff, identity, run


def test_complete_and_cpu_only_reexport(setup):
    root, handoff, identity, run = setup
    result = run()
    assert FakeRuntime.calls == ["parity", "scale"]
    assert result["images_decoded"] == 0 and not result["full_fit_runner_ready"]
    original = handoff.read_bytes()
    class Never(FakeRuntime):
        def __init__(self): raise AssertionError("No GPU on completed replay")
    assert run(Never, lambda: pytest.fail("No live environment on completed replay")) == result
    assert handoff.read_bytes() == original
    with zipfile.ZipFile(handoff) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest)|{"handoff-manifest.json"}
        assert all(runner.hashlib.sha256(z.read(k)).hexdigest()==h for k,h in manifest.items())
        assert all(name.endswith('.json') for name in z.namelist())


def test_interrupt_resumes_completed_stage_with_same_identity(setup):
    root, _, _, run = setup
    class Stop(FakeRuntime):
        def scale(self, check): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): run(Stop)
    assert (root/"parity/receipt.json").exists()
    FakeRuntime.calls = []
    run()
    assert FakeRuntime.calls == ["scale"]
    records = runner.attempts(root)
    assert [x["status"] for x in records] == ["interrupted", "completed"] or sorted(x["status"] for x in records)==["completed","interrupted"]


def test_failed_attempt_blocks_retry_and_keeps_parity(setup):
    root, _, _, run = setup
    class Bad(FakeRuntime):
        def scale(self, check): raise RuntimeError("resource failure")
    with pytest.raises(RuntimeError): run(Bad)
    assert (root/"parity/receipt.json").exists()
    with pytest.raises(ValueError, match="failed/resource"): run()


def test_tampered_cached_phase_rejected_without_gpu(setup):
    root, _, _, run = setup
    run()
    (root/"parity/value.json").write_text('{}')
    with pytest.raises(ValueError, match="incomplete or changed"): run()


def test_changed_identity_preserved(setup):
    root, _, identity, run = setup
    run(); identity["extra"] = "changed"
    with pytest.raises(ValueError, match="incomplete or changed"): run()


def test_forced_termination_refused(setup):
    root, _, identity, run = setup
    root.mkdir()
    runner.publish(root, "identity", p.fit.digest(identity), identity)
    (root/"attempt-lost").mkdir()
    with pytest.raises(ValueError, match="Unfinished"): run()


def test_changed_environment_blocks_resume(setup):
    _, _, _, run = setup
    class Stop(FakeRuntime):
        def scale(self, check): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): run(Stop)
    with pytest.raises(ValueError, match="differs"): run(env=lambda: {"changed": True})


def test_installed_cuda_parity_when_available():
    if os.environ.get("DEFECTOPS_CUDA_TESTS") != "1":
        pytest.skip("explicit DEFECTOPS_CUDA_TESTS=1 required for real GPU work")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("requires CUDA, not a CPU equivalence claim")
    pytest.importorskip("anomalib")
    result = runtime.Runtime().parity(lambda: None)
    assert result["passed"] and len(result["cases"]) == 4
