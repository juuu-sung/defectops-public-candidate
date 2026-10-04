"""CPU orchestration tests. Fakes are NOT evidence of CUDA feasibility or accuracy."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts import run_mpi_supersimplenet_probe as r
from scripts import mpi_supersimplenet_protocol as p


def test_schedule_fixed_batch8_100_updates_covers_all96():
    first = [i for step in range(12) for i in p.batch_indices(step)]
    assert sorted(first) == list(range(96))
    assert p.batch_indices(0) == p.batch_indices(0)
    assert p.batch_indices(12) != p.batch_indices(0)
    assert sum(len(p.batch_indices(i)) for i in range(100)) == 800
    for i in (-1, 100):
        with pytest.raises(ValueError): p.batch_indices(i)


def test_signed_raw_logits_and_scalar_batch1_are_preserved():
    values = p.validate_prediction(np.array(-2.5), np.full((1, 1, 256, 256), -3.), 1)
    assert values.shape == (1, 65537) and values[0, 0] == -2.5 and values[0, 1] == -3
    assert p.validate_prediction(np.arange(8), np.zeros((8, 1, 256, 256)), 8).shape == (8, 65537)


@pytest.mark.parametrize("scores,maps,n", [
    (np.array(0), np.zeros((8, 1, 256, 256)), 8),
    (np.zeros((1, 1)), np.zeros((1, 1, 256, 256)), 1),
    (np.array(0), np.zeros((1, 256, 256)), 1),
    (np.array(float("nan")), np.zeros((1, 1, 256, 256)), 1),
    (np.array(0), np.full((1, 1, 256, 256), np.inf), 1),
])
def test_prediction_rejects_bad_shapes_and_nonfinite(scores, maps, n):
    with pytest.raises(ValueError): p.validate_prediction(scores, maps, n)


def test_replay_checks_both_heads():
    good = np.zeros((1, 65537))
    assert p.replay_difference(good, good)["maximum_raw_score_difference"] == 0
    for column in (0, 127):
        bad = good.copy()
        bad[0, column] = .001
        with pytest.raises(ValueError, match="exceeds"): p.replay_difference(bad, good)


def test_source_pin_accepts_crlf_rejects_edits(tmp_path, monkeypatch):
    data = b"hello\nworld\n"
    monkeypatch.setattr(p, "SOURCE_HASHES", {"model.py": r.hashlib.sha256(data).hexdigest()})
    (tmp_path / "model.py").write_bytes(data.replace(b"\n", b"\r\n"))
    assert p.verify_sources(tmp_path) == p.SOURCE_HASHES
    (tmp_path / "model.py").write_bytes(b"different")
    with pytest.raises(ValueError, match="implementation differs"): p.verify_sources(tmp_path)


class FakeRuntime:
    interrupt = None
    def __init__(self, *args): self.value = 0
    def fingerprints(self):
        return {"backbone": "frozen", "adaptor": str(self.value),
                "segmentation": str(self.value), "classification": str(self.value)}
    def step(self, images, index):
        assert len(images) == 8
        if self.interrupt == index: raise KeyboardInterrupt()
        self.value += 1
        return {"step": index+1, "synthetic_anomalous_samples": 8, "fake_not_accuracy": True}
    def predict(self, images): return np.full((len(images), 65537), -self.value, dtype=np.float32)
    def save(self, path, signature, step):
        path.write_text(json.dumps({"identity": signature, "step": step, "value": self.value}))
    def load(self, path, signature, step):
        payload = json.loads(path.read_text())
        assert payload["identity"] == signature and payload["step"] == step
        self.value = payload["value"]
    def memory(self):
        return {"torch_cuda_peak_allocated_bytes": 0, "torch_cuda_peak_reserved_bytes": 0}


class FakeMonitor:
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake_monitor_not_cuda": True}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    # Test entries deliberately unreadable. An accidental read would be a test failure.
    with zipfile.ZipFile(archive, "w") as z: z.writestr("test/defect/never.png", b"invalid")
    seen = []
    def checked(source, row):
        assert row["member"].startswith("train/good/")
        seen.append(row["member"])
        return Image.new("RGB", (256, 256))
    monkeypatch.setattr(r.fit, "checked_image", checked)
    identity = {"policy": p.POLICY, "rows": [{"member": f"train/good/{i}"} for i in range(96)],
                "environment": {"fake": True}}
    return tmp_path / "run", archive, identity, tmp_path / "weights", tmp_path / "handoff.zip", seen


def execute(inputs, runtime=FakeRuntime, monitor=FakeMonitor):
    return r.execute(*inputs[:5], runtime_factory=runtime, monitor_factory=monitor)


def test_complete_reload_json_only_idempotent(inputs):
    result = execute(inputs)
    assert result["completed_updates"] == result["cumulative_attempted_updates"] == 100
    assert result["formal_training_started"] is result["accuracy_measured"] is False
    assert result["test_images_decoded"] == result["test_masks_decoded"] == 0
    assert result["checks"]["backbone_unchanged"] is True
    assert [x["step"] for x in result["reload_checks"]] == [50, 100]
    assert len(set(inputs[-1])) == 96
    count = len(inputs[-1])
    assert execute(inputs) == result and len(inputs[-1]) == count
    with zipfile.ZipFile(inputs[4]) as z:
        assert all(x.endswith(".json") for x in z.namelist())
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert all(r.hashlib.sha256(z.read(k)).hexdigest() == v for k, v in manifest.items())


def test_changed_identity_and_checkpoint_preserved(inputs):
    execute(inputs)
    inputs[2]["changed"] = True
    with pytest.raises(ValueError): execute(inputs)
    del inputs[2]["changed"]
    (inputs[0] / "step-050/model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="incomplete or changed"): execute(inputs)


def test_changed_score_after_reload_blocks_export(inputs):
    class Wrong(FakeRuntime):
        def load(self, *args):
            super().load(*args)
            self.value += 1
    with pytest.raises(ValueError, match="Reload"): execute(inputs, Wrong)
    assert (inputs[0] / "step-050").exists() and not inputs[4].exists()
    with pytest.raises(ValueError, match="Prior failed"): execute(inputs)


def test_boundary_interrupt_resumes_exact100_updates(inputs):
    class Boundary(FakeMonitor):
        def check(self):
            if (inputs[0] / "step-025/receipt.json").exists(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): execute(inputs, monitor=Boundary)
    result = execute(inputs)
    assert result["completed_updates"] == result["cumulative_attempted_updates"] == 100
    assert result["reload_checks"][0]["step"] == 25


def test_mid_update_interruption_does_not_extend_budget(inputs):
    class Interrupted(FakeRuntime): interrupt = 25
    with pytest.raises(KeyboardInterrupt): execute(inputs, Interrupted)
    with pytest.raises(RuntimeError, match="attempted-update cap"): execute(inputs)
    assert not inputs[4].exists()


def test_resource_failure_blocks_retry_and_preserves(inputs):
    class Limited(FakeMonitor):
        def check(self):
            if (inputs[0] / "step-025").exists(): raise RuntimeError("resource limit")
    with pytest.raises(RuntimeError, match="resource limit"): execute(inputs, monitor=Limited)
    assert (inputs[0] / "step-025/model.pt").exists()
    with pytest.raises(ValueError, match="Prior failed"): execute(inputs)


def test_initialization_failure_keeps_diagnostic_and_blocks_retry(inputs):
    class BadInit(FakeRuntime):
        def __init__(self, *args): raise RuntimeError("simulated constructor OOM")
    with pytest.raises(RuntimeError, match="constructor OOM"): execute(inputs, BadInit)
    records = list(inputs[0].glob("attempt-*/finished.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_bytes())["attempted_updates"] == 0
    assert not inputs[4].exists()
    with pytest.raises(ValueError, match="Prior failed"): execute(inputs)


def test_backbone_change_is_not_a_success(inputs):
    class BadBackbone(FakeRuntime):
        def fingerprints(self): return {**super().fingerprints(), "backbone": str(self.value)}
    with pytest.raises(ValueError, match="Frozen backbone"): execute(inputs, BadBackbone)
    assert not inputs[4].exists()


def test_batch8_invalid_shape_is_not_a_success(inputs):
    class BadBatch(FakeRuntime):
        def predict(self, images): return super().predict(images)[:1]
    with pytest.raises(ValueError, match="Batch8 inference"): execute(inputs, BadBatch)
    assert not inputs[4].exists()


def test_no_synthetic_class_is_not_a_success(inputs):
    class BadSynthetic(FakeRuntime):
        def step(self, images, index):
            return {**super().step(images, index), "synthetic_anomalous_samples": 0}
    with pytest.raises(ValueError, match="both normal and synthetic"): execute(inputs, BadSynthetic)
    assert not inputs[4].exists()


def test_hard_kill_and_policy_changes_refused(inputs):
    inputs[0].mkdir(exist_ok=True)
    r.publish(inputs[0], "identity", r.fit.digest(inputs[2]), inputs[2])
    (inputs[0] / "attempt-hard-kill").mkdir()
    with pytest.raises(ValueError, match="Unfinished"): execute(inputs)
    inputs[2]["policy"] = {**p.POLICY, "batch_size": 4}
    with pytest.raises(ValueError, match="policy mismatch"): execute(inputs)


def test_source_cohort_preparation_is_pinned_train_only():
    path = Path.home() / "Downloads/mpi-pcb-normal-bank-preparation-v1.zip"
    if not path.exists(): pytest.skip("Reviewed handoff not available")
    plan = r.prepare(r.ROOT, path)
    assert len(plan["photo_ids"]) == 8 and len(plan["rows"]) == 96
    assert plan["photo_ids"] == ["1619", "0833", "1605", "1075", "0203", "1088", "0465", "0908"]
    assert all("/train/good/" in row["member"] for row in plan["rows"])
    assert len({row["decoded_rgb_sha256"] for row in plan["rows"]}) == 96


def test_cli_help_has_no_formal_fit_or_test_options(capsys):
    with pytest.raises(SystemExit) as exc: r.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--allow-download" in help_text and "--epochs" not in help_text
    assert p.POLICY["full_training_authorized"] is False
