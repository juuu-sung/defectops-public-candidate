"""CPU tests of the formal runner contract, not EfficientAD accuracy/CUDA tests."""
import copy
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_efficientad_fit_protocol as p
from scripts import mpi_efficientad_fit_runtime as rt
from scripts import run_mpi_efficientad_fit as r


def test_frozen_schedule_and_separate_epoch_orders():
    assert p.POLICY["updates"] == 70000
    assert p.learning_rate(66499) == 1e-4
    assert p.learning_rate(66500) == p.learning_rate(69999) == 1e-5
    a = [p.sample_index(i, 768, "normal") for i in range(768)]
    b = [p.sample_index(i+768, 768, "normal") for i in range(768)]
    assert sorted(a) == sorted(b) == list(range(768)) and a != b
    assert a != list(p.epoch_order(768, 0, "auxiliary"))
    assert p.sample_index(1521, 9469, "auxiliary") == p.sample_index(1521, 9469, "auxiliary")
    assert p.POLICY["selected_checkpoint"] == "final_70000_only"
    assert p.POLICY["future_evaluation"]["executed_by_this_runner"] is False


def test_raw_map_validation_and_signed_top1():
    raw = np.zeros((12, 2, 256, 256), dtype=np.float32)
    q = [{"q90": 1., "q995": 2.}] * 2
    assert np.allclose(p.tile_scores(raw, q), -.1)  # no accidental clipping
    raw[0, :, :3] = 2
    assert p.tile_scores(raw, q)[0] == pytest.approx(.1)
    for bad in [np.ones((2, 1, 256), np.float32), raw.astype(np.float64), raw - 3, raw * np.nan]:
        with pytest.raises(ValueError): p.validate_maps(bad)


def test_exact_map_quantiles_and_degenerate_stop():
    raw = np.arange(2*256*256, dtype=np.float32).reshape(1, 2, 256, 256)
    q = p.map_quantiles([raw, raw+100])
    for c in range(2):
        values = np.concatenate([raw[:, c].ravel(), (raw+100)[:, c].ravel()])
        assert [q[c]["q90"], q[c]["q995"]] == list(np.quantile(values, [.9, .995]))
    with pytest.raises(ValueError, match="Degenerate"):
        p.map_quantiles([np.zeros((1, 2, 256, 256), np.float32)])


def test_calibration_median_q90_q95_and_no_test():
    dev = np.arange(128)[:, None] + np.arange(12)[None, :] - 150.
    cal = np.arange(169)[:, None] + np.arange(12)[None, :] - 150.
    contract = p.calibrate(dev, cal, [{"q90": 1, "q995": 2}]*2)
    assert np.array_equal(contract["centers"], np.median(dev, axis=0))
    ordered = sorted(p.photo_scores(cal, contract))
    assert contract["threshold"] == ordered[int(np.ceil(.95*168))]
    assert contract["test_scores_used"] == 0
    with pytest.raises(ValueError, match="Incomplete"):
        p.calibrate(dev[:-1], cal, [])
    with pytest.raises(ValueError, match="Degenerate"):
        p.calibrate(np.zeros((128, 12)), cal, [])


def cohorts():
    result = {}
    offset = 0
    for g, n in (("training", 64), ("development", 128), ("calibration", 169)):
        ids = [f"{i:04d}" for i in range(offset, offset+n)]
        result[g] = {"photo_ids": ids, "rows": [
            {"photo_id": pid, "region": region, "member": f"extracted/{region}/train/good/{pid}.png",
             "decoded_rgb_sha256": f"{pid}/{region}"} for pid in ids for region in p.fit.REGIONS]}
        offset += n
    return result


def test_cohorts_disjoint_exact_pixels_and_normal_paths():
    c = cohorts()
    p.validate_cohorts(c)
    for field, value in [("member", "extracted/test/defect/x.png"),
                         ("decoded_rgb_sha256", c["training"]["rows"][0]["decoded_rgb_sha256"])]:
        bad = copy.deepcopy(c)
        bad["development"]["rows"][0][field] = value
        with pytest.raises(ValueError): p.validate_cohorts(bad)
    bad = copy.deepcopy(c)
    bad["development"]["photo_ids"][0] = c["training"]["photo_ids"][0]
    with pytest.raises(ValueError): p.validate_cohorts(bad)


def test_full_imagenette_inventory_train_only():
    def member(name):
        value = tarfile.TarInfo(name)
        value.size = 10
        return value
    members = [member(f"imagenette2/train/n/{i:05d}.JPEG") for i in range(9469)]
    assert len(rt.train_members(members + [member("imagenette2/val/n/unused.JPEG")])) == 9469
    for bad in [[member("../escape")], members[:-1], members+[members[0]]]:
        with pytest.raises(ValueError): rt.train_members(bad)


class FakeRuntime:
    interrupt = None
    drift = False
    def __init__(self, *args): self.step_value = 0; self.loaded = False
    def statistics(self, images, check):
        for image in images: image.close(); check()
        return {"fake": True}
    def step(self, image, index):
        if index == self.interrupt: raise KeyboardInterrupt()
        self.step_value = index+1
        return {"step": index+1, "loss_components": [1., 2., 3.], "gradient_l2": 1.}
    def predict(self, image):
        n = image.getpixel((0, 0))[0]
        return np.full((2, 256, 256), n+self.step_value+(1 if self.loaded and self.drift else 0), np.float32)
    def save(self, path, signature, step):
        path.write_text(json.dumps({"identity": signature, "step": step}))
    def load(self, path, signature, step):
        d = json.loads(path.read_text()); assert d == {"identity": signature, "step": step}
        self.step_value = step; self.loaded = True


class FakeMonitor:
    fail = False
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self):
        if self.fail: raise RuntimeError("resource limit fake")
    def snapshot(self): return {"fake": True}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    policy = copy.deepcopy(p.POLICY)
    policy.update(updates=4, checkpoint_interval=2, maximum_attempted_updates=6)
    groups = {}
    members = []
    for g, n in (("training", 1), ("development", 2), ("calibration", 3)):
        ids = [f"{i:04d}" for i in range(n)]
        rows = [{"member": f"train/good/{g}/{i}/{j}", "photo_id": ids[i], "region": j}
                for i in range(n) for j in range(12)]
        groups[g] = {"photo_ids": ids, "rows": rows}
        members += rows
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for row in members: z.writestr(row["member"], b"fake")
        z.writestr("test/defect/never-read", b"forbidden")
    seen = []
    def checked(z, row):
        assert row["member"].startswith("train/good/")
        seen.append(row["member"])
        return Image.new("RGB", (256, 256), (int(row["member"].split("/")[-2])+1, 0, 0))
    monkeypatch.setattr(r.fit, "checked_image", checked)
    monkeypatch.setattr(p, "map_quantiles", lambda maps, check: [{"q90": 0., "q995": 10.}]*2)
    monkeypatch.setattr(p, "calibrate", lambda d, c, q: {"map_quantiles": q, "centers": [0.]*12,
        "scales": [1.]*12, "threshold": .06})
    return (tmp_path / "run", archive, {"policy": policy, "cohorts": groups, "environment": {"fake": True}},
            tmp_path / "teacher", tmp_path / "aux", tmp_path / "handoff.zip", seen)


def execute(fixture, runtime=FakeRuntime, monitor=FakeMonitor):
    return r.execute(*fixture[:6], runtime_factory=runtime, monitor_factory=monitor)


def test_complete_fit_calibration_reload_idempotent_json_export(fixture):
    report = execute(fixture)
    assert report["completed_updates"] == 4 and report["test_images_decoded"] == 0
    count = len(fixture[-1])
    assert execute(fixture) == report and len(fixture[-1]) == count
    with zipfile.ZipFile(fixture[5]) as z:
        assert all(n.endswith(".json") for n in z.namelist())
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert all(r.hashlib.sha256(z.read(k)).hexdigest() == v for k, v in manifest.items())
        assert json.loads(z.read("reload.json"))["decision_changes"] == 0


def test_ctrl_c_resumes_last_checkpoint_and_counts_attempted_work(fixture):
    class Interrupted(FakeRuntime): interrupt = 3
    with pytest.raises(KeyboardInterrupt): execute(fixture, Interrupted)
    assert (fixture[0] / "step-000002/model.pt").exists()
    report = execute(fixture)
    assert report["completed_updates"] == 4 and report["cumulative_attempted_updates"] == 6


def test_failed_resource_attempt_is_not_automatically_retried(fixture):
    class Failing(FakeMonitor): fail = True
    with pytest.raises(RuntimeError, match="resource"): execute(fixture, monitor=Failing)
    with pytest.raises(ValueError, match="review"): execute(fixture)


def test_reload_drift_stops_and_preserves_checkpoints(fixture):
    class Drifting(FakeRuntime): drift = True
    with pytest.raises(ValueError, match="drift"): execute(fixture, Drifting)
    assert (fixture[0] / "step-000004/model.pt").exists()
    assert not fixture[5].exists()


def test_tampered_checkpoint_maps_and_identity_are_rejected(fixture):
    execute(fixture)
    identity = fixture[2]
    identity["changed"] = True
    with pytest.raises(ValueError): execute(fixture)
    identity.pop("changed")
    f = fixture[0] / "development/photo-000/maps.npy"
    f.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="incomplete or changed"): execute(fixture)


def test_unfinished_attempt_rejected(fixture):
    execute(fixture)
    directory = fixture[0] / "attempt-force-kill"
    directory.mkdir()
    with pytest.raises(ValueError, match="Unfinished"): execute(fixture)


def test_checkpoint_gap_rejected(fixture):
    execute(fixture)
    # Recoverable move in temporary test directory, never remove real output.
    (fixture[0] / "step-000002").rename(fixture[0] / ".saved-two")
    with pytest.raises(ValueError, match="gap"): execute(fixture)


def test_handoff_tamper_is_not_overwritten(fixture):
    execute(fixture)
    with zipfile.ZipFile(fixture[5], "w") as z: z.writestr("wrong.json", b"{}")
    before = fixture[5].read_bytes()
    with pytest.raises(ValueError, match="handoff differs"): execute(fixture)
    assert fixture[5].read_bytes() == before


def test_probe_gate_rejects_unreviewed_bundle(tmp_path):
    path = tmp_path / "probe.zip"
    path.write_bytes(b"not-the-reviewed-probe")
    with pytest.raises(ValueError, match="reviewed"): p.read_probe(path)


def test_resume_time_cap_checked_before_gpu(fixture):
    fixture[2]["policy"]["resource_plan"]["comparison_seconds"] = 0
    with pytest.raises(RuntimeError, match="time cap"): execute(fixture)
    assert not list(fixture[0].glob("attempt-*"))


def test_budget_does_not_reset_on_resume(fixture):
    class Interrupted(FakeRuntime): interrupt = 1
    fixture[2]["policy"]["maximum_attempted_updates"] = 4
    with pytest.raises(KeyboardInterrupt): execute(fixture, Interrupted)
    with pytest.raises(RuntimeError, match="attempted-update"): execute(fixture)
    assert not fixture[5].exists()


def test_actual_runtime_source_keeps_teacher_frozen_and_full_auxiliary_schedule():
    source = Path(rt.__file__).read_text()
    assert 'sample_index(index, 9469, "auxiliary")' in source
    assert 'self.model.teacher.eval()' in source
    assert 'no_grad()' in source
    assert 'weights_only=True' in source
    assert 'formal_efficientad_research_not_service' in source
    assert 'normalize=False' in source
