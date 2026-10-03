import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import run_mpi_efficientad_probe as r
from scripts import mpi_efficientad_probe_assets as a


def member(name, kind=tarfile.REGTYPE):
    m = tarfile.TarInfo(name)
    m.size, m.type = 10, kind
    return m


def test_aux_selection_fixed_order_train_only():
    members = [member(f"imagenette2/train/n0/{i}.JPEG") for i in range(120)]
    members += [member("imagenette2/val/n0/ignore.JPEG")]
    names = a.auxiliary_names(members)
    assert len(names) == 96 and all("/train/" in n for n in names)
    assert names == a.auxiliary_names(reversed(members))


@pytest.mark.parametrize("bad", [member("../outside"), member("/outside"), member("a\\b"),
                                member("link", tarfile.SYMTYPE), member("link", tarfile.LNKTYPE)])
def test_aux_archive_safety(bad):
    with pytest.raises(ValueError, match="Unsafe"):
        a.auxiliary_names([bad])


def test_aux_duplicate_and_insufficient():
    m = member("imagenette2/train/n/one.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        a.auxiliary_names([m, m])
    with pytest.raises(ValueError, match="Too few"):
        a.auxiliary_names([m])


class Response(io.BytesIO):
    def geturl(self): return "https://example.org/asset"


def test_download_pinned_reuse_and_failure_preserves(tmp_path):
    spec = {"filename": "asset.bin", "url": "https://example.org/asset",
            "sha256": r.hashlib.sha256(b"ok").hexdigest(), "maximum_bytes": 10}
    with pytest.raises(ValueError, match="requires"):
        a.download(tmp_path, spec, False)
    path = a.download(tmp_path, spec, True, lambda *a, **k: Response(b"ok"))
    assert path.read_bytes() == b"ok"
    assert a.download(tmp_path, spec, False) == path
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="preserved"):
        a.download(tmp_path, spec, True)
    assert path.read_bytes() == b"wrong"


def test_bad_download_is_not_published(tmp_path):
    spec = {"filename": "asset.bin", "url": "https://example.org/asset", "sha256": "0"*64, "maximum_bytes": 2}
    with pytest.raises(RuntimeError, match="limit"):
        a.download(tmp_path, spec, True, lambda *a, **k: Response(b"too large"))
    assert not (tmp_path / "asset.bin").exists()
    assert list(tmp_path.glob("download-*/asset.bin"))


def test_asset_prepare_streams_selected_images_and_reuses_receipts(tmp_path, monkeypatch):
    teacher = tmp_path / "weights.zip"
    with zipfile.ZipFile(teacher, "w") as z:
        z.writestr("official/pretrained_teacher_small.pth", b"fake-weights-never-loaded")
        z.writestr("official/pretrained_teacher_medium.pth", b"not-selected")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    data = tmp_path / "imagenette.tgz"
    with tarfile.open(data, "w:gz") as z:
        for i in range(100):
            m = member(f"imagenette2/train/n0/{i}.JPEG")
            m.size = len(jpeg)
            z.addfile(m, io.BytesIO(jpeg))
        m = member("imagenette2/val/n0/never-read.jpg")
        m.size = 7
        z.addfile(m, io.BytesIO(b"invalid"))
    specs = {name: {"filename": path.name, "sha256": r.fit.sha256_file(path),
                    "url": "https://example.org", "maximum_bytes": 1024**2}
             for name, path in (("teacher", teacher), ("imagenette", data))}
    monkeypatch.setattr(a, "ASSETS", specs)
    root, manifest, _ = a.prepare(tmp_path, False)
    assert len(manifest["auxiliary"]) == 96
    assert (root / "teacher.pth").read_bytes() == b"fake-weights-never-loaded"
    assert a.prepare(tmp_path, False)[1] == manifest
    (root / "aux-000.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="incomplete or changed"):
        a.prepare(tmp_path, False)


class FakeRuntime:
    """CPU fake tests orchestration only, not EfficientAD/CUDA compatibility or accuracy."""
    interrupt = None
    predictions = 0
    def __init__(self, *args): self.value = 0
    def statistics(self, images, check):
        for image in images:
            image.close()
            check()
        return {"fake": True}
    def step(self, image, index):
        if self.interrupt == index: raise KeyboardInterrupt()
        self.value += 1
        return {"step": index+1, "fake": True}
    def predict(self, image):
        self.predictions += 1
        return np.full((1, 1, 256, 256), self.value, np.float32)
    def save(self, path, signature, step):
        path.write_text(json.dumps({"step": step, "identity": signature, "value": self.value}))
    def load(self, path, signature, step):
        state = json.loads(path.read_text())
        assert state["identity"] == signature and state["step"] == step
        self.value = state["value"]


class FakeMonitor:
    def __init__(self, *args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"fake_monitor_not_cuda": True}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w"): pass
    seen = []
    def checked(z, row):
        assert row["member"].startswith("train/good/")
        seen.append(row["member"])
        return Image.new("RGB", (256, 256))
    monkeypatch.setattr(r.fit, "checked_image", checked)
    identity = {"rows": [{"member": f"train/good/{i}"} for i in range(96)], "environment": {"fake": True}}
    return tmp_path / "run", archive, identity, tmp_path, tmp_path / "handoff.zip", seen


def run(inputs, cls=FakeRuntime):
    return r.execute(*inputs[:5], runtime_factory=cls, monitor_factory=FakeMonitor)


def test_probe_exact100_reload_export_and_idempotent(inputs):
    report = run(inputs)
    assert report["completed_updates"] == report["cumulative_attempted_updates"] == 100
    assert report["accuracy_measured"] is False and report["test_images_decoded"] == 0
    assert [a["step"] for a in report["reload_checks"]] == [50, 100]
    assert len(set(inputs[-1])) == 96
    count = len(inputs[-1])
    assert run(inputs) == report and len(inputs[-1]) == count
    with zipfile.ZipFile(inputs[4]) as z:
        assert all(n.endswith(".json") for n in z.namelist())
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert all(r.hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())


def test_changed_identity_and_checkpoint_rejected(inputs):
    run(inputs)
    changed = {**inputs[2], "changed": True}
    with pytest.raises(ValueError):
        r.execute(inputs[0], inputs[1], changed, inputs[3], inputs[4], FakeRuntime, FakeMonitor)
    (inputs[0] / "step-050/model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="incomplete or changed"):
        run(inputs)


def test_interruption_does_not_allow_more_than100_attempted_updates(inputs):
    class Interrupted(FakeRuntime): interrupt = 25
    with pytest.raises(KeyboardInterrupt): run(inputs, Interrupted)
    assert (inputs[0] / "step-025/model.pt").exists()
    # The interrupted attempted step is counted conservatively; it cannot silently extend the cap.
    with pytest.raises(RuntimeError, match="attempted-update cap"):
        run(inputs)
    assert not inputs[4].exists()
    with pytest.raises(ValueError, match="Prior failed"):
        run(inputs)


def test_failed_reload_stops_without_export(inputs):
    class BadReload(FakeRuntime):
        def load(self, *args):
            super().load(*args)
            self.value += 1
    with pytest.raises(ValueError, match="reload map difference"):
        run(inputs, BadReload)
    assert (inputs[0] / "step-050/model.pt").exists()
    assert not (inputs[0] / "step-075").exists() and not inputs[4].exists()


def test_unfinished_attempt_fail_closed(inputs):
    root = inputs[0]
    root.mkdir()
    r.publish(root, "identity", r.fit.digest(inputs[2]), inputs[2])
    (root / "attempt-hard-kill").mkdir()
    with pytest.raises(ValueError, match="Unfinished"):
        run(inputs)


def test_resume_from_committed_boundary_completes_without_extra_updates(inputs):
    class BoundaryInterrupt(FakeMonitor):
        def check(self):
            if (inputs[0] / "step-025/receipt.json").exists():
                raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        r.execute(*inputs[:5], runtime_factory=FakeRuntime, monitor_factory=BoundaryInterrupt)
    assert (inputs[0] / "step-025/model.pt").exists()
    result = run(inputs)
    assert result["cumulative_attempted_updates"] == 100 and result["completed_updates"] == 100
    assert result["reload_checks"][0]["step"] == 25


def test_resource_failure_preserves_checkpoints_and_blocks_retry(inputs):
    class CapMonitor(FakeMonitor):
        def check(self):
            if (inputs[0] / "step-025").exists():
                raise RuntimeError("simulated resource cap")
    with pytest.raises(RuntimeError, match="resource cap"):
        r.execute(*inputs[:5], runtime_factory=FakeRuntime, monitor_factory=CapMonitor)
    assert (inputs[0] / "step-025/model.pt").exists() and not inputs[4].exists()
    with pytest.raises(ValueError, match="Prior failed"):
        run(inputs)


def test_source_contract_reuses_reviewed_training_only():
    path = Path.home() / "Downloads/mpi-pcb-normal-bank-preparation-v1.zip"
    if not path.exists(): pytest.skip("Reviewed local handoff unavailable")
    plan = r.prepare(r.ROOT, path)
    assert len(plan["photo_ids"]) == 8 and len(plan["rows"]) == 96
    assert all("/train/good/" in row["member"] for row in plan["rows"])
    assert len({row["decoded_rgb_sha256"] for row in plan["rows"]}) == 96


def test_cli_help_and_policy_no_full_training(capsys):
    with pytest.raises(SystemExit) as e:
        r.main(["--help"])
    assert e.value.code == 0 and "--allow-download" in capsys.readouterr().out
    assert r.POLICY["optimizer_updates"] == 100
    assert r.POLICY["full_training_authorized"] is False
