"""Synthetic CUDA boundary; real PNG checks, probe, caches, metrics and ZIP export."""

import copy
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import evaluate_mpi_pcb_normal_bank as ev
from tests.test_mpi_pcb_normal_bank_fit import png_row

PROJECT = Path(__file__).resolve().parents[1]


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class Runtime:
    def __init__(self, calls, offset=0, fail_after=None):
        self.calls, self.offset, self.fail_after, self.predictions = calls, offset, fail_after, 0
        calls.append("create")

    def load(self, checkpoint, context):
        self.calls.append("load")
        assert checkpoint.read_bytes() == b"synthetic checkpoint"

    def predict(self, image):
        self.predictions += 1
        self.calls.append("predict")
        if self.fail_after and self.predictions > self.fail_after:
            raise RuntimeError("synthetic interruption")
        score = float(image.getpixel((0, 0))[0] + self.offset)
        return score, np.full((256, 256), score, dtype=np.float32)

    def synchronize(self):
        pass

    def reset_peak(self):
        pass

    def memory(self):
        return {"kind": "synthetic_not_gpu_measurement"}

    def environment(self):
        return {"kind": "synthetic_not_real_cuda"}

    def release(self):
        self.calls.append("release")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EXPECTED_COUNTS", (4, 2, 2))
    monkeypatch.setattr(ev, "PROBE", ev.PROBE | {"normal_photos": 2, "warmup_tiles": 1})
    archive_path = tmp_path / "source.zip"
    normal_ids, normal_rows, normal_scores = ["0100", "0101"], [], []
    audited, old_photos, cases = [], [], []
    recovered = tmp_path / "recovered.png"
    with zipfile.ZipFile(archive_path, "w") as source:
        for index, photo_id in enumerate(normal_ids):
            rs = []
            for region in ev.fit.REGIONS:
                raw, row = png_row(photo_id, region, 10 + index)
                source.writestr(row["member"], raw)
                normal_rows.append(row)
                rs.append({"region": region, "member": row["member"], "score": float(10 + index)})
            normal_scores.append({"photo_id": photo_id, "score": float(10 + index), "regions": rs})
        for index, key in enumerate(("good/0000", "good/0001", "defect/0006", "defect/0054")):
            group, photo_id = key.split("/")
            label, rows, masks = int(group == "defect"), [], []
            value = (10, 70, 10, 70)[index]
            for region in ev.fit.REGIONS:
                member = f"extracted/{region}/test/{key}.png"
                image = Image.new("RGB", (256, 256), (value, 7, 9))
                raw = png(image)
                if member == ev.REPAIR_MEMBER:
                    recovered.write_bytes(raw)
                    monkeypatch.setattr(ev.maps, "REPAIRED_RGB_SHA256", hashlib.sha256(image.tobytes()).hexdigest())
                    monkeypatch.setattr(ev, "DAMAGED_SHA256", hashlib.sha256(b"broken image").hexdigest())
                    source.writestr(member, b"broken image")
                else:
                    source.writestr(member, raw)
                row = {"region": region, "member": member, "member_sha256": hashlib.sha256(raw).hexdigest(),
                       "decoded_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "mask_member": None}
                mask = np.zeros((256, 256), dtype=np.bool_)
                if label:
                    mask[10:20, 10:20] = True
                    mask_raw = png(Image.fromarray(mask.astype(np.uint8)))
                    row["mask_member"] = f"extracted/{region}/ground_truth/defect/{photo_id}.png"
                    if row["mask_member"] == ev.MASK_MEMBER:
                        mask_raw = b"broken known mask"
                        monkeypatch.setattr(ev, "MASK_SHA256", hashlib.sha256(mask_raw).hexdigest())
                        masks.append(None)
                    else:
                        row["mask"] = ev.mask_details(mask_raw)
                        masks.append(mask)
                    row["mask_sha256"] = hashlib.sha256(mask_raw).hexdigest()
                    source.writestr(row["mask_member"], mask_raw)
                else:
                    masks.append(mask)
                rows.append(row)
            audit_photo = {"key": key, "photo_id": photo_id, "label": label, "regions": rows}
            audited.append(audit_photo)
            old = {"key": key, "photo_id": photo_id, "label": label, "threshold": 60.0, "score": float(value),
                   "decision": "REVIEW_REQUIRED" if value >= 60 else "BELOW_THRESHOLD_NOT_AUTO_PASS",
                   "regions": [{"region": r["region"], "member": r["member"], "score": float(value)} for r in rows]}
            old_photos.append(old)
            location = ev.maps.describe_photo(old, [np.full((256, 256), value, dtype=np.float32)] * 12,
                                               masks, key != "defect/0006")
            cases.append({"key": key, "label": label, "group": "synthetic", "localization_eligible": key != "defect/0006",
                          "cards": [], "models": {k: copy.deepcopy(location) for k in ev.maps.MODEL_KEYS}})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    records = {"fit.json": {"checkpoint_context": {"synthetic": True}},
               "reload.json": {"photos": normal_scores, "receipt": {"runtime": {"kind": "synthetic_not_real_cuda"}}}}
    previous = {"test-audit.json": {"photos": audited}, "cohort-amendment.json": {"photo_keys": [p["key"] for p in audited]},
                "recovery.json": {"recovered_png_sha256": ev.fit.sha256_file(recovered)}, "evaluation.json": {"synthetic": True},
                **{f"{key}-scores.json": {"photos": copy.deepcopy(old_photos)} for key in ev.maps.MODEL_KEYS}}
    diagnosis = {"cases.json": cases, "analysis.json": {"summary": ev.maps.summarize(cases)}}
    return {"root": tmp_path / "run", "source_path": archive_path, "recovered_path": recovered,
            "checkpoint": checkpoint, "config": {}, "preparation": {"calibration_photo_ids": normal_ids, "calibration_rows": normal_rows},
            "records": records, "previous": previous, "diagnosis": diagnosis,
            "identity": {"synthetic": True}, "git": {"commit": "synthetic"}, "handoff": tmp_path / "result.zip"}


def execute(data, calls, **options):
    args = {k: v for k, v in data.items() if k != "source_path"}
    with zipfile.ZipFile(data["source_path"]) as source:
        return ev.execute(**args, source=source, runtime_factory=lambda cfg: Runtime(calls, **options))


def test_complete_real_orchestration_no_training_and_json_only(fixture):
    calls = []
    result = execute(fixture, calls)
    assert calls[0:2] == ["create", "load"] and calls[-1] == "release"
    assert calls.count("predict") == 1 + 24 + 48
    assert result["status"] == ev.COMPLETE
    assert result["models"][ev.MODEL]["fixed_threshold"]["counts"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert result["localization"]["defect_photo_denominator"] == 1
    assert result["localization"]["models"][ev.MODEL]["all_defects"]["gt_photo_denominator"] == 1
    assert result["sensitivity_without_reconstructed_photo"]["original_photos"] == 3
    assert result["sensitivity_without_reconstructed_photo"]["models"][ev.MODEL]["fixed_threshold"]["counts"]["fn"] == 1
    assert result["training_performed"] is result["automatic_pass_allowed"] is False
    assert result["thresholds_changed"] is result["confirmatory_claim_allowed"] is False
    assert result["latency_ms"]["tile_scoring"]["count"] == 48
    assert result["latency_ms"]["photo_scoring_ms"]["count"] == 4
    with zipfile.ZipFile(fixture["handoff"]) as archive:
        assert len(archive.namelist()) == 9 and all(n.endswith(".json") for n in archive.namelist())
        manifest = json.loads(archive.read("handoff-manifest.json"))
        assert all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in manifest.items())
        probe = json.loads(archive.read("normal-probe.json"))["contributing_attempts"][0]
        assert probe["passed"] and probe["threshold_recalculated"] is False
        assert len(probe["photos"]) == 2


def test_completed_rerun_uses_no_runtime_and_preserves_zip(fixture):
    first = execute(fixture, [])
    before = fixture["handoff"].read_bytes()
    calls = []
    assert execute(fixture, calls) == first
    assert calls == [] and fixture["handoff"].read_bytes() == before


def test_score_drift_stops_before_any_test_or_recovery_read(fixture, monkeypatch):
    monkeypatch.setattr(ev, "recovered_bytes", lambda *args: pytest.fail("Test tile read before probe passed"))
    with pytest.raises(ValueError, match="normal_score_drift"):
        execute(fixture, [], offset=1)
    probes = list(fixture["root"].glob("attempt-*/normal-probe.json"))
    probe = json.loads(probes[0].read_bytes())
    assert not probe["passed"] and len(probe["photos"]) == 1
    assert not list((fixture["root"] / "samples").glob("photo-*"))
    assert not fixture["handoff"].exists()


def test_slow_probe_stops_early_and_can_retry_same_input(fixture, monkeypatch):
    policy = ev.PROBE
    monkeypatch.setattr(ev, "PROBE", ev.PROBE | {"maximum_mean_tile_ms": 0.0})
    with pytest.raises(ValueError, match="severe_latency_regression"):
        execute(fixture, [])
    probe = json.loads(next(fixture["root"].glob("attempt-*/normal-probe.json")).read_bytes())
    assert len(probe["photos"]) == 1
    # Production policy is part of identity and cannot be changed between retries.
    # This test restores the injected artificial slowness bound only.
    monkeypatch.setattr(ev, "PROBE", policy)
    assert execute(fixture, [])["status"] == ev.COMPLETE


def test_interruption_reuses_only_complete_photo_and_repeats_normal_gate(fixture):
    with pytest.raises(RuntimeError, match="interruption"):
        execute(fixture, [], fail_after=1 + 24 + 12)
    sample = fixture["root"] / "samples/photo-001"
    before = (sample / "maps.npz").read_bytes()
    assert sample.exists() and not (fixture["root"] / "samples/photo-002").exists()
    assert len(list((fixture["root"] / "samples").glob(".photo-002-attempt-*"))) == 1
    calls = []
    result = execute(fixture, calls)
    assert calls.count("predict") == 1 + 24 + 36
    assert (sample / "maps.npz").read_bytes() == before
    assert result["latency_ms"]["mixed_attempts"] is True
    with zipfile.ZipFile(fixture["handoff"]) as archive:
        assert len(json.loads(archive.read("normal-probe.json"))["contributing_attempts"]) == 2


@pytest.mark.parametrize("name", ["samples/photo-001/maps.npz", "samples/photo-001/sample.json", "samples/photo-001/receipt.json", "result/evaluation.json"])
def test_corrupt_completed_output_rejected_before_runtime(fixture, name):
    execute(fixture, [])
    target = fixture["root"] / name
    target.write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(fixture, calls)
    assert calls == [] and target.read_bytes() == b"changed"


def test_changed_identity_is_not_silently_reused(fixture):
    execute(fixture, [])
    fixture["identity"] = {"synthetic": "different"}
    with pytest.raises(ValueError, match="identity differs"):
        execute(fixture, [])


def test_changed_runtime_is_rejected_before_predict(fixture):
    fixture["records"]["reload.json"]["receipt"]["runtime"] = {"kind": "different"}
    calls = []
    with pytest.raises(ValueError, match="environment differs"):
        execute(fixture, calls)
    assert "predict" not in calls and calls[-1] == "release"


def test_missing_or_changed_recovery_preserves_normal_probe_and_does_not_score_test(fixture):
    fixture["recovered_path"].write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="recovered tile changed"):
        execute(fixture, calls)
    assert calls.count("predict") == 25
    assert not list((fixture["root"] / "samples").glob("photo-*"))


def test_pairing_reports_gained_and_lost_defects_not_only_net_change():
    old = [{"key": f"defect/{i}", "label": 1, "score": s, "threshold": 5} for i, s in enumerate([9, 1, 9, 1])]
    new = [{**p, "score": s} for p, s in zip(old, [1, 9, 9, 1], strict=True)]
    result = ev.paired_changes(old, new)["defect"]
    assert result["counts"] == {"both_review": 1, "old_only_review": 1, "new_only_review": 1, "neither_review": 1}
    assert result["photo_keys"]["old_only_review"] == ["defect/0"]
    with pytest.raises(ValueError, match="cohort differs"):
        ev.paired_changes(old, new[::-1])


def test_bad_archive_hash_and_cli_before_gpu(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a ZIP")
    with pytest.raises(ValueError, match="checksum differs"):
        ev.checked_bundle(path, "0" * 64)
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/evaluate_mpi_pcb_normal_bank.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    assert ev.main(["--project", str(tmp_path), "--archive", str(path)]) == 1


def test_negative_or_nan_map_is_rejected(fixture, monkeypatch):
    original = Runtime.predict
    def negative(self, image):
        value, array = original(self, image)
        array[0, 0] = -1
        return value, array
    monkeypatch.setattr(Runtime, "predict", negative)
    with pytest.raises(ValueError, match="normal probe map"):
        execute(fixture, [])


def test_summary_rejects_missing_or_reordered_test_photos(fixture):
    execute(fixture, [])
    signature = ev.fit.digest(fixture["identity"])
    samples = [ev.read_sample(fixture["root"] / "samples", f"photo-{i:03d}", signature, p)
               for i, p in enumerate(fixture["previous"]["test-audit.json"]["photos"], 1)]
    for values in (samples[:-1], samples[::-1]):
        with pytest.raises(ValueError, match="test cohort"):
            ev.summarize(values, fixture["previous"], fixture["diagnosis"])


def test_handoff_corruption_is_not_overwritten(fixture):
    execute(fixture, [])
    fixture["handoff"].write_bytes(b"changed")
    with pytest.raises(zipfile.BadZipFile):
        execute(fixture, [])
    assert fixture["handoff"].read_bytes() == b"changed"


def test_public_run_wires_fixed_inputs_and_rejects_changed_checkpoint(fixture, monkeypatch, tmp_path):
    project = tmp_path / "project"
    checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(fixture["checkpoint"].read_bytes())
    monkeypatch.setattr(ev, "CHECKPOINT_SHA256", ev.fit.sha256_file(checkpoint))
    monkeypatch.setattr(ev.fit, "EXPECTED_ARCHIVE_SHA256", ev.fit.sha256_file(fixture["source_path"]))
    monkeypatch.setattr(ev.fit, "load_config", lambda path: fixture["config"])
    monkeypatch.setattr(ev.fit, "load_preparation", lambda path, config: fixture["preparation"])
    monkeypatch.setattr(ev, "load_fit", lambda *args: fixture["records"])
    monkeypatch.setattr(ev, "load_prior", lambda *args: (fixture["previous"], fixture["diagnosis"]))
    monkeypatch.setattr(ev.fit, "git_state", lambda path: fixture["git"])
    expected_recovery = project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png"
    seen = []
    def inspect(*args):
        assert args[0] == project / "artifacts/evaluations" / ev.NAME
        assert isinstance(args[1], zipfile.ZipFile)
        assert args[2] == expected_recovery and args[3] == checkpoint
        assert args[-1] == project / "artifacts/handoff" / f"{ev.NAME}.zip"
        seen.append("execute")
        return {"status": "synthetic_wiring_verified"}
    monkeypatch.setattr(ev, "execute", inspect)
    assert ev.run(project, fixture["source_path"])["status"] == "synthetic_wiring_verified"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint bytes differ"):
        ev.run(project, fixture["source_path"])
    assert seen == ["execute"]


def test_cli_argument_names_and_powershell_entrypoint(fixture, monkeypatch):
    calls = []
    def run(project, archive_path):
        calls.append((project, archive_path))
        return {"status": ev.COMPLETE, "models": {}, "localization": {}, "latency_ms": {}}
    monkeypatch.setattr(ev, "run", run)
    assert ev.main(["--project", str(PROJECT), "--archive", str(fixture["source_path"])]) == 0
    assert calls == [(PROJECT.resolve(), fixture["source_path"].resolve())]
    script = (PROJECT / "scripts/run_mpi_pcb_normal_bank_evaluation.ps1").read_text()
    assert '".venv\\Scripts\\python.exe"' in script
    assert '"scripts\\evaluate_mpi_pcb_normal_bank.py" --project $projectPath --archive $Archive' in script
    assert "fit_mpi_pcb_normal_bank.py" not in script


def test_missing_cached_photo_under_completed_report_cannot_claim_success(fixture):
    execute(fixture, [])
    sample = fixture["root"] / "samples/photo-001"
    sample.rename(fixture["root"] / "samples/preserved-photo-001")
    calls = []
    with pytest.raises(ValueError, match="missing cached photos"):
        execute(fixture, calls)
    assert calls == []


def test_lost_identity_link_recovers_without_overwriting_cached_results(fixture):
    execute(fixture, [])
    (fixture["root"] / "input-lock.json").unlink()
    calls = []
    assert execute(fixture, calls)["status"] == ev.COMPLETE
    assert calls == [] and (fixture["root"] / "input-lock.json").is_file()
