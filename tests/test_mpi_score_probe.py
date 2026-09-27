"""CPU-only protocol/cache/card integration. Not evidence of real GPU performance."""

import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import probe_mpi_scores as probe
from scripts import mpi_score_probe_protocol as p
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime
from tests.test_mpi_pcb_normal_bank_fit import png_row


def fingerprint(photo, region, rgb=None):
    return {"photo_id": photo, "region": region,
            "member": f"extracted/{region}/train/good/{photo}.png",
            "decoded_rgb_sha256": rgb or hashlib.sha256(f"{photo}/{region}".encode()).hexdigest()}


def test_selection_is_deterministic_disjoint_and_never_uses_scores_labels():
    rows = [fingerprint(f"{i:04d}", r) for i in range(20) for r in p.fit.REGIONS]
    preparation = {"training_photo_ids": ["0000"], "calibration_photo_ids": ["0001"],
                   "training_rows": rows[:12], "calibration_rows": rows[12:24]}
    test = [{"regions": [rows[24]]}]
    selected = p.select_development(rows, preparation, test, count=8)
    assert not {"0000", "0001", "0002"} & set(selected["photo_ids"])
    assert selected == p.select_development(list(reversed(rows)), preparation, test, count=8)
    assert len(selected["rows"]) == 96
    with pytest.raises(ValueError, match="Insufficient"):
        p.select_development(rows, preparation, test, count=100)
    with pytest.raises(ValueError, match="Incomplete"):
        p.select_development(rows[:-1], preparation, test, count=8)
    with pytest.raises(ValueError, match="Duplicate"):
        p.select_development(rows + rows[:1], preparation, test, count=8)


def test_duplicate_development_tiles_exclude_whole_later_photo():
    rows = [fingerprint(f"{i:04d}", r) for i in range(10) for r in p.fit.REGIONS]
    rows[12]["decoded_rgb_sha256"] = rows[0]["decoded_rgb_sha256"]
    empty = {"training_photo_ids": [], "calibration_photo_ids": [], "training_rows": [], "calibration_rows": []}
    result = p.select_development(rows, empty, [], count=9)
    assert not {"0000", "0001"}.issubset(result["photo_ids"])
    assert len(result["rows"]) == 108


def scored(prefix, photo_id, value, raw=None):
    return {"key": f"{prefix}/{photo_id}", "photo_id": photo_id, "label": 0,
            "regions": [{"region": r, "member": f"extracted/{r}/train/good/{photo_id}.png",
                         "top1": value + i, "native_raw": raw if raw is not None else value * 2}
                        for i, r in enumerate(p.fit.REGIONS)]}


def test_mad_fit_formula_all_regions_raw_not_map_max_and_negative_allowed():
    dev = [scored("development", str(i), value) for i, value in enumerate((10, 12, 14))]
    stats = p.fit_normalization(dev)
    assert stats[0]["median"] == 12 and stats[0]["mad"] == 2
    sample = scored("test", "9999", 10, raw=77)
    assert p.score(sample, p.SCORERS[0], stats) == 21
    assert p.score(sample, p.SCORERS[1], stats) == -1
    assert p.score(sample, p.SCORERS[2], stats) == 77
    with pytest.raises(ValueError, match="Degenerate"):
        p.fit_normalization([dev[0], dev[0]])
    with pytest.raises(ValueError, match="only separate"):
        p.fit_normalization([scored("calibration", "x", 10)])
    with pytest.raises(ValueError, match="Unknown scorer"):
        p.score(sample, "combined", stats)


def test_calibration_separation_and_threshold_is_photo_max(monkeypatch):
    dev = [scored("development", str(i), value) for i, value in enumerate((10, 12, 14))]
    cal = [scored("calibration", str(100+i), value, raw=value*3) for i, value in enumerate((13, 14, 15, 16))]
    monkeypatch.setattr(p.ev, "THRESHOLD", 27)
    result = p.calibration_contract(dev, cal)
    assert result["thresholds"] == {"baseline_top1": 27, "a_region_mad_top1": 2, "b_native_raw": 48}
    assert result["test_scores_used"] == 0
    overlap = copy.deepcopy(cal)
    overlap[0]["photo_id"] = "0"
    with pytest.raises(ValueError, match="overlap"):
        p.calibration_contract(dev, overlap)
    with pytest.raises(ValueError, match="only normal"):
        p.calibration_contract(dev, [{**cal[0], "label": 1}])


def test_invalid_raw_or_changed_replay_rejected():
    sample = scored("calibration", "0001", 10)
    photo = {**sample, "regions": [{"region": r["region"], "member": r["member"]} for r in sample["regions"]]}
    expected = {"score": 21, "regions": [{"region": r["region"], "member": r["member"], "score": r["top1"]} for r in sample["regions"]]}
    assert p.replay(sample, expected) == 0
    sample["regions"][0]["native_raw"] = float("nan")
    with pytest.raises(ValueError, match="Invalid score"):
        p.validate_sample(sample, photo)
    sample["regions"][0]["top1"] += .01
    with pytest.raises(ValueError, match="drift"):
        p.replay(sample, expected)


@pytest.fixture
def experiment(fixture, monkeypatch):
    data = fixture
    monkeypatch.setattr(probe.ev, "THRESHOLD", 11.0)
    ids, rows = ["0200", "0201", "0202"], []
    with zipfile.ZipFile(data["source_path"], "a") as source:
        for i, photo in enumerate(ids):
            for region in p.fit.REGIONS:
                raw, row = png_row(photo, region, 15+i)
                source.writestr(row["member"], raw)
                rows.append(row)
    baseline = copy.deepcopy(data["previous"][f"{probe.ev.maps.MODEL_KEYS[0]}-scores.json"]["photos"])
    for s in baseline:
        s["threshold"] = 11
        s["decision"] = "REVIEW_REQUIRED" if s["score"] >= 11 else "BELOW_THRESHOLD_NOT_AUTO_PASS"
    args = {k: data[k] for k in ("root", "checkpoint", "recovered_path", "config", "preparation", "records", "previous", "identity", "git", "handoff")}
    args.update(baseline=baseline, development={"photo_ids": ids, "rows": rows},
                selection=p.card_selection(data["records"]["reload.json"]["photos"], baseline))
    return data["source_path"], args


def execute(experiment, calls, **options):
    archive, args = experiment
    with zipfile.ZipFile(archive) as source:
        return probe.execute(source=source, **args, runtime_factory=lambda cfg: Runtime(calls, **options))


def test_full_orchestration_scores_cards_manifest_sensitivity_and_rerun(experiment):
    calls = []
    result = execute(experiment, calls)
    assert calls.count("predict") == 1 + 24 + (3+2+4)*12
    assert calls[0:2] == ["create", "load"] and calls[-1] == "release"
    assert result["status"] == "score_probe_completed_exploratory"
    assert result["baseline_maximum_absolute_replay_difference"] == 0
    assert result["automatic_pass_allowed"] is result["training_performed"] is result["service_model_changed"] is False
    assert result["original_photos"] == 4 and result["defect_photos"] == 2
    assert result["sensitivity_without_reconstructed_photo"]["baseline_top1"]["fixed_threshold"]["counts"] == {"tp": 0, "fp": 1, "tn": 1, "fn": 1}
    _, args = experiment
    with zipfile.ZipFile(args["handoff"]) as archive:
        manifest = json.loads(archive.read("handoff-manifest.json"))
        assert set(archive.namelist()) == set(manifest) | {"handoff-manifest.json"}
        assert all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in manifest.items())
        assert not any(n.endswith((".pt", ".npz")) for n in archive.namelist())
        assert len([n for n in archive.namelist() if n.endswith(".jpg")]) == len(args["selection"])*2
        assert 'href="cards/defect-0006-1.jpg"' in archive.read("index.html").decode()
        assert "동일 시험의 점수 비교" in archive.read("index.html").decode()
        locations = json.loads(archive.read("card-locations.json"))
        assert all(r["gt"] is None for r in locations["defect/0006"]["regions"])
    before, calls = args["handoff"].read_bytes(), []
    assert execute(experiment, calls) == result
    assert calls == [] and args["handoff"].read_bytes() == before


def test_interrupted_samples_reuse_and_probe_repeated(experiment):
    with pytest.raises(RuntimeError, match="interruption"):
        execute(experiment, [], fail_after=1+24+12)
    _, args = experiment
    prior = (args["root"] / "development/photo-001/sample.json").read_bytes()
    calls = []
    result = execute(experiment, calls)
    assert result["mixed_attempts"] is True
    assert calls.count("predict") == 1+24+(2+2+4)*12
    assert (args["root"] / "development/photo-001/sample.json").read_bytes() == prior


def test_probe_blocks_before_development_and_test(experiment, monkeypatch):
    monkeypatch.setattr(probe.ev, "recovered_bytes", lambda *args: pytest.fail("test read"))
    with pytest.raises(ValueError, match="Normal probe blocked"):
        execute(experiment, [], offset=1)
    assert not list((experiment[1]["root"] / "development").glob("photo-*"))


def test_unusable_mad_stops_before_test_payloads(experiment, monkeypatch):
    class FlatDevelopment(Runtime):
        def predict(self, image):
            raw, array = super().predict(image)
            if 15 <= raw <= 17:
                array.fill(15)
            return raw, array
    monkeypatch.setattr(probe.ev, "recovered_bytes", lambda *args: pytest.fail("test opened before usable normalization"))
    archive, args = experiment
    with zipfile.ZipFile(archive) as source, pytest.raises(ValueError, match="Degenerate"):
        probe.execute(source=source, **args, runtime_factory=lambda cfg: FlatDevelopment([]))
    assert not list((args["root"] / "test").glob("photo-*"))


def test_test_scoring_starts_only_after_frozen_normal_contract(experiment, monkeypatch):
    archive, args = experiment
    original = probe.ev.read_tile
    def guarded_read(*params):
        contract = json.loads((args["root"] / "contract/calibration.json").read_bytes())
        assert contract["test_scores_used"] == 0 and set(contract["thresholds"]) == set(p.SCORERS)
        return original(*params)
    monkeypatch.setattr(probe.ev, "read_tile", guarded_read)
    assert execute(experiment, [])["status"] == "score_probe_completed_exploratory"


@pytest.mark.parametrize("name", ["development/photo-001/sample.json", "calibration/photo-001/card-1.jpg", "test/photo-001/receipt.json", "contract/calibration.json", "result/comparison.json"])
def test_corrupt_cache_preserved_and_rejected_before_gpu(experiment, name):
    execute(experiment, [])
    path = experiment[1]["root"] / name
    path.write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(experiment, calls)
    assert calls == [] and path.read_bytes() == b"changed"


def test_changed_identity_rejected(experiment):
    execute(experiment, [])
    experiment[1]["identity"] = {"changed": True}
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(experiment, [])


def test_cli_help_and_windows_entrypoint():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(project / "scripts/probe_mpi_scores.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    wrapper = (project / "scripts/run_mpi_score_probe.ps1").read_text()
    assert "scripts\\probe_mpi_scores.py" in wrapper and "$LASTEXITCODE" in wrapper
    assert "Remove-Item" not in wrapper
