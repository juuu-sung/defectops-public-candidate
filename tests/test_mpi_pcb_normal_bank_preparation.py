import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from scripts import prepare_mpi_pcb_normal_bank as pilot
from tests.test_mpi_pcb_selected_normal_decode_audit import corrupt_idat_crc

PROJECT = Path(__file__).resolve().parents[1]


def small_config():
    return {**pilot.EXPECTED_CONFIG, "training_original_photos": 2, "calibration_original_photos": 1}


def fake_rows(ids=("cal", "a", "b", "c", "d")):
    return [{"photo_id": p, "region": r, "member": f"extracted/{r}/train/good/{p}.png",
             "member_sha256": hashlib.sha256(f"bytes/{p}/{r}".encode()).hexdigest(),
             "decoded_rgb_sha256": hashlib.sha256(f"pixels/{p}/{r}".encode()).hexdigest(),
             "width": 256, "height": 256} for p in ids for r in pilot.REGIONS]


def calibration(rows):
    return {"calibration_photo_ids": ["cal"], "calibration_rows": [dict(r) for r in rows if r["photo_id"] == "cal"]}


def test_frozen_config_and_budget():
    config = pilot.load_config(PROJECT / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    budget = pilot.patch_budget(config)
    assert budget["training_tiles"] == 768
    assert budget["candidate_embeddings"] == 786432
    assert budget["coreset_embeddings"] == 78643
    assert budget["candidate_float32_bytes"] / 1024**3 == 2.25
    assert budget["peak_memory_measured"] is False
    with pytest.raises(ValueError, match="budget exceeded"):
        pilot.patch_budget({**config, "training_original_photos": 100})


def test_config_changes_are_not_silently_accepted(tmp_path):
    source = (PROJECT / "configs/mpi-pcb-normal-bank-pilot-v1.toml").read_text()
    path = tmp_path / "changed.toml"
    path.write_text(source.replace("training_original_photos = 64", "training_original_photos = 65"))
    with pytest.raises(ValueError, match="frozen"):
        pilot.load_config(path)


def test_selection_is_deterministic_and_keeps_whole_photos():
    rows = fake_rows()
    first = pilot.build_split(rows, calibration(rows), {}, small_config())
    second = pilot.build_split(list(reversed(rows)), calibration(rows), {}, small_config())
    assert first == second
    expected = sorted(["a", "b", "c", "d"], key=lambda p: hashlib.sha256(
        f'{small_config()["selection_salt"]}\0{p}'.encode()).hexdigest())[:2]
    assert first["training_photo_ids"] == expected
    assert len(first["training_rows"]) == 24
    assert all(r["photo_id"] != "cal" for r in first["training_rows"])


def test_any_calibration_or_test_tile_excludes_entire_fit_photo():
    rows = fake_rows()
    rows[12]["decoded_rgb_sha256"] = rows[0]["decoded_rgb_sha256"]  # a -> calibration.
    test_rgb = {rows[24]["decoded_rgb_sha256"]: ["test/id/region"]}  # b -> test.
    result = pilot.build_split(rows, calibration(rows), test_rgb, small_config())
    assert set(result["training_photo_ids"]) == {"c", "d"}
    assert {r["photo_id"] for r in result["excluded_fit_candidates"]} == {"a", "b"}
    # Exclusion is solely a fingerprint lookup: there are no labels or scores to consult.
    assert not set(test_rgb) & {r["decoded_rgb_sha256"] for r in result["training_rows"]}


def test_full_photo_duplicates_deduplicated_partial_tiles_only_reported():
    rows = fake_rows(("cal", "a", "b", "c"))
    for a, b in zip(rows[12:24], rows[24:36], strict=True):
        b["decoded_rgb_sha256"] = a["decoded_rgb_sha256"]
    rows[36]["decoded_rgb_sha256"] = rows[12]["decoded_rgb_sha256"]
    result = pilot.build_split(rows, calibration(rows), {}, small_config())
    assert len(result["training_photo_ids"]) == 2
    assert "c" in result["training_photo_ids"]
    assert len(result["excluded_fit_candidates"]) == 1
    assert result["excluded_fit_candidates"][0]["reason"] == "identical_full_photo"
    assert len(result["partial_tile_duplicate_groups_in_fit"]) == 1


def test_insufficient_cohort_does_not_silently_shrink():
    rows = fake_rows(("cal", "a"))
    result = pilot.build_split(rows, calibration(rows), {}, small_config())
    assert result["status"].endswith("insufficient_unique_photos")
    assert result["eligible_original_photos"] == 1
    assert result["training_rows"] == result["training_photo_ids"] == []


def test_calibration_overlap_is_reported_without_changing_calibration():
    rows = fake_rows()
    cal = calibration(rows)
    result = pilot.build_split(rows, cal, {rows[0]["decoded_rgb_sha256"]: ["good/0013/region"]}, small_config())
    assert result["calibration_rows"] == cal["calibration_rows"]
    assert result["calibration_test_exact_tile_overlap"]


def test_changed_calibration_or_incomplete_group_fails():
    rows = fake_rows()
    frozen = calibration(rows)
    rows[0]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Calibration bytes/pixels"):
        pilot.build_split(rows, frozen, {}, small_config())
    with pytest.raises(ValueError, match="Incomplete"):
        pilot.build_split(rows[:-1], frozen, {}, small_config())
    with pytest.raises(ValueError, match="duplicate"):
        pilot.build_split(rows + rows[:1], frozen, {}, small_config())


def png(value=1, mode="RGB", size=(256, 256)):
    stream = io.BytesIO()
    color = (value, value // 2, 7) if mode == "RGB" else (value, 0, 0, 255)
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("payload", [png(mode="RGBA"), png(size=(128, 128)), corrupt_idat_crc(png()), b"broken"])
def test_bad_normal_rejected_without_repair(tmp_path, payload):
    archive = tmp_path / "bad.zip"
    region = pilot.REGIONS[0]
    name = f"extracted/{region}/train/good/a.png"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(name, payload)
    with zipfile.ZipFile(archive) as source:
        with pytest.raises(ValueError, match="integrity failure"):
            pilot.scan_normals(source, {region: {"a": source.getinfo(name)}}, ["a"])
    with zipfile.ZipFile(archive) as source:
        assert source.read(name) == payload


def test_scan_rejects_test_member_before_read(tmp_path):
    archive = tmp_path / "guard.zip"
    region = pilot.REGIONS[0]
    name = f"extracted/{region}/test/good/a.png"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(name, b"must not decode")
    with zipfile.ZipFile(archive) as source, patch.object(source, "read", side_effect=AssertionError("payload accessed")):
        with pytest.raises(ValueError, match="Unexpected normal"):
            pilot.scan_normals(source, {region: {"a": source.getinfo(name)}}, ["a"])


def test_handoff_atomic_no_overwrite_resume_and_tamper(tmp_path):
    output = tmp_path / "handoff.zip"
    identity = {"code": "same"}
    report = {"input_identity": identity}
    pilot.write_handoff(output, report, [], {})
    before = output.read_bytes()
    assert pilot.read_completed(output, identity) == report
    with pytest.raises(FileExistsError):
        pilot.write_handoff(output, report, [], {})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".mpi-normal-preparation-*"))
    with pytest.raises(ValueError, match="different inputs"):
        pilot.read_completed(output, {"code": "changed"})
    other = tmp_path / "tampered.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(other, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"tampered" if name == "config.json" else source.read(name))
    with pytest.raises(ValueError, match="checksum"):
        pilot.read_completed(other, identity)


def test_run_and_cli_real_normal_scan_and_packaging(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    ids = ["cal", "a", "b", "broken"]
    with zipfile.ZipFile(archive, "w") as target:
        for number, photo in enumerate(ids):
            for idx, region in enumerate(pilot.REGIONS):
                raw = b"invalid whole photo must be skipped" if photo == "broken" else png(1 + number * 12 + idx)
                target.writestr(f"extracted/{region}/train/good/{photo}.png", raw)
        target.writestr("extracted/grid1-1_s1024/test/defect/a.png", b"never open")
    with zipfile.ZipFile(archive) as source:
        normal = {region: {p: source.getinfo(f"extracted/{region}/train/good/{p}.png") for p in ids} for region in pilot.REGIONS}
        rows = pilot.scan_normals(source, normal, ["cal"])
    cal = calibration(rows)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text('{"registration_json_sha256":"fixture"}')
    monkeypatch.setattr(pilot, "SOURCE_MD5", pilot.archive_hashes(archive)[0])
    monkeypatch.setattr(pilot, "EXPECTED_ARCHIVE_SHA256", pilot.sha256_file(archive))
    monkeypatch.setattr(pilot, "EXPECTED_POOL_REPORT_SHA256", pilot.sha256_file(pool_path))
    monkeypatch.setattr(pilot, "load_config", lambda path: small_config())
    monkeypatch.setattr(pilot, "validate_v4_report", lambda path: cal)
    monkeypatch.setattr(pilot, "read_test_fingerprints", lambda path: {})
    monkeypatch.setattr(pilot, "audit_members", lambda infos: {"errors": []})
    monkeypatch.setattr(pilot, "normal_members", lambda infos: normal)
    monkeypatch.setattr(pilot, "validate_pool_report", lambda *args: ([], {"broken"}))
    # Do not stub the runtime-facing scan, split selection, CLI or archive writer.
    real_read = zipfile.ZipFile.read
    reads = []

    def guarded_read(source, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        reads.append(name)
        assert "/test/" not in name and "/ground_truth/" not in name
        assert "/broken.png" not in name
        return real_read(source, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    output = tmp_path / "result.zip"
    args = ["--project", str(PROJECT), "--archive", str(archive), "--pool-audit", str(pool_path),
            "--preflight", "fixture", "--evaluation", "fixture", "--output", str(output)]
    assert pilot.main(args) == 0
    with zipfile.ZipFile(output) as source:
        report = json.loads(source.read("preparation.json"))
        assert all(name.endswith(".json") for name in source.namelist())
    assert report["training_performed"] is False
    assert report["model_scores_computed"] == report["test_images_decoded_this_run"] == 0
    assert report["normal_tiles_decoded_this_run"] == 36
    assert set(report["training_photo_ids"]) == {"a", "b"}
    assert report["invalid_original_photo_ids_excluded"] == ["broken"]
    before = output.read_bytes()
    reads.clear()
    assert pilot.main(args) == 0
    assert not any("/train/good/" in name for name in reads)  # Verified reuse; no re-scan.
    assert output.read_bytes() == before


def test_entry_point_can_run_help_without_torch():
    result = subprocess.run([sys.executable, "scripts/prepare_mpi_pcb_normal_bank.py", "--help"],
                            cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--archive" in result.stdout and "--evaluation" in result.stdout


def test_unreviewed_source_or_evaluation_refused(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not the official data")
    with pytest.raises(ValueError, match="checksum differs"):
        pilot.read_test_fingerprints(bad)
    with pytest.raises(ValueError, match="source archive checksum"):
        pilot.run(PROJECT, bad, bad, bad, bad, PROJECT / "configs/mpi-pcb-normal-bank-pilot-v1.toml", tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()
