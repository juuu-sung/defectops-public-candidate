import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from scripts import export_mpi_normal_visual_review as m


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    real = Path(__file__).resolve().parents[1] / "configs" / f"{m.NAME}.json"
    config = json.loads(real.read_text(encoding="utf-8"))
    archive = tmp_path / "extracted.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, photo in enumerate(config["photos"]):
            for j, tile in enumerate(photo["tiles"]):
                image = Image.new("RGB", (256, 256), (i * 20, j * 20, 100))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                raw = buffer.getvalue()
                tile["member_sha256"] = m.sha(raw)
                tile["decoded_rgb_sha256"] = m.sha(image.tobytes())
                z.writestr(tile["member"], raw)
        z.writestr("extracted/grid1-1_s1024/test/defect/0000.png", b"MUST NOT READ")
        z.writestr("extracted/grid1-1_s1024/ground_truth/defect/0000.png", b"MUST NOT READ")
    config["archive_sha256"] = m.file_sha(archive)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / f"{m.NAME}.json"
    data = m.json_bytes(config)
    path.write_bytes(data)
    monkeypatch.setattr(m, "CONFIG_SHA", m.sha(data))
    return tmp_path, archive, config


def test_pinned_real_inventory_and_crlf(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / f"{m.NAME}.json"
    config, digest = m.load_config(path)
    assert digest == m.CONFIG_SHA
    assert len(config["photos"]) == 10
    assert sum(len(p["tiles"]) for p in config["photos"]) == 120
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert m.load_config(crlf) == (config, digest)


def test_export_no_test_payload_exact_pixels_and_idempotence(synthetic, monkeypatch):
    project, archive, config = synthetic
    before = m.file_sha(archive)
    original_read = zipfile.ZipFile.read
    seen = []
    def guarded(z, name, *args, **kwargs):
        if Path(z.filename) == archive:
            member = name.filename if isinstance(name, zipfile.ZipInfo) else name
            assert "/train/good/" in member
            seen.append(member)
        return original_read(z, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "read", guarded)
    report = m.run(project, archive)
    assert len(seen) == len(set(seen)) == 120
    assert report["source_tiles_decoded_this_invocation"] == 120
    assert report["test_images_decoded"] == report["model_scores_computed"] == 0
    result = Path(report["viewer"]).parent
    assert "아님" in (result / "index.html").read_text(encoding="utf-8")
    for i, photo in enumerate(config["photos"]):
        with Image.open(result / "photos" / photo["photo_id"] / "contact.png") as card:
            assert card.size == (824, 1224)
            for j, tile in enumerate(photo["tiles"]):
                x, y = 12 + j % 3 * 272, 92 + j // 3 * 288
                assert m.sha(card.crop((x, y, x+256, y+256)).tobytes()) == tile["decoded_rgb_sha256"]
                assert m.file_sha(result / "photos" / photo["photo_id"] / (tile["region"] + ".png")) == tile["member_sha256"]
    second = m.run(project, archive)
    assert second["source_tiles_decoded_this_invocation"] == 0
    assert second["handoff_sha256"] == report["handoff_sha256"]
    assert len(seen) == 120
    assert before == m.file_sha(archive)


def test_interrupted_photo_resumes_completed_evidence(synthetic, monkeypatch):
    project, archive, _ = synthetic
    original = m.decode
    calls = 0
    def interrupt(raw, tile):
        nonlocal calls
        calls += 1
        if calls == 14:
            raise RuntimeError("synthetic interruption")
        return original(raw, tile)
    monkeypatch.setattr(m, "decode", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        m.run(project, archive)
    root = project / "artifacts" / "analyses" / m.NAME / "photos"
    receipt = root / "1554" / "receipt.json"
    prior = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    assert list(root.glob(".1681-attempt-*"))
    monkeypatch.setattr(m, "decode", original)
    report = m.run(project, archive)
    assert report["source_tiles_decoded_this_invocation"] == 108
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == prior


@pytest.mark.parametrize("where", ["photos/1554/contact.png", "result/index.html"])
def test_changed_evidence_is_preserved_and_rejected(synthetic, where):
    project, archive, _ = synthetic
    m.run(project, archive)
    path = project / "artifacts" / "analyses" / m.NAME / where
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        m.run(project, archive)
    assert path.read_bytes() == b"changed"


def test_changed_bundle_rejected(synthetic):
    project, archive, _ = synthetic
    report = m.run(project, archive)
    path = Path(report["handoff"])
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("wrong.txt", "changed")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="inventory differs"):
        m.run(project, archive)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["test", "size", "cohort", "duplicate"])
def test_selection_constraints_even_with_new_digest(synthetic, monkeypatch, mutation):
    project, _, config = synthetic
    if mutation == "test":
        config["photos"][0]["tiles"][0]["member"] = "../../test.png"
    elif mutation == "size":
        config["photos"][0]["tiles"][0]["width"] = 512
    elif mutation == "cohort":
        config["photos"][0]["cohort"] = "test"
    else:
        config["photos"][0]["tiles"][1] = config["photos"][0]["tiles"][0]
    path = project / "configs" / f"{m.NAME}.json"
    data = m.json_bytes(config)
    path.write_bytes(data)
    monkeypatch.setattr(m, "CONFIG_SHA", m.sha(data))
    with pytest.raises(ValueError):
        m.load_config(path)


def test_archive_and_config_mismatch_before_output(synthetic):
    project, archive, _ = synthetic
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive SHA256"):
        m.run(project, archive)
    assert not (project / "artifacts").exists()
    path = project / "configs" / f"{m.NAME}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="configuration"):
        m.load_config(path)


@pytest.mark.parametrize("kind", ["raw", "pixels", "size", "png_crc"])
def test_strict_image_verification(kind):
    image = Image.new("RGB", (128, 256) if kind == "size" else (256, 256), "blue")
    b = io.BytesIO()
    image.save(b, format="PNG")
    raw = b.getvalue()
    if kind == "png_crc":
        raw = bytearray(raw)
        offset = raw.index(b"IDAT") + 5
        raw[offset] ^= 1
        raw = bytes(raw)
    tile = {"member": "normal.png", "member_sha256": m.sha(raw),
            "decoded_rgb_sha256": m.sha(image.tobytes())}
    if kind == "raw":
        tile["member_sha256"] = "0" * 64
    if kind == "pixels":
        tile["decoded_rgb_sha256"] = "0" * 64
    with pytest.raises((ValueError, SyntaxError, OSError)):
        m.decode(raw, tile)


@pytest.mark.parametrize("duplicate", [True, False])
def test_zip_member_missing_or_duplicate(synthetic, monkeypatch, duplicate):
    project, archive, config = synthetic
    if duplicate:
        with zipfile.ZipFile(archive, "a") as z:
            with pytest.warns(UserWarning):
                z.writestr(config["photos"][0]["tiles"][0]["member"], b"duplicate")
    else:
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("unrelated.txt", b"missing")
    config["archive_sha256"] = m.file_sha(archive)
    data = m.json_bytes(config)
    (project / "configs" / f"{m.NAME}.json").write_bytes(data)
    monkeypatch.setattr(m, "CONFIG_SHA", m.sha(data))
    with pytest.raises(ValueError, match="missing or duplicated"):
        m.run(project, archive)


def test_cli_arguments_and_error_exit(monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setattr(m, "run", lambda project, archive: seen.append((project, archive)) or {"ok": True})
    assert m.main(["--project", str(tmp_path), "--archive", "normal.zip"]) == 0
    assert seen == [(tmp_path, Path("normal.zip"))]
    def fail(*args):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(m, "run", fail)
    assert m.main(["--archive", "normal.zip"]) == 1
    assert "preserved" in capsys.readouterr().out


def test_single_writer_lock_and_release(tmp_path):
    with m.locked(tmp_path):
        with pytest.raises(RuntimeError, match="Another export"):
            with m.locked(tmp_path):
                pytest.fail("Second writer must not enter")
    with m.locked(tmp_path):
        pass
