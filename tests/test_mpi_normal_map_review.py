import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import review_mpi_normal_maps as m
from scripts import mpi_normal_map_cards as cards


def fake_values(case_index, region_index):
    return (np.arange(256*256, dtype=np.float32).reshape(256, 256)/65536 + 25 + case_index + region_index/10).astype(np.float32)


@pytest.fixture
def inputs(tmp_path):
    config, _ = m.visual.load_config(Path(__file__).resolve().parents[1]/"configs/mpi-normal-visual-review-v1.json")
    source_path = tmp_path/"normal.zip"
    entries = []
    with zipfile.ZipFile(source_path, "w") as z:
        for j, pid in enumerate(m.IDS):
            photo = copy.deepcopy(next(p for p in config["photos"] if p["photo_id"] == pid))
            rows = []
            for i, tile in enumerate(photo["tiles"]):
                a = np.full((256, 256, 3), 150, dtype=np.uint8)
                a[:, :10] = 0
                a[64, 64] = [j, i, 200]
                image = Image.fromarray(a)
                b = io.BytesIO()
                image.save(b, format="PNG")
                tile["member_sha256"] = m.visual.sha(b.getvalue())
                tile["decoded_rgb_sha256"] = m.visual.sha(a.tobytes())
                z.writestr(f"photos/{pid}/{tile['region']}.png", b.getvalue())
                values = fake_values(j, i)
                rows.append({"region": tile["region"], "member": tile["member"], "native_raw": float(values.max()),
                             "top1": m.fit.candidate_image_score(0, values, m.fit.CANDIDATE)})
            entries.append({"photo": photo, "expected": {"photo_id": pid, "key": f"calibration/{pid}", "label": 0, "regions": rows}})
        z.writestr("test/DO_NOT_READ.png", b"not normal")
        z.writestr("photos/1554/DO_NOT_READ.png", b"not selected")
    method = {"field": "top1", "threshold": 2.3,
              "regions": [{"region": r, "center": 20., "scale": 5.} for r in m.fit.REGIONS]}
    environment = {"kind": "synthetic_cpu_test_not_model_evidence"}
    records = {"fit.json": {"checkpoint_context": {}}, "reload.json": {"receipt": {"runtime": environment}}}
    state = {"calls": 0, "loads": 0, "releases": 0, "fail_at": None, "delta": 0., "environment": environment}
    class FakeRuntime:
        def __init__(self, config):
            pass
        def load(self, *args):
            state["loads"] += 1
        def environment(self):
            return state["environment"]
        def synchronize(self):
            pass
        def predict(self, image):
            state["calls"] += 1
            if state["calls"] == state["fail_at"]:
                raise RuntimeError("synthetic interruption")
            j, i, _ = image.getpixel((64, 64))
            values = fake_values(j, i) + state["delta"]
            return float(values.max()), values
        def release(self):
            state["releases"] += 1
    def execute(identity=None):
        with zipfile.ZipFile(source_path) as z:
            return m.execute(tmp_path/"run", z, tmp_path/"model.pt", {}, records, entries, method,
                             identity or {"test": True}, tmp_path/"handoff.zip", FakeRuntime)
    return tmp_path, source_path, entries, method, records, state, execute


def test_full_run_only_selected_normals_and_cache_without_runtime(inputs, monkeypatch):
    root, source, _, _, _, state, execute = inputs
    original = zipfile.ZipFile.read
    selected = set()
    def guarded(z, name, *args, **kwargs):
        if Path(z.filename) == source:
            n = name.filename if isinstance(name, zipfile.ZipInfo) else name
            assert n.startswith(tuple(f"photos/{pid}/" for pid in m.IDS))
            selected.add(n)
        return original(z, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "read", guarded)
    report = execute()
    assert report["normal_tiles"] == len(selected) == state["calls"] == 60
    assert report["maximum_absolute_replay_difference"] == 0
    assert report["test_images_decoded"] == 0
    assert state["loads"] == state["releases"] == 1
    before = (root/"handoff.zip").read_bytes()
    assert execute() == report
    assert state["calls"] == 60 and state["loads"] == 1
    assert (root/"handoff.zip").read_bytes() == before
    with zipfile.ZipFile(root/"handoff.zip") as z:
        assert len([n for n in z.namelist() if n.endswith("maps.npz")]) == 5
        assert all("test/" not in n for n in z.namelist())
    case = json.loads((root/"run/cases/0290/case.json").read_bytes())
    assert all(sum(p["top_pixels"] for p in r["map"]["partitions"].values()) == 656 for r in case["regions"])


def test_interrupted_resume_preserves_first_photo(inputs):
    root, _, _, _, _, state, execute = inputs
    state["fail_at"] = 14
    with pytest.raises(RuntimeError, match="interruption"):
        execute()
    receipt = root/"run/cases/1671/receipt.json"
    before = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    assert list((root/"run/cases").glob(".0616-attempt-*"))
    assert not (root/"handoff.zip").exists()
    state["fail_at"] = None
    execute()
    assert state["calls"] == 14+48
    assert state["loads"] == state["releases"] == 2
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before


@pytest.mark.parametrize("problem", ["drift", "environment"])
def test_replay_or_environment_failure_no_publication(inputs, problem):
    root, _, _, _, _, state, execute = inputs
    if problem == "drift":
        state["delta"] = .1
    else:
        state["environment"] = {"kind": "wrong"}
    with pytest.raises(ValueError):
        execute()
    assert not (root/"run/cases/1671").exists()
    assert not (root/"handoff.zip").exists()
    assert state["releases"] == 1
    assert list((root/"run").glob("attempt-*/failure.json"))


def test_altered_cache_rejected_without_gpu(inputs):
    root, _, _, _, _, state, execute = inputs
    execute()
    target = root/"run/cases/1671/maps.npz"
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute()
    assert state["loads"] == 1 and target.read_bytes() == b"tampered"


def test_semantically_changed_cache_rejected_even_when_resealed(inputs):
    root, _, entries, _, records, state, execute = inputs
    execute()
    path = root/"run/cases/1671"
    case = json.loads((path/"case.json").read_bytes())
    case["regions"][0]["map"]["top_count"] = 655
    (path/"case.json").write_bytes(m.fit.canonical(case))
    receipt = json.loads((path/"receipt.json").read_bytes())
    receipt["files"]["case.json"] = m.fit.sha256_file(path/"case.json")
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = m.fit.digest(receipt)
    (path/"receipt.json").write_bytes(m.fit.canonical(receipt))
    with pytest.raises(ValueError, match="attribution"):
        execute()
    assert state["loads"] == 1


def test_identity_change_preserved(inputs):
    root, _, _, _, _, state, execute = inputs
    execute()
    before = (root/"handoff.zip").read_bytes()
    with pytest.raises(ValueError):
        execute({"test": "changed"})
    assert (root/"handoff.zip").read_bytes() == before and state["loads"] == 1


def test_selection_and_original_integrity(inputs):
    _, source, entries, _, _, _, execute = inputs
    entries[0]["photo"]["cohort"] = "test"
    with pytest.raises(ValueError, match="calibration"):
        execute()
    entries[0]["photo"]["cohort"] = "calibration"
    entries[0]["photo"]["tiles"][0]["member_sha256"] = "0"*64
    with zipfile.ZipFile(source) as z:
        with pytest.raises(ValueError, match="checksum"):
            m.images_for(z, entries[0])


@pytest.mark.parametrize("problem", ["shape", "nan", "negative", "native", "decision"])
def test_describe_rejects_invalid_maps_or_boundary_flip(inputs, problem):
    _, source, entries, method, _, _, _ = inputs
    with zipfile.ZipFile(source) as z:
        images = m.images_for(z, entries[0])
    arrays = np.stack([fake_values(0, i) for i in range(12)])
    native = [float(v.max()) for v in arrays]
    if problem == "shape":
        arrays = arrays[:1]
    elif problem == "nan":
        arrays[0, 0, 0] = np.nan
    elif problem == "negative":
        arrays[0, 0, 0] = -1
    elif problem == "native":
        native[0] = float("inf")
    else:
        method["threshold"] = m.scores.score(entries[0]["expected"], method)["score"]
        arrays -= np.float32(5e-5)
    try:
        with pytest.raises(ValueError):
            m.describe(entries[0], arrays, native, images, method)
    finally:
        for image in images:
            image.close()


def test_black_partition_connectivity_band_and_top_ties():
    a = np.full((256, 256, 3), 100, dtype=np.uint8)
    a[:, :10] = 0
    a[100:103, 100:103] = 0  # Interior black component is not border-connected.
    image = Image.fromarray(a)
    values = np.zeros((256, 256), dtype=np.float32)
    values[:, :3] = 30
    top, masks = cards.spatial_masks(image, values)
    assert masks["edge_connected_black"].sum() == 2560
    assert masks["adjacent_nonblack_2px"].sum() == 512
    assert masks["other_pixels"][101, 101]
    assert np.all(sum(x.astype(int) for x in masks.values()) == 1)
    result = cards.statistics(image, values)
    assert result["top_count"] == 656 and result["top_mean"] == 30
    assert result["pixels_equal_to_cutoff"] == 768
    assert result["partitions"]["edge_connected_black"]["top_fraction"] == 1
    assert top[0, 0] and not top[-1, 0]
    a[100, 11] = 0  # Nearby but disconnected black is not a nonblack band pixel.
    _, masks = cards.spatial_masks(Image.fromarray(a), values)
    assert masks["adjacent_nonblack_2px"].sum() == 511
    assert masks["other_pixels"][100, 11]


def test_fixed_color_scale_zero_map_and_no_black():
    image = Image.new("RGB", (256, 256), (100, 100, 100))
    a = np.full((256, 256), 20, dtype=np.float32)
    b = a.copy()
    b[-1, -1] = 200
    assert cards.panels(image, a)[1].getpixel((0, 0)) == cards.panels(image, b)[1].getpixel((0, 0))
    assert cards.statistics(image, b)["display_clipped_high_pixels"] == 1
    zero = cards.statistics(image, np.zeros_like(a))
    assert zero["top_count"] == 656
    assert zero["partitions"]["other_pixels"]["top_value_sum_fraction"] is None
    assert zero["partitions"]["edge_connected_black"]["area_pixels"] == 0


def test_cli_and_wrapper_contract(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(m, "run", lambda project, visual_path: called.append((project, visual_path)) or {})
    assert m.main(["--project", str(tmp_path), "--visual-bundle", "normal.zip"]) == 0
    assert called == [(tmp_path, Path("normal.zip"))]
    wrapper = (Path(__file__).resolve().parents[1]/"scripts/run_mpi_normal_map_review.ps1").read_text()
    assert "review_mpi_normal_maps.py' --project $projectPath" in wrapper
    def fail(*args):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(m, "run", fail)
    assert m.main([]) == 1
