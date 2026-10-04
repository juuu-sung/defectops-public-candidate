"""CPU-only saved-map mathematics, provenance, decoding, export and CLI regression tests."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_efficientad_map_diagnostic as q
from scripts import mpi_efficientad_map_cards as cards
from scripts import review_mpi_efficientad_maps as runner


def contract():
    return {"map_quantiles": [{"q90": .2, "q995": 1.2}]*2,
            "centers": [0.]*12, "scales": [1.]*12, "threshold": 5.}


def entry(key="defect/0054", group="test"):
    return {"group": group, "index": 0, "role": "fixed_miss" if group == "test" else "normal_high",
            "photo": {"key": key, "photo_id": key.split("/")[-1], "label": int(group == "test"), "regions": []}}


def raw_maps():
    a = np.zeros((12, 2, 256, 256), dtype=np.float32)
    a[:, 0, 100:108, 100:108] = 8.
    return a


def describe(item=None):
    item = item or entry()
    raw, c = raw_maps(), contract()
    sample = q.e.sample(item["photo"], q.p.tile_scores(raw, c["map_quantiles"]), c)
    masks = np.zeros((12, 256, 256), dtype=np.bool_)
    masks[:, 100:108, 100:108] = True
    case, arrays = q.describe(item, raw, masks, sample, c)
    return case, arrays, masks


def test_same_support_contributions_preserve_signed_values():
    case, arrays, _ = describe()
    for row in case["regions"]:
        assert row["st_half_contribution_on_combined_top656"] > 0
        assert row["ae_half_contribution_on_combined_top656"] < 0
        assert row["tile_score"] == pytest.approx(row["st_half_contribution_on_combined_top656"] + row["ae_half_contribution_on_combined_top656"])
        assert row["combined"]["location"]["top656_gt_overlap_bounds"] == [64, 64]
        assert row["combined"]["location"]["within_tile_auroc"] == 1.
        assert row["tile_margin"] < 0
    assert arrays[2].min() < 0 and not case["sample"]["alert"]


def test_ties_and_missing_gt_are_not_invented():
    values = np.zeros((256, 256))
    mask = np.zeros((256, 256), dtype=np.bool_)
    mask[0, :10] = True
    stats = q.map_stats(values, mask)
    assert stats["support_including_ties_pixels"] == 65536
    assert stats["location"]["top656_gt_overlap_bounds"] == [0, 10]
    assert stats["location"]["within_tile_auroc"] == .5
    case, _, _ = describe(entry("defect/0006"))
    assert not case["location_eligible"]
    assert all(r[m]["location"] is None for r in case["regions"] for m in ("st", "ae", "combined"))


@pytest.mark.parametrize("corruption", ["nan", "negative", "shape", "dtype", "score"])
def test_invalid_maps_or_attribution_rejected(corruption):
    item, raw, c = entry(), raw_maps(), contract()
    sample = q.e.sample(item["photo"], q.p.tile_scores(raw, c["map_quantiles"]), c)
    if corruption == "nan": raw[0, 0, 0, 0] = np.nan
    elif corruption == "negative": raw[0, 0, 0, 0] = -1
    elif corruption == "shape": raw = raw[:11]
    elif corruption == "dtype": raw = raw.astype(np.float64)
    else: sample["tile_scores"][0] += 1
    with pytest.raises(ValueError): q.describe(item, raw, [None]*12, sample, c)


def test_selection_is_frozen_and_contains_anchor_and_median():
    groups = {"test": [entry(k)["photo"] for k in q.e.POLICY["acceptance"]["regression_keys"]]}
    tiles = {}
    for g, n in (("development", 128), ("calibration", 169)):
        groups[g] = [entry(f"{g}/{i:04d}", g)["photo"] for i in range(n)]
        tiles[g] = np.tile(np.arange(n)[:, None], (1, 12)).tolist()
    c = contract() | {"threshold": 160.}
    selected = q.selections({"cohorts": groups, "contract": c}, {"tile_scores": tiles})
    assert len(selected) == 20
    assert [s["photo"]["key"] for s in selected[:12]] == q.e.POLICY["acceptance"]["regression_keys"]
    assert selected[-2]["photo"]["photo_id"] == "0160"
    assert selected[-1]["photo"]["photo_id"] == "0084"


def test_real_recovered_image_and_mask_exception(tmp_path, monkeypatch):
    original = Image.new("RGB", (256, 256), (20, 40, 60))
    out = io.BytesIO(); original.save(out, format="PNG"); png = out.getvalue()
    damaged = b"damaged fixture"
    repaired = tmp_path / "approved.png"; repaired.write_bytes(png)
    rgb_sha = hashlib.sha256(original.tobytes()).hexdigest()
    monkeypatch.setattr(q.ev, "DAMAGED_SHA256", hashlib.sha256(damaged).hexdigest())
    monkeypatch.setattr(q.ev.maps, "REPAIRED_RGB_SHA256", rgb_sha)
    item = entry()
    item["photo"]["regions"] = [{"member": q.ev.REPAIR_MEMBER, "member_sha256": hashlib.sha256(png).hexdigest(),
                                  "decoded_rgb_sha256": rgb_sha, "mask_member": None}]*12
    class Archive:
        def read(self, name):
            assert name == q.ev.REPAIR_MEMBER  # no other image or mask access
            return damaged
    prior = {"recovery.json": {"recovered_png_sha256": hashlib.sha256(png).hexdigest()}}
    images, masks = runner.images_and_masks(Archive(), item, repaired, prior)
    assert len(images) == 12 and not any(m.any() for m in masks)
    for image in images: image.close()
    repaired.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="recovered tile changed"):
        runner.images_and_masks(Archive(), item, repaired, prior)


@pytest.mark.parametrize("group", ["test", "calibration"])
def test_load_saved_maps_checks_receipts_and_replays_scores(tmp_path, group):
    item, c, raw = entry("defect/0054" if group == "test" else "calibration/0001", group), contract(), raw_maps()
    root = tmp_path / "saved"; (root / group).mkdir(parents=True)
    plan = {"contract": c, "fit_signature": "fit"}
    sig = q.fit.digest(plan) if group == "test" else "fit"
    sample = q.e.sample(item["photo"], q.p.tile_scores(raw, c["map_quantiles"]), c)
    gate = {"passed": True}
    q.fit.write_json(root / "gate.json", gate)
    with q.fit.phase_stage(root / group, "photo-000") as stage:
        np.save(stage / "maps.npy", raw, allow_pickle=False)
        files = ["maps.npy"]
        if group == "test":
            sample["photo_seconds"] = .2
            q.fit.write_json(stage / "sample.json", sample); files.append("sample.json")
        receipt = q.fit.seal_phase(stage, "photo-000", sig,
            {"photo_sha256": q.fit.digest(item["photo"]), "gate_path": "gate.json", "gate_sha256": q.fit.digest(gate)}, files)
    # seal_phase writes its receipt; read it as the reviewed handoff would.
    receipt = json.loads((root / group / "photo-000/receipt.json").read_bytes())
    data = {"plan.json": plan, "screen.json": {"predictions": [sample]},
            "sample-receipts.json": {"test/photo-000/receipt.json": receipt}}
    records = {"map-receipts.json": [receipt], "normal-results.json": {"tile_scores": {group: [sample["tile_scores"]]}}}
    actual, replay, _ = q.load_maps(item, data, records, root, root)
    assert np.array_equal(actual, raw) and replay["score"] == sample["score"]
    (root / group / "photo-000/maps.npy").write_bytes(b"changed")
    with pytest.raises(ValueError): q.load_maps(item, data, records, root, root)


def test_cpu_orchestration_export_resume_and_tamper(tmp_path, monkeypatch):
    item, c = entry(), contract()
    raw = raw_maps()
    sample = q.e.sample(item["photo"], q.p.tile_scores(raw, c["map_quantiles"]), c)
    identity = {"entries": [item], "contract": c}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w"): pass
    calls = []
    monkeypatch.setattr(q, "load_maps", lambda *a: (raw, sample, {"receipt_sha256": "reviewed"}))
    def images(*a):
        calls.append("images")
        return [Image.new("RGB", (256, 256), (50, 60, 70)) for _ in range(12)], [np.zeros((256, 256), dtype=np.bool_) for _ in range(12)]
    monkeypatch.setattr(runner, "images_and_masks", images)
    args = (tmp_path / "out", archive, tmp_path, tmp_path, tmp_path, {}, {}, {"previous": {}}, identity, tmp_path / "handoff.zip")
    report = runner.execute(*args)
    assert report["model_inference_calls"] == 0 and not report["threshold_changed"]
    assert runner.execute(*args) == report and calls == ["images"]
    with zipfile.ZipFile(args[-1]) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        assert all(not n.endswith((".pt", ".npy", ".npz")) for n in z.namelist())
    card = args[0] / "cases/case-00/card-1.png"
    card.write_bytes(b"changed")
    with pytest.raises(ValueError): runner.execute(*args)


def test_cli_and_fixed_signed_colors():
    path = Path(runner.__file__)
    result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    values = np.array([[-.05, 0., .5, 5.]])
    rendered = cards.colors(values)
    assert (rendered[0, 1] == 255).all()
    assert np.array_equal(rendered[0, 2], rendered[0, 3])
    assert values[0, 3] == 5.  # presentation clipping cannot change data
    imports = subprocess.run([sys.executable, "-c",
        "from scripts import review_mpi_efficientad_maps; import sys; assert 'torch' not in sys.modules"],
        cwd=path.parents[1], capture_output=True, text=True)
    assert imports.returncode == 0, imports.stderr


def test_main_passes_cli_paths_without_loading_a_model(tmp_path, monkeypatch):
    fit_root = tmp_path / "artifacts" / q.p.NAME
    evaluation_root = tmp_path / "artifacts/analyses" / q.e.NAME
    fit_root.mkdir(parents=True); evaluation_root.mkdir(parents=True)
    archive = tmp_path / "input.zip"; archive.write_bytes(b"fixture")
    monkeypatch.setattr(q.fit, "EXPECTED_ARCHIVE_SHA256", hashlib.sha256(b"fixture").hexdigest())
    seen = []
    def prepare(project, handoffs, actual_fit, actual_evaluation):
        assert project == tmp_path and handoffs == tmp_path / "artifacts/handoff"
        assert actual_fit == fit_root and actual_evaluation == evaluation_root
        return {}, {}, {"previous": {}}, {"entries": []}
    monkeypatch.setattr(q, "prepare", prepare)
    monkeypatch.setattr(runner, "execute", lambda *args: seen.append(args) or {"fixture": True})
    assert runner.main(["--project", str(tmp_path), "--archive", str(archive)]) == 0
    assert seen[0][0] == tmp_path / "artifacts/analyses" / q.NAME
    assert seen[0][-1] == tmp_path / "artifacts/handoff" / (q.NAME + ".zip")
    assert not (fit_root / "step-070000").exists()
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        runner.main(["--project", str(tmp_path), "--archive", str(archive)])
    assert len(seen) == 1


def test_interrupted_card_write_can_resume_without_deletion(tmp_path, monkeypatch):
    item, c, raw = entry(), contract(), raw_maps()
    sample = q.e.sample(item["photo"], q.p.tile_scores(raw, c["map_quantiles"]), c)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w"): pass
    monkeypatch.setattr(q, "load_maps", lambda *a: (raw, sample, {"receipt_sha256": "reviewed"}))
    monkeypatch.setattr(runner, "images_and_masks", lambda *a:
        ([Image.new("RGB", (256, 256)) for _ in range(12)], [None]*12))
    render = cards.render
    def interrupted(*args):
        if args[-1] == 1: raise KeyboardInterrupt()
        return render(*args)
    monkeypatch.setattr(cards, "render", interrupted)
    args = (tmp_path / "out", archive, tmp_path, tmp_path, tmp_path, {}, {}, {"previous": {}},
            {"entries": [item], "contract": c}, tmp_path / "handoff.zip")
    with pytest.raises(KeyboardInterrupt): runner.execute(*args)
    partials = list((args[0] / "cases").rglob("card-1.png"))
    assert len(partials) == 1 and not args[-1].exists()
    monkeypatch.setattr(cards, "render", render)
    assert runner.execute(*args)["cases"] == 1
    assert partials[0].exists()  # interrupted output is preserved, not treated as complete
