"""No GPU required: frozen selections, descriptive math, fail-closed replay and resume."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
import pytest
from scripts import mpi_efficientad_trajectory as t
from scripts import mpi_efficientad_trajectory_cards as cards
from scripts import run_mpi_efficientad_trajectory as runner


def raw():
    a = np.zeros((12, 2, 256, 256), dtype=np.float32)
    a[:, 0, 100:108, 100:108] = 1.
    return a


def contract():
    return {"map_quantiles": [{"q90": .1, "q995": 1.1}]*2,
            "centers": [0.]*12, "scales": [1.]*12, "threshold": 5.}


def test_raw_summary_is_not_a_decision_or_normalized_score():
    masks = np.zeros((12, 256, 256), dtype=bool); masks[:, 100:108, 100:108] = True
    summary = t.raw_summary(raw(), masks, True)
    assert summary[0]["st"]["top656_mean"] == pytest.approx(64/656)
    assert summary[0]["st"]["location"]["within_tile_auroc"] == 1.
    assert summary[0]["ae"]["location"]["top656_gt_overlap_bounds"] == [0, 64]
    assert not any(k.startswith('display') for k in summary[0]["st"])
    assert "alert" not in json.dumps(summary) and "threshold" not in json.dumps(summary)
    excluded = t.raw_summary(raw(), masks, False)
    assert all(r[c]["location"] is None for r in excluded for c in ("st", "ae"))


@pytest.mark.parametrize("fault", ["values", "shape", "nan", "dtype", "decision"])
def test_final_replay_stops_for_difference_or_flip(fault):
    a, b, c = raw(), raw(), contract()
    photo = {"key": "defect/0048", "photo_id": "0048", "label": 1}
    assert t.final_replay(a, b, photo, c)["passed"]
    if fault == "values": a += .01
    elif fault == "shape": a = a[:11]
    elif fault == "nan": a[0, 0, 0, 0] = np.nan
    elif fault == "dtype": a = a.astype(np.float64)
    else:
        c["threshold"] = t.q.e.sample(photo, t.p.tile_scores(b, c["map_quantiles"]), c)["score"]
        a *= .999999
    with pytest.raises(ValueError): t.final_replay(a, b, photo, c)


def test_checkpoint_receipt_and_step_are_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "STEPS", (2, 1, 3))
    plan = {"frozen": True}; signature = t.fit.digest(plan)
    receipts = []
    for step in (1, 2, 3):
        name = f"step-{step:06d}"
        with t.fit.phase_stage(tmp_path, name) as stage:
            (stage/"model.pt").write_bytes(b"not deserialized in CPU test")
            np.save(stage/"reference.npy", np.zeros(1))
            t.fit.write_json(stage/"progress.json", {"step": step,
                "losses": [{"step": i, "loss_components": [1., 2., 3.]} for i in range(1, step+1)],
                "monitor_raw_map_means": [[1., 2.]]*12})
            receipts.append(t.fit.seal_phase(stage, name, signature, {}, t.CHECKPOINT_FILES))
    records = {"plan.json": plan, "checkpoint-receipts.json": receipts}
    assert t.checkpoints(tmp_path, records)["2"]["last_100_loss_component_means"] == [1., 2., 3.]
    (tmp_path/"step-000001/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError): t.checkpoints(tmp_path, records)


def test_common_scale_cards_and_atlas(tmp_path):
    image = Image.new("RGB", (256, 256), (80, 100, 120))
    item = {"group": "test", "photo": {"key": "defect/0048", "photo_id": "0048"}}
    maps = {str(s): raw() for s in t.STEPS}
    masks = np.zeros((12, 256, 256), dtype=bool)
    summaries = {str(s): {"regions": t.raw_summary(maps[str(s)], masks, True)} for s in t.STEPS}
    with cards.render(item, [image]*12, masks, maps, summaries, contract()["map_quantiles"], 0) as card:
        assert card.size == (1792, 1900)
        card.save(tmp_path/"card.png")
        # Same ST values at 10k/30k/70k produce identical pixels, no per-step rescaling.
        assert np.array_equal(np.asarray(card)[100:356, 256:512], np.asarray(card)[100:356, 512:768])
    with cards.atlas([image]*16, [f"{i:04d}" for i in range(16)], t.fit.REGIONS[0], 0) as atlas:
        assert atlas.size == (768, 928)
    image.close()


class Monitor:
    def __init__(self, runtime, prior, started, policy): self.runtime = runtime
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def check(self): pass
    def snapshot(self): return {"test_only": True}


def setup_fake(tmp_path, monkeypatch):
    entries = [{"group": "test" if i < 12 else "calibration", "index": i,
        "photo": {"key": f"defect/{i:04d}" if i < 12 else f"calibration/{i:04d}",
                  "photo_id": f"{i:04d}", "label": int(i < 12)}, "role": "test"} for i in range(20)]
    identity = {"entries": entries, "training": {}, "fit_identity": "frozen", "contract": contract()}
    archive = tmp_path/"archive.zip"
    with zipfile.ZipFile(archive, "w"): pass
    monkeypatch.setattr(runner, "images_and_masks", lambda *a: ([Image.new("RGB", (256, 256)) for _ in range(12)], [None]*12))
    monkeypatch.setattr(t, "raw_summary", lambda *a: [])
    monkeypatch.setattr(t.q, "load_maps", lambda *a: (raw(), None, None))
    monkeypatch.setattr(runner, "atlas_phase", lambda *a: [])
    monkeypatch.setattr(runner, "make_cards", lambda *a: None)
    monkeypatch.setattr(runner, "finish", lambda root, handoff, signature, identity, report: report)
    class Runtime:
        loaded = []
        fail = None
        def __init__(self, path, signature, step): self.step = step; self.loaded.append(step)
        def input_audit(self, image): return {"fixed_teacher_and_statistics_sha256": {"teacher.": "same", "mean_std.": "same"}}
        def predict(self, image):
            if self.step == self.fail: raise KeyboardInterrupt("simulated interruption")
            return raw()[0]
        def release(self): pass
    def execute():
        return runner.execute(tmp_path/"out", archive, tmp_path/"fit", tmp_path/"eval", None,
            {}, {}, {"previous": {}}, identity, tmp_path/"handoff.zip", Runtime, Monitor)
    return Runtime, execute, identity


def test_execution_order_resume_and_no_new_training(tmp_path, monkeypatch):
    runtime, execute, _ = setup_fake(tmp_path, monkeypatch)
    runtime.fail = 10000
    with pytest.raises(KeyboardInterrupt, match="simulated"): execute()
    assert runtime.loaded == [70000, 10000]
    preserved = (tmp_path/"out/step-070000/case-00/maps.npy").read_bytes()
    runtime.fail = None
    result = execute()
    assert runtime.loaded == [70000, 10000, 10000, 30000]
    assert not result["training_performed"] and result["earlier_checkpoint_decisions_computed"] == 0
    assert len(result["attempts"]) == 2
    assert (tmp_path/"out/step-070000/case-00/maps.npy").read_bytes() == preserved
    execute()
    assert runtime.loaded == [70000, 10000, 10000, 30000]


def test_final_failure_prevents_loading_earlier_checkpoints(tmp_path, monkeypatch):
    runtime, execute, _ = setup_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(t.q, "load_maps", lambda *a: (raw()+1, None, None))
    with pytest.raises(ValueError, match="replay mismatch"): execute()
    assert runtime.loaded == [70000]
    assert not (tmp_path/"out/result").exists()
    with pytest.raises(ValueError, match="needs review"): execute()


def test_real_card_export_pipeline_with_fake_gpu(tmp_path, monkeypatch):
    real_finish, real_cards, real_summary = runner.finish, runner.make_cards, t.raw_summary
    runtime, execute, _ = setup_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "finish", real_finish)
    monkeypatch.setattr(runner, "make_cards", real_cards)
    monkeypatch.setattr(t, "raw_summary", real_summary)
    monkeypatch.setattr(runner, "images_and_masks", lambda *a: (
        [Image.new("RGB", (256, 256), (80, 100, 120)) for _ in range(12)],
        np.zeros((12, 256, 256), dtype=bool)))
    def atlas(root, archive, training, signature, check):
        names = [f"region-{i:02d}-page-{j}.png" for i in range(12) for j in range(1, 5)]
        with t.fit.phase_stage(root, "training-atlas") as stage:
            for name in names:
                with Image.new("RGB", (16, 16)) as image: image.save(stage/name)
            t.fit.seal_phase(stage, "training-atlas", signature, {}, names)
    monkeypatch.setattr(runner, "atlas_phase", atlas)
    result = execute()
    assert result['status'] == 'checkpoint_trajectory_completed_exploratory'
    with zipfile.ZipFile(tmp_path/'handoff.zip') as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert len([n for n in manifest if n.endswith('.png')]) == 88
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        summary = json.loads(z.read('summaries.json'))
        assert set(summary) == {'10000', '30000', '70000'}
        assert all(s['final_replay'] is None for s in summary['10000']+summary['30000'])
        assert all(s['final_replay']['passed'] for s in summary['70000'])
    execute()  # Fully completed run verifies existing maps/cards/bundle without a GPU load.
    assert runtime.loaded == [70000, 10000, 30000]
    (tmp_path/'out/step-010000/case-00/maps.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='incomplete or changed'): execute()


def test_atlas_covers_all_training_rows_without_inference(tmp_path, monkeypatch):
    seen = []
    def checked(archive, row):
        seen.append(row['id'])
        return Image.new('RGB', (256, 256))
    monkeypatch.setattr(t.fit, 'checked_image', checked)
    training = {'photo_ids': [f'{i:04d}' for i in range(64)], 'rows': [{'id': i} for i in range(768)]}
    paths = runner.atlas_phase(tmp_path, None, training, 'signature', lambda: None)
    assert len(paths) == 49 and sorted(seen) == list(range(768))
    runner.atlas_phase(tmp_path, None, training, 'signature', lambda: None)
    assert len(seen) == 768


def test_export_inventory_and_tampering(tmp_path):
    root = tmp_path/"root"; root.mkdir()
    (root/"card.png").write_bytes(b"image fixture")
    payload = {"report.json": {"training": False}, "index.html": "<html></html>"}
    handoff = tmp_path/"delivery.zip"
    runner.export(root, handoff, "signature", payload, ["card.png"])
    runner.export(root, handoff, "signature", payload, ["card.png"])
    with zipfile.ZipFile(handoff) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist()) == {*manifest, 'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
    with zipfile.ZipFile(handoff, "a") as z: z.writestr("unexpected", b"changed")
    with pytest.raises(ValueError, match="Existing ZIP"): runner.export(root, handoff, "signature", payload, ["card.png"])


def test_cli_help_and_inference_only_boundary():
    result = subprocess.run([sys.executable, "scripts/run_mpi_efficientad_trajectory.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    source = Path("scripts/mpi_efficientad_trajectory_runtime.py").read_text()
    assert 'weights_only=True' in source and 'TrainingRuntime.tensor(self, image)' in source
    assert '.backward(' not in source and 'optimizer' not in source.replace('optimizer/fit', '')
    assert t.POLICY["earlier_checkpoint_thresholds"] is None
