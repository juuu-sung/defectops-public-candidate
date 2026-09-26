"""Real orchestration and image checks with an explicitly synthetic GPU boundary."""

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

from scripts import fit_mpi_pcb_normal_bank as fit
from scripts.prepare_mpi_pcb_normal_bank import EXPECTED_CONFIG
from tests.test_mpi_pcb_selected_normal_decode_audit import corrupt_idat_crc

PROJECT = Path(__file__).resolve().parents[1]


def png_row(photo, region, value):
    image = Image.new("RGB", (256, 256), (value, 7, 9))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    return raw, {"photo_id": photo, "region": region,
                 "member": f"extracted/{region}/train/good/{photo}.png",
                 "member_sha256": hashlib.sha256(raw).hexdigest(),
                 "decoded_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


class FakeRuntime:
    """No torch, CUDA, or actual PatchCore performance claims in these tests."""

    def __init__(self, config, calls, *, fail_predict=False, reload_offset=0.0):
        self.config, self.calls = config, calls
        self.fail_predict, self.reload_offset = fail_predict, reload_offset
        self.offset = 0.0
        calls.append("created")

    def fit(self, images):
        self.calls.append("fit")
        count = 0
        for image in images:
            assert image.size == (256, 256)
            image.close()
            count += 1
        budget = fit.patch_budget(self.config)
        assert count == budget["training_tiles"]
        return {"candidate_embeddings": budget["candidate_embeddings"],
                "memory_bank_shape": [budget["coreset_embeddings"], self.config["feature_dimension"]],
                "feature_extraction_seconds": 0.0, "coreset_seconds": 0.0,
                "runtime": self.environment(), "memory": self.memory()}

    def save(self, path, context):
        self.calls.append("save")
        path.write_bytes(fit.canonical({"fake_checkpoint_only": True, "context": context}))

    def load(self, path, context):
        self.calls.append("load")
        assert json.loads(path.read_bytes())["context"] == context
        self.offset = self.reload_offset

    def predict(self, image):
        self.calls.append("predict")
        if self.fail_predict:
            raise RuntimeError("synthetic calibration interruption")
        value = image.getpixel((0, 0))[0] + self.offset
        return float(value), np.full((256, 256), value, dtype=np.float32)

    def reset_peak(self):
        pass

    def synchronize(self):
        pass

    def memory(self):
        return {"kind": "synthetic_not_measured"}

    def environment(self):
        return {"kind": "synthetic_test_runtime"}

    def release(self):
        self.calls.append("release")


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    regions = fit.REGIONS[:2]
    monkeypatch.setattr(fit, "REGIONS", regions)
    config = {**EXPECTED_CONFIG, "training_original_photos": 2,
              "calibration_original_photos": 2, "regions_per_photo": 2}
    preparation = {"training_photo_ids": ["0001", "0002"], "calibration_photo_ids": ["0003", "0004"]}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        for role in ("training", "calibration"):
            rows = []
            for p in preparation[f"{role}_photo_ids"]:
                for index, region in enumerate(regions):
                    raw, row = png_row(p, region, int(p) * 10 + index)
                    source.writestr(row["member"], raw)
                    rows.append(row)
            preparation[f"{role}_rows"] = rows
        # If orchestration ever tries to decode test data, this is not a valid PNG.
        source.writestr("extracted/grid1-1_s1024/test/defect/0000.png", b"DO NOT READ TEST")
    return {"root": tmp_path / "run", "archive_path": archive, "preparation": preparation,
            "config": config, "execution": {"synthetic": True}, "git": {"commit": "synthetic"},
            "handoff": tmp_path / "fit.zip"}


def execute(args, calls, **runtime_options):
    return fit.run_pipeline(**args, runtime_factory=lambda cfg: FakeRuntime(cfg, calls, **runtime_options))


def test_full_pipeline_checks_images_calibrates_reloads_and_exports_json_only(small_run):
    calls = []
    result = execute(small_run, calls)
    assert result["status"] == fit.FINAL_STATUS
    assert result["training_region_images"] == result["calibration_region_images"] == 4
    assert result["image_threshold"] == 41.0  # Per-photo max, then higher quantile.
    assert calls.count("fit") == calls.count("save") == calls.count("load") == 1
    assert calls.count("predict") == 8
    assert calls.index("save") < calls.index("predict") < calls.index("load")
    assert result["reload_verification"]["maximum_region_difference"] == 0.0
    assert result["test_scores_computed"] == result["test_images_decoded"] == result["test_masks_decoded"] == 0
    assert result["automatic_pass_allowed"] is result["confirmatory_claim_allowed"] is False
    assert result["pixel_threshold_calibrated"] is False
    with zipfile.ZipFile(small_run["handoff"]) as source:
        assert len(source.namelist()) == 6
        assert all(p.endswith(".json") for p in source.namelist())
        manifest = json.loads(source.read("handoff-manifest.json"))
        assert len(manifest) == 5
        assert all(hashlib.sha256(source.read(k)).hexdigest() == v for k, v in manifest.items())
        cal = json.loads(source.read("calibration.json"))
        assert cal["receipt"]["model_origin"] == "live_fit"
        assert cal["receipt"]["runtime"]["kind"] == "synthetic_test_runtime"
        assert json.loads(source.read("reload.json"))["receipt"]["model_origin"] == "loaded_checkpoint"


def test_completed_rerun_does_not_construct_gpu_runtime_or_change_zip(small_run):
    original = execute(small_run, [])
    before = small_run["handoff"].read_bytes()
    def forbidden(config):
        pytest.fail("Completed phases must not run GPU inference again")
    assert fit.run_pipeline(**small_run, runtime_factory=forbidden) == original
    assert small_run["handoff"].read_bytes() == before


def test_calibration_interruption_preserves_fit_and_resumes_from_checkpoint(small_run):
    calls = []
    with pytest.raises(RuntimeError, match="interruption"):
        execute(small_run, calls, fail_predict=True)
    root = small_run["root"]
    before = (root / "fit/model.pt").read_bytes()
    receipt_before = (root / "fit/receipt.json").read_bytes()
    assert len(list(root.glob(".calibration-attempt-*"))) == 1
    assert not (root / "calibration").exists() and not small_run["handoff"].exists()
    assert calls[-1] == "release"
    resumed = []
    execute(small_run, resumed)
    assert "fit" not in resumed and resumed.count("load") == 2
    assert (root / "fit/model.pt").read_bytes() == before
    assert (root / "fit/receipt.json").read_bytes() == receipt_before
    assert json.loads((root / "calibration/receipt.json").read_bytes())["model_origin"] == "loaded_checkpoint"
    assert len(list(root.glob(".calibration-attempt-*"))) == 1


@pytest.mark.parametrize("relative", ["fit/model.pt", "fit/receipt.json", "calibration/photos.json", "reload/photos.json"])
def test_corrupt_completed_phase_rejected_without_gpu_or_overwrite(small_run, relative):
    execute(small_run, [])
    path = small_run["root"] / relative
    path.write_bytes(b"corrupted")
    calls = []
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(small_run, calls)
    assert calls == [] and path.read_bytes() == b"corrupted"


def test_changed_execution_rejected_without_mutating_outputs(small_run):
    execute(small_run, [])
    before = (small_run["root"] / "fit.json").read_bytes()
    small_run["execution"] = {"synthetic": "changed"}
    calls = []
    with pytest.raises(ValueError, match="execution/config differs"):
        execute(small_run, calls)
    assert calls == [] and (small_run["root"] / "fit.json").read_bytes() == before


def test_out_of_order_phase_stops_before_gpu(small_run):
    root = small_run["root"]
    (root / "calibration").mkdir(parents=True)
    (root / "input-lock.json").write_bytes(fit.canonical({"execution": small_run["execution"], "config": small_run["config"]}))
    calls = []
    with pytest.raises(ValueError, match="out of order"):
        execute(small_run, calls)
    assert calls == []


def test_reload_drift_stops_before_final_report_preserving_fit_and_calibration(small_run):
    with pytest.raises(ValueError, match="Reload score difference"):
        execute(small_run, [], reload_offset=0.1)
    root = small_run["root"]
    assert (root / "fit/model.pt").is_file() and (root / "calibration/photos.json").is_file()
    assert not (root / "reload").exists() and not (root / "fit.json").exists()
    assert not small_run["handoff"].exists()
    assert len(list(root.glob(".reload-attempt-*"))) == 1
    calls = []
    execute(small_run, calls)
    assert "fit" not in calls and calls.count("predict") == 4


def test_region_reload_drift_detected_even_if_photo_maximum_does_not_change(small_run):
    execute(small_run, [])
    first = json.loads((small_run["root"] / "calibration/photos.json").read_bytes())
    second = copy.deepcopy(first)
    second[0]["regions"][0]["score"] += 0.01
    assert second[0]["score"] == first[0]["score"]
    with pytest.raises(ValueError, match="Reload score difference"):
        fit.reload_differences(first, second, small_run["preparation"]["calibration_photo_ids"])


def test_handoff_is_not_overwritten(small_run):
    small_run["handoff"].write_bytes(b"existing unrelated data")
    with pytest.raises(zipfile.BadZipFile):
        execute(small_run, [])
    assert small_run["handoff"].read_bytes() == b"existing unrelated data"
    assert (small_run["root"] / "fit.json").is_file()


def test_normal_image_checks_block_test_paths_traversal_and_changed_pixels(small_run):
    row = small_run["preparation"]["training_rows"][0]
    with zipfile.ZipFile(small_run["archive_path"]) as source:
        with fit.checked_image(source, row) as image:
            assert image.size == (256, 256)
        for changes in ({"member": row["member"].replace("train/good", "test/defect")},
                        {"photo_id": "../0001"}, {"region": "../../test"}):
            with pytest.raises(ValueError, match="non-training"):
                fit.checked_image(source, {**row, **changes})
        with pytest.raises(ValueError, match="bytes changed"):
            fit.checked_image(source, {**row, "member_sha256": "0" * 64})
        with pytest.raises(ValueError, match="pixels changed"):
            fit.checked_image(source, {**row, "decoded_rgb_sha256": "0" * 64})


def test_corrupt_png_rejected_even_with_matching_file_hash(tmp_path):
    raw, row = png_row("0001", fit.REGIONS[0], 10)
    raw = corrupt_idat_crc(raw)
    row["member_sha256"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as source:
        source.writestr(row["member"], raw)
    with zipfile.ZipFile(path) as source, pytest.raises(SyntaxError, match="checksum"):
        fit.checked_image(source, row)


def test_spatial_plan_is_exactly_1024_of_4096_patches():
    indices = fit.spatial_patch_indices(64, 2)
    assert len(indices) == 1024 and indices[:3] == [0, 2, 4] and indices[-1] == 4030
    class Embeddings:
        shape = (4096, 768)
        def index_select(self, axis, selected):
            assert axis == 0 and selected == indices
            return "selected"
    store = [Embeddings()]
    fit.retain_spatial_patches(store, 1, indices, EXPECTED_CONFIG)
    assert store == ["selected"]
    with pytest.raises(ValueError, match="layout differs"):
        fit.retain_spatial_patches([Embeddings()], 2, indices, EXPECTED_CONFIG)


def test_photo_threshold_uses_higher_quantile_not_interpolation():
    assert fit.threshold([{"score": float(x)} for x in range(169)]) == 160.0


def test_run_lock_blocks_concurrency_and_releases_after_exception(tmp_path):
    with fit.locked_run(tmp_path):
        with pytest.raises(RuntimeError, match="Another MPI"):
            with fit.locked_run(tmp_path):
                pytest.fail("Concurrent lock acquired")
    with pytest.raises(RuntimeError, match="interrupted"):
        with fit.locked_run(tmp_path):
            raise RuntimeError("interrupted")
    with fit.locked_run(tmp_path):
        assert (tmp_path / ".run.lock").is_file()


def test_phase_failure_preserves_partial_attempt(tmp_path):
    with pytest.raises(RuntimeError, match="interrupted"):
        with fit.phase_stage(tmp_path, "fit") as stage:
            (stage / "partial").write_bytes(b"preserve")
            raise RuntimeError("interrupted")
    assert not (tmp_path / "fit").exists()
    assert next(tmp_path.glob(".fit-attempt-*/partial")).read_bytes() == b"preserve"


def test_execution_code_hashes_are_lf_normalized(tmp_path):
    baseline = fit.execution_identity(PROJECT, EXPECTED_CONFIG)
    for relative in baseline["source_files_lf_sha256"]:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((PROJECT / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert fit.execution_identity(tmp_path, EXPECTED_CONFIG) == baseline


def test_cli_help_without_gpu_dependencies_and_error_reporting(tmp_path):
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/fit_mpi_pcb_normal_bank.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    assert fit.main(["--project", str(tmp_path), "--archive", str(tmp_path / "missing.zip")]) == 1


def test_preparation_wrong_zip_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "not-reviewed.zip"
    path.write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="not the reviewed"):
        fit.load_preparation(path, EXPECTED_CONFIG)
