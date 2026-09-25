import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts import calibrate_mpi_pcb_normals as calibration


def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buffer, format="PNG")
    return buffer.getvalue()


class MpiPcbNormalCalibrationTests(unittest.TestCase):
    def test_existing_fit_metadata_builds_both_frozen_model_contexts(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            real = project / "artifacts" / "realiad-followup-development-v1" / "pcb"
            visa = project / "artifacts" / "visa-pcb-normal-bank-development-v2" / "pcb3"
            real.mkdir(parents=True)
            visa.mkdir(parents=True)
            (real / "category-result.json").write_text(json.dumps({
                "layer_groups": {"layer1-layer2": {"layers": ["layer1", "layer2"]}},
                "development_manifest_sha256": "a" * 64,
                "execution_config_sha256": "b" * 64,
            }), encoding="utf-8")
            (visa / "fit.json").write_text(json.dumps({
                "layers": ["layer1", "layer2"],
                "plan_sha256": "c" * 64,
                "execution_config_sha256": "d" * 64,
            }), encoding="utf-8")
            locked = {"backbone": "wide_resnet50_2", "image_size": 256, "num_neighbors": 9,
                      "layers": ["layer1", "layer2"]}
            protocol = {"models": {"realiad_pcb": locked, "visa_pcb3_v2": locked}}
            base = {**locked, "coreset_sampling_ratio": 0.10}
            with mock.patch.object(calibration, "load_realiad_followup_protocol", return_value={"base_model": base}):
                with mock.patch.object(calibration, "load_execution", return_value={**base, "config_sha256": "d" * 64}):
                    contexts = calibration.model_contexts(project, protocol)
            self.assertEqual(contexts["realiad_pcb"]["checkpoint_context"]["manifest_sha256"], "a" * 64)
            self.assertEqual(contexts["visa_pcb3_v2"]["checkpoint_context"]["manifest_sha256"], "c" * 64)

    def test_selected_training_normals_only_and_photo_aggregation(self):
        photo_ids = ["0001", "0002"]
        rows = []
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as writer:
            for index, region in enumerate(calibration.REGIONS):
                for photo_id in photo_ids:
                    raw = png_bytes((index + (100 if photo_id == "0002" else 0), 10, 20))
                    name = f"extracted/{region}/train/good/{photo_id}.png"
                    writer.writestr(name, raw)
                    with Image.open(io.BytesIO(raw)) as image:
                        pixels = hashlib.sha256(image.tobytes()).hexdigest()
                    rows.append({
                        "region": region, "photo_id": photo_id, "member": name,
                        "member_sha256": hashlib.sha256(raw).hexdigest(),
                        "decoded_rgb_sha256": pixels,
                    })
            writer.writestr("extracted/grid1-1_s1024/test/defect/0001.png", b"invalid test bytes")

        def fake_predict(_model, image, _recipe, _gpu):
            level = float(image.getpixel((0, 0))[0])
            return level, np.full((256, 256), level, dtype=np.float32)

        with zipfile.ZipFile(io.BytesIO(archive_bytes.getvalue())) as source:
            with mock.patch.object(source, "read", wraps=source.read) as reader:
                result = calibration.score_normal_cohort(
                    source, rows, photo_ids, None, {}, (), predict=fake_predict,
                )
        self.assertEqual([photo["photo_id"] for photo in result], photo_ids)
        self.assertEqual([photo["score"] for photo in result], [11.0, 111.0])
        self.assertEqual([len(photo["regions"]) for photo in result], [12, 12])
        self.assertEqual(reader.call_count, 24)
        self.assertTrue(all("/train/good/" in call.args[0] for call in reader.call_args_list))

    def test_modified_normal_png_fails_before_inference(self):
        name = "extracted/grid1-1_s1024/train/good/0001.png"
        raw = png_bytes((20, 30, 40))
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as writer:
            writer.writestr(name, raw)
        rows = [{
            "region": region, "photo_id": "0001",
            "member": f"extracted/{region}/train/good/0001.png",
            "member_sha256": "0" * 64,
            "decoded_rgb_sha256": "0" * 64,
        } for region in calibration.REGIONS]
        with zipfile.ZipFile(io.BytesIO(archive_bytes.getvalue())) as source:
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                calibration.score_normal_cohort(
                    source, rows, ["0001"], None, {}, (),
                    predict=lambda *_: self.fail("Inference must not run on changed bytes"),
                )

    def test_preflight_gate_rejects_changed_or_open_test_report(self):
        models = {
            "realiad_pcb": {"checkpoint_path": "C:/one.pt", "checkpoint_sha256": "a" * 64},
            "visa_pcb3_v2": {"checkpoint_path": "C:/two.pt", "checkpoint_sha256": "b" * 64},
        }
        protocol = {"config_sha256": "c" * 64}
        valid = {
            "status": "mpi_pcb_model_protocol_locked_test_sealed",
            "protocol_config_sha256": protocol["config_sha256"],
            "normal_preflight_v4_sha256": calibration.EXPECTED_V4_SHA256,
            "normal_photo_ids_sha256": calibration.EXPECTED_IDS_SHA256,
            "archive_sha256": calibration.EXPECTED_ARCHIVE_SHA256,
            "models": models,
            "normal_original_photos": 169,
            "regions_per_photo": 12,
            "image_score": "mean_of_top_1pct_anomaly_map_then_max_of_12_regions",
            "threshold": "model_specific_95pct_higher_of_169_normal_photo_scores_not_yet_computed",
            "normal_scores_computed": 0,
            "test_images_decoded": 0,
            "test_masks_decoded": 0,
            "test_scores_computed": 0,
            "automatic_pass_allowed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(calibration, "EXPECTED_MODEL_PREFLIGHT_SHA256", digest):
                with mock.patch.object(calibration, "checked_model_paths", return_value=models):
                    self.assertEqual(calibration.validate_model_preflight(path, Path(temporary), protocol), valid)
                    for field, value in (("test_images_decoded", 1), ("normal_scores_computed", 1)):
                        changed = {**valid, field: value}
                        path.write_text(json.dumps(changed), encoding="utf-8")
                        changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        with mock.patch.object(calibration, "EXPECTED_MODEL_PREFLIGHT_SHA256", changed_digest):
                            with self.subTest(field=field), self.assertRaises(ValueError):
                                calibration.validate_model_preflight(path, Path(temporary), protocol)


if __name__ == "__main__":
    unittest.main()
