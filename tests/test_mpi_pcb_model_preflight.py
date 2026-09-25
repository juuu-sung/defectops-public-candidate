import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_mpi_pcb_models import (
    checked_model_paths,
    load_protocol,
    normal_threshold,
    photo_score,
)


class MpiPcbModelPreflightTests(unittest.TestCase):
    def test_protocol_lock_rejects_changed_score_rule_or_test_access(self):
        project = Path(__file__).resolve().parents[1]
        valid = load_protocol(project / "configs" / "mpi-pcb-model-protocol-v1.toml")
        self.assertFalse(valid["test_image_decoding_allowed"])
        self.assertEqual(set(valid["models"]), {"realiad_pcb", "visa_pcb3_v2"})
        original = (project / "configs" / "mpi-pcb-model-protocol-v1.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "altered.toml"
            for before, after in (
                ("maximum_of_12_region_scores", "average_of_12_region_scores"),
                ("test_image_decoding_allowed = false", "test_image_decoding_allowed = true"),
                ("calibration_quantile = 0.95", "calibration_quantile = 0.90"),
            ):
                altered.write_text(original.replace(before, after), encoding="utf-8")
                with self.subTest(before=before), self.assertRaises(ValueError):
                    load_protocol(altered)

    def test_photo_aggregation_and_normal_only_quantile(self):
        self.assertEqual(photo_score([float(index) for index in range(12)]), 11.0)
        self.assertEqual(normal_threshold([float(index) for index in range(169)]), 160.0)
        for scores in ([1.0] * 11, [1.0] * 11 + [float("nan")]):
            with self.assertRaises(ValueError):
                photo_score(scores)
        with self.assertRaises(ValueError):
            normal_threshold([0.0] * 168)

    def test_checkpoint_reports_and_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            real = project / "artifacts" / "realiad-followup-development-v1" / "pcb"
            visa = project / "artifacts" / "visa-pcb-normal-bank-development-v2" / "pcb3"
            real.mkdir(parents=True)
            visa.mkdir(parents=True)
            real_model = real / "fine.pt"
            visa_model = visa / "model.pt"
            real_model.write_bytes(b"frozen realiad weights")
            visa_model.write_bytes(b"frozen visa weights")
            real_sha = hashlib.sha256(real_model.read_bytes()).hexdigest()
            visa_sha = hashlib.sha256(visa_model.read_bytes()).hexdigest()
            protocol = {"models": {
                "realiad_pcb": {"checkpoint_sha256": real_sha},
                "visa_pcb3_v2": {"checkpoint_sha256": visa_sha},
            }}
            (real / "category-result.json").write_text(json.dumps({
                "status": "evaluated_preregistered_development", "category": "realiad_pcb",
                "seed": 42, "final_holdout_evaluated": False,
                "automatic_pass_allowed": False,
                "layer_groups": {"layer1-layer2": {
                    "layers": ["layer1", "layer2"], "checkpoint": "fine.pt",
                    "checkpoint_sha256": real_sha,
                }},
            }), encoding="utf-8")
            (visa / "fit.json").write_text(json.dumps({
                "status": "fit_calibrated_post_test_exploratory_normal_bank", "category": "pcb3",
                "model_seed": 42, "checkpoint_sha256": visa_sha,
                "normal_train_count": 849, "normal_calibration_count": 56,
                "official_test_scores_computed": 0, "automatic_pass_allowed": False,
            }), encoding="utf-8")
            checked = checked_model_paths(project, protocol)
            self.assertEqual(checked["realiad_pcb"]["checkpoint_sha256"], real_sha)
            visa_model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                checked_model_paths(project, protocol)


if __name__ == "__main__":
    unittest.main()
