import json
import tempfile
import unittest
from pathlib import Path

from defectops.patchcore_evaluation import (
    EXPECTED_EVALUATION_PURPOSE,
    EXPECTED_PREREGISTERED_EVALUATION_PURPOSE,
    EXPECTED_REALIAD_EVALUATION_PURPOSE,
    EXPECTED_REPEAT_EVALUATION_PURPOSE,
    load_patchcore_evaluation_config,
)
from defectops.patchcore_experiment import (
    EXPECTED_COMPARISON_STATUS,
    EXPECTED_PREREGISTERED_STATUS,
    EXPECTED_REALIAD_STATUS,
    load_patchcore_experiment,
)


class PatchCoreExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest = json.loads(
            (cls.root / "data/manifests/transistor-v1.json").read_text(encoding="utf-8")
        )

    def test_patchcore_config_is_locked_to_exploratory_defaults(self):
        experiment = load_patchcore_experiment(
            self.root / "configs/patchcore-pilot.toml", self.manifest, seed=None
        )

        self.assertEqual(experiment["comparison_status"], EXPECTED_COMPARISON_STATUS)
        self.assertEqual(experiment["model"], "patchcore")
        self.assertEqual(experiment["backbone"], "wide_resnet50_2")
        self.assertEqual(experiment["layers"], ["layer2", "layer3"])
        self.assertEqual(experiment["coreset_sampling_ratio"], 0.1)
        self.assertEqual(experiment["num_neighbors"], 9)
        self.assertEqual(experiment["image_size"], 256)
        self.assertEqual(experiment["model_seed"], 42)

    def test_patchcore_config_rejects_confirmatory_relabeling(self):
        original = (self.root / "configs/patchcore-pilot.toml").read_text(encoding="utf-8")
        changed = original.replace(
            'comparison_status = "exploratory_after_padim_test_observed"',
            'comparison_status = "confirmatory"',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison_status"):
                load_patchcore_experiment(path, self.manifest, seed=None)

    def test_patchcore_repeat_config_discloses_observed_seed42_test(self):
        repeat = load_patchcore_experiment(
            self.root / "configs/patchcore-repeat-v1.toml", self.manifest, seed=43
        )
        self.assertEqual(repeat["model_seed"], 43)
        with self.assertRaisesRegex(ValueError, "configured model seeds"):
            load_patchcore_experiment(
                self.root / "configs/patchcore-repeat-v1.toml", self.manifest, seed=42
            )
        original = (self.root / "configs/patchcore-repeat-v1.toml").read_text(
            encoding="utf-8"
        )
        changed = original.replace(
            'repeat_status = "exploratory_repeat_after_patchcore_seed42_test_observed"',
            'repeat_status = "confirmatory"',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeat_status"):
                load_patchcore_experiment(path, self.manifest, seed=43)

    def test_patchcore_evaluation_uses_shared_metric_contract(self):
        config = load_patchcore_evaluation_config(
            self.root / "configs/patchcore-evaluation-v1.toml"
        )

        self.assertEqual(config["purpose"], EXPECTED_EVALUATION_PURPOSE)
        self.assertEqual(config["comparison_status"], EXPECTED_COMPARISON_STATUS)
        self.assertEqual(config["pixel_calibration_quantile"], 0.995)
        self.assertEqual(config["decision_rule"], "score >= threshold")
        self.assertEqual(config["bootstrap_seed"], 20260915)

        repeat = load_patchcore_evaluation_config(
            self.root / "configs/patchcore-evaluation-repeat-v1.toml"
        )
        self.assertEqual(repeat["purpose"], EXPECTED_REPEAT_EVALUATION_PURPOSE)

        preregistered = load_patchcore_evaluation_config(
            self.root / "configs/additional-patchcore-evaluation-v1.toml"
        )
        self.assertEqual(
            preregistered["purpose"], EXPECTED_PREREGISTERED_EVALUATION_PURPOSE
        )
        self.assertEqual(
            preregistered["comparison_status"], EXPECTED_PREREGISTERED_STATUS
        )

        realiad = load_patchcore_evaluation_config(
            self.root / "configs/realiad-patchcore-evaluation-v1.toml"
        )
        self.assertEqual(realiad["purpose"], EXPECTED_REALIAD_EVALUATION_PURPOSE)
        self.assertEqual(realiad["comparison_status"], EXPECTED_REALIAD_STATUS)
        self.assertEqual(realiad["ground_truth_threshold"], 128)

    def test_additional_category_configs_are_preregistered_before_test(self):
        for category, config_name in (
            ("cable", "cable-patchcore-v1.toml"),
            ("metal_nut", "metal-nut-patchcore-v1.toml"),
        ):
            manifest = {
                "protocol": "A-normal-only-v1",
                "split_seed": 42,
                "categories": [category],
            }
            experiment = load_patchcore_experiment(
                self.root / "configs" / config_name, manifest, seed=43
            )
            self.assertEqual(
                experiment["comparison_status"], EXPECTED_PREREGISTERED_STATUS
            )
            self.assertEqual(experiment["model_seed"], 43)

    def test_preregistered_patchcore_config_rejects_scope_relabeling(self):
        original = (self.root / "configs/cable-patchcore-v1.toml").read_text(
            encoding="utf-8"
        )
        changed = original.replace(
            'comparison_status = "preregistered_before_cable_and_metal_nut_test_observed"',
            'comparison_status = "confirmatory"',
        )
        manifest = {
            "protocol": "A-normal-only-v1",
            "split_seed": 42,
            "categories": ["cable"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.toml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "comparison_status"):
                load_patchcore_experiment(path, manifest, seed=42)

    def test_realiad_configs_are_locked_to_registered_scope(self):
        for category, config_name in (
            ("realiad_transistor1", "realiad-transistor1-patchcore-v1.toml"),
            ("realiad_plastic_nut", "realiad-plastic-nut-patchcore-v1.toml"),
        ):
            manifest = {
                "dataset": "real_iad",
                "protocol": "A-normal-only-v1",
                "split_seed": 42,
                "categories": [category],
            }
            experiment = load_patchcore_experiment(
                self.root / "configs" / config_name, manifest, seed=44
            )
            self.assertEqual(experiment["comparison_status"], EXPECTED_REALIAD_STATUS)
            self.assertEqual(experiment["model_seed"], 44)


if __name__ == "__main__":
    unittest.main()
