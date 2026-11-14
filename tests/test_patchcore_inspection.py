import unittest
from pathlib import Path

from defectops.padim_inspection import (
    _patchcore_bundle_statuses,
    _validate_evaluation_config_snapshot,
    inspection_limitations,
    validate_expected_model,
)
from defectops.patchcore_evaluation import load_patchcore_evaluation_config
from defectops.patchcore_experiment import (
    EXPECTED_COMPARISON_STATUS,
    EXPECTED_PREREGISTERED_STATUS,
)


class PatchCoreInspectionTests(unittest.TestCase):
    def test_historical_mask_default_matches_without_mutating_snapshot(self):
        path = Path(__file__).resolve().parents[1] / "configs/patchcore-evaluation-v1.toml"
        current = load_patchcore_evaluation_config(path)
        saved = {k: v for k, v in current.items() if k != "ground_truth_threshold"}
        before = saved.copy()
        _validate_evaluation_config_snapshot(saved, current, "patchcore")
        self.assertEqual(saved, before)
        self.assertNotIn("ground_truth_threshold", saved)
        _validate_evaluation_config_snapshot(current.copy(), current, "patchcore")

    def test_legacy_compatibility_does_not_hide_other_mismatches(self):
        path = Path(__file__).resolve().parents[1] / "configs/patchcore-evaluation-v1.toml"
        current = load_patchcore_evaluation_config(path)
        legacy = {k: v for k, v in current.items() if k != "ground_truth_threshold"}
        cases = [
            ({**legacy, "ground_truth_threshold": 128}, current, "patchcore"),
            ({**legacy, "ground_truth_threshold": None}, current, "patchcore"),
            ({**legacy, "pixel_calibration_quantile": 0.5}, current, "patchcore"),
            ({**legacy, "unexpected": 1}, current, "patchcore"),
            ({k: v for k, v in legacy.items() if k != "purpose"}, current, "patchcore"),
            (legacy, {**current, "ground_truth_threshold": 128}, "patchcore"),
            (legacy, {**current, "comparison_status": "realiad"}, "patchcore"),
            (legacy, current, "padim"),
            (None, current, "patchcore"),
        ]
        for saved, expected, model in cases:
            with self.subTest(saved=saved, model=model):
                with self.assertRaisesRegex(ValueError, "configuration snapshot differs"):
                    _validate_evaluation_config_snapshot(saved, expected, model)

    def test_patchcore_cli_rejects_padim_bundle(self):
        with self.assertRaisesRegex(ValueError, "requires a patchcore fit bundle"):
            validate_expected_model("padim", "patchcore")

    def test_patchcore_cli_accepts_patchcore_bundle(self):
        validate_expected_model("patchcore", "patchcore")

    def test_patchcore_limitations_preserve_exploratory_status(self):
        limitations = inspection_limitations("patchcore")
        self.assertTrue(any("37 of 40" in text for text in limitations))
        self.assertTrue(any("exploratory" in text for text in limitations))
        self.assertTrue(any("not a safety certification" in text for text in limitations))

    def test_preregistered_bundles_are_diagnostics_only(self):
        self.assertEqual(
            _patchcore_bundle_statuses(EXPECTED_COMPARISON_STATUS, False),
            (
                "fit_calibrated_exploratory",
                "evaluated_exploratory",
                "exploratory_rules_frozen_before_test_images_loaded",
            ),
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            _patchcore_bundle_statuses(EXPECTED_PREREGISTERED_STATUS, False)
        self.assertEqual(
            _patchcore_bundle_statuses(EXPECTED_PREREGISTERED_STATUS, True),
            (
                "fit_calibrated_preregistered",
                "evaluated_preregistered",
                "preregistered_rules_frozen_before_test_images_loaded",
            ),
        )


if __name__ == "__main__":
    unittest.main()
