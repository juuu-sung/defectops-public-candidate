import unittest

from defectops.padim_inspection import (
    _patchcore_bundle_statuses,
    inspection_limitations,
    validate_expected_model,
)
from defectops.patchcore_experiment import (
    EXPECTED_COMPARISON_STATUS,
    EXPECTED_PREREGISTERED_STATUS,
)


class PatchCoreInspectionTests(unittest.TestCase):
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
