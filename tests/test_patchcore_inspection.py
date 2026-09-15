import unittest

from defectops.padim_inspection import (
    inspection_limitations,
    validate_expected_model,
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


if __name__ == "__main__":
    unittest.main()
