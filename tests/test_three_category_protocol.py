import tempfile
import unittest
from pathlib import Path

from defectops.three_category_protocol import (
    EXPECTED_EVIDENCE_STATUS,
    EXPECTED_REGISTRATION_STATUS,
    load_three_category_protocol,
)


class ThreeCategoryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.path = self.root / "configs/three-category-comparison-v1.toml"

    def test_registered_scope_and_rules_load(self):
        config = load_three_category_protocol(self.path)
        self.assertEqual(config["registration_status"], EXPECTED_REGISTRATION_STATUS)
        self.assertEqual(config["evidence_status"], EXPECTED_EVIDENCE_STATUS)
        self.assertEqual(
            config["categories"], ["transistor", "cable", "metal_nut"]
        )
        self.assertEqual(config["model_seeds"], [42, 43, 44])
        self.assertFalse(config["selection"]["automatic_pass_allowed"])

    def test_test_time_tuning_or_scope_change_is_rejected(self):
        original = self.path.read_text(encoding="utf-8")
        for changed in (
            original.replace('model_seeds = [42, 43, 44]', 'model_seeds = [42]'),
            original.replace('image_quantile = 0.95', 'image_quantile = 0.99'),
            original.replace(
                'automatic_pass_allowed = false', 'automatic_pass_allowed = true'
            ),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "changed.toml"
                path.write_text(changed, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    load_three_category_protocol(path)


if __name__ == "__main__":
    unittest.main()
