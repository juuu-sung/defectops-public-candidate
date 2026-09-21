import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectops.padim_inspection import (
    InputValidationError,
    _load_validated_png,
    colorize_anomaly_values,
    scale_bbox_to_original,
)


class PadimInspectionTests(unittest.TestCase):
    def test_color_map_is_deterministic_and_bounded(self):
        values = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        colors = colorize_anomaly_values(values)
        self.assertEqual(colors.shape, (1, 3, 3))
        self.assertEqual(colors.dtype, np.uint8)
        np.testing.assert_array_equal(colors, colorize_anomaly_values(values))

    def test_bbox_scaling_preserves_exclusive_endpoints(self):
        self.assertEqual(
            scale_bbox_to_original([0, 0, 256, 256], (256, 256), (1024, 1024)),
            [0, 0, 1024, 1024],
        )
        self.assertEqual(
            scale_bbox_to_original([1, 1, 2, 2], (256, 256), (1024, 1024)),
            [4, 4, 8, 8],
        )

    def test_input_contract_accepts_only_1024_rgb_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.png"
            Image.new("RGB", (1024, 1024), (1, 2, 3)).save(valid)
            array, size = _load_validated_png(valid)
            self.assertEqual(array.shape, (1024, 1024, 3))
            self.assertEqual(size, (1024, 1024))

            invalid = root / "invalid.png"
            Image.new("L", (32, 32), 0).save(invalid)
            with self.assertRaises(InputValidationError):
                _load_validated_png(invalid)

            other_size = root / "other-size.png"
            Image.new("RGB", (700, 700), (4, 5, 6)).save(other_size)
            array, size = _load_validated_png(other_size, expected_size=None)
            self.assertEqual(array.shape, (700, 700, 3))
            self.assertEqual(size, (700, 700))


if __name__ == "__main__":
    unittest.main()
