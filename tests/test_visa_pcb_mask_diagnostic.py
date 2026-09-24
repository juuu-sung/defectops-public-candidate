"""The diagnostic must distinguish raw class IDs from a truly empty mask."""

from __future__ import annotations

import unittest

from PIL import Image

from defectops.visa_pcb_mask_diagnostic import inspect_mask


class VisaPcbMaskDiagnosticTests(unittest.TestCase):
    def test_class_id_one_is_not_an_empty_defect(self) -> None:
        image = Image.new("L", (256, 256), 0)
        image.putpixel((20, 30), 1)
        result = inspect_mask(image)
        self.assertEqual(result["original_nonzero_pixels"], 1)
        self.assertEqual(result["current_parser_resized_pixels_at_least_128"], 0)
        self.assertEqual(result["nonzero_first_resized_pixels_at_least_128"], 1)

    def test_empty_mask_stays_empty(self) -> None:
        result = inspect_mask(Image.new("L", (256, 256), 0))
        self.assertEqual(result["original_nonzero_pixels"], 0)
        self.assertEqual(result["nonzero_first_resized_pixels_at_least_128"], 0)


if __name__ == "__main__":
    unittest.main()
