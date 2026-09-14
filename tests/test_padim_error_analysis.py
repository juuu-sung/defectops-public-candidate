import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectops.padim_error_analysis import (
    _write_contact_sheet,
    confusion_group,
    describe_values,
    spatial_metrics,
)
from defectops.padim_experiment import checksum


class PadimErrorAnalysisTests(unittest.TestCase):
    def test_confusion_group_uses_fixed_decision(self):
        self.assertEqual(confusion_group(1, "REVIEW_REQUIRED"), "TP")
        self.assertEqual(confusion_group(0, "REVIEW_REQUIRED"), "FP")
        self.assertEqual(confusion_group(0, "PASS"), "TN")
        self.assertEqual(confusion_group(1, "PASS"), "FN")

    def test_describe_values_handles_empty_and_finite_values(self):
        self.assertEqual(describe_values([])["count"], 0)
        summary = describe_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["mean"], 2.0)

    def test_spatial_metrics_are_exact(self):
        anomaly_map = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        ground_truth = np.asarray([[False, True], [False, False]])
        metrics = spatial_metrics(anomaly_map, ground_truth, 1.0)
        self.assertEqual(metrics["predicted_pixels"], 2)
        self.assertEqual(metrics["ground_truth_pixels"], 1)
        self.assertEqual(metrics["intersection_pixels"], 1)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        self.assertAlmostEqual(metrics["dice"], 2 / 3)
        self.assertAlmostEqual(metrics["pixel_precision"], 0.5)
        self.assertAlmostEqual(metrics["pixel_recall"], 1.0)

    def test_contact_sheet_is_created_with_fixed_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.png"
            output = root / "sheet.png"
            Image.new("RGB", (1024, 1024), "white").save(source)
            anomaly_map = np.zeros((1, 256, 256), dtype=np.float32)
            anomaly_map[0, 10:20, 30:40] = 2.0
            ground_truth = np.zeros_like(anomaly_map, dtype=np.bool_)
            ground_truth[0, 12:18, 32:38] = True
            case = {
                "group": "FN",
                "image_path": "sample.png",
                "raw_score": 1.0,
                "score_margin": -0.5,
            }
            _write_contact_sheet(
                [case],
                {"sample.png": 0},
                anomaly_map,
                ground_truth,
                1.0,
                root,
                {"sample.png": {"image_sha256": checksum(source)}},
                output,
            )
            with Image.open(output) as sheet:
                self.assertEqual(sheet.size, (1024, 292))


if __name__ == "__main__":
    unittest.main()
