"""Real PNG fixtures exercise corruption, mask validity and test-statistic exclusion."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image, PngImagePlugin
from defectops.data_audit import audit_dataset
from defectops.manifest import build_manifest


class DataAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i in range(5):
            self.save(f"transistor/train/good/{i:03}.png", "RGB", (12, 12), (i * 20, 0, 0))
        self.save("transistor/test/good/000.png", "RGB", (12, 12), (0, 100, 0))
        self.save("transistor/test/bent_lead/000.png", "RGB", (12, 12), (0, 0, 100))
        self.mask = "transistor/ground_truth/bent_lead/000_mask.png"
        self.save(self.mask, "L", (12, 12), 255)

    def save(self, rel, mode, size, color):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, color).save(path)

    def audit(self):
        return audit_dataset(self.root, build_manifest(self.root, ["transistor"]))

    def test_valid_data_and_no_test_quality_statistics(self):
        report = self.audit()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["images_decoded"], 7)
        self.assertEqual(report["masks_decoded"], 1)
        self.assertEqual(set(report["quality_summary"]), {"transistor/train", "transistor/calibration"})
        for rows in report["normal_quality_rows"].values():
            self.assertTrue(all("/train/good/" in row["image_path"] for row in rows))

    def test_corrupt_png(self):
        (self.root / "transistor/train/good/000.png").write_bytes(b"not-a-real-image")
        report = self.audit()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(e["path"].endswith("train/good/000.png") for e in report["errors"]))

    def test_mismatched_mask(self):
        self.save(self.mask, "L", (11, 12), 255)
        self.assertTrue(any("size mismatch" in e["error"] for e in self.audit()["errors"]))

    def test_nonbinary_mask(self):
        self.save(self.mask, "L", (12, 12), 127)
        self.assertTrue(any("binary" in e["error"] for e in self.audit()["errors"]))

    def test_empty_mask(self):
        self.save(self.mask, "L", (12, 12), 0)
        self.assertTrue(any("empty mask" in e["error"] for e in self.audit()["errors"]))

    def test_pixels_duplicated_with_different_png_metadata(self):
        path = self.root / "transistor/test/good/000.png"
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("fixture", "different byte encoding")
        Image.new("RGB", (12, 12), (0, 0, 0)).save(path, pnginfo=metadata)
        report = self.audit()
        self.assertEqual(len(report["decoded_pixel_duplicates"]), 1)
        self.assertEqual(report["status"], "failed")

    def test_unlisted_image(self):
        self.save("transistor/unexpected.png", "RGB", (12, 12), (255, 0, 0))
        self.assertTrue(any("not in manifest" in e["error"] for e in self.audit()["errors"]))


if __name__ == "__main__":
    unittest.main()
