"""Test inventory behavior with byte fixtures, not model/image decoding accuracy."""

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defectops.manifest import build_manifest, verify_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        for i in range(10):
            self.put(f"transistor/train/good/{i:03}.png", f"normal-{i}".encode())
        self.put("transistor/test/good/000.png", b"test-good")
        self.put("transistor/test/damaged_case/000.png", b"test-defect")
        self.put("transistor/ground_truth/damaged_case/000_mask.png", b"mask")

    def put(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def build(self):
        return build_manifest(self.root, ["transistor"])

    def test_portable_and_disjoint(self):
        manifest = self.build()
        other = Path(self.temp.name) / "other-machine"
        shutil.copytree(self.root, other)
        self.assertEqual(manifest, build_manifest(other, ["transistor"]))
        self.assertEqual(sum(r["split"] == "train" for r in manifest["records"]), 8)
        self.assertEqual(sum(r["split"] == "calibration" for r in manifest["records"]), 2)
        self.assertEqual(sum(r["split"] == "test" for r in manifest["records"]), 2)
        for row in manifest["records"]:
            self.assertFalse(Path(row["image_path"]).is_absolute())
            if row["split"] != "test":
                self.assertEqual(row["label"], 0)
                self.assertEqual(row["official_split"], "train")
        verify_manifest(other, manifest)

    def test_changed_image_rejected(self):
        manifest = self.build()
        self.put("transistor/train/good/000.png", b"modified")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_changed_mask_rejected(self):
        manifest = self.build()
        self.put("transistor/ground_truth/damaged_case/000_mask.png", b"new-mask")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_added_image_rejected(self):
        manifest = self.build()
        self.put("transistor/train/good/999.png", b"additional")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_missing_mask_rejected(self):
        (self.root / "transistor/ground_truth/damaged_case/000_mask.png").unlink()
        with self.assertRaisesRegex(ValueError, "mask"):
            self.build()

    def test_duplicate_across_train_test_rejected(self):
        self.put("transistor/test/good/000.png", b"normal-0")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.build()

    def test_split_tampering_rejected(self):
        manifest = copy.deepcopy(self.build())
        manifest["records"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_empty_training_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "no training"):
            build_manifest(self.root, ["transistor"], calibration_fraction=0.99)

    def test_train_anomaly_rejected(self):
        self.put("transistor/train/defect/000.png", b"unexpected-train-defect")
        with self.assertRaisesRegex(ValueError, "only good"):
            self.build()

    def test_path_escape_rejected(self):
        outside = Path(self.temp.name) / "outside.png"
        outside.write_bytes(b"outside-data")
        try:
            (self.root / "transistor/train/good/999.png").symlink_to(outside)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows requires Developer Mode or symlink privileges for this fixture")
            raise
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()
