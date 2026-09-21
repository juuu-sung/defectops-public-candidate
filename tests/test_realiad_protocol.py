"""Tests for the registered Real-IAD single-view manifest contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from defectops.manifest import verify_manifest
from defectops.realiad_protocol import build_realiad_manifest


class RealIADProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.object_name = "transistor1"
        annotation = {
            "meta": {
                "prefix": "transistor1/",
                "normal_class": "OK",
                "pre_transform": False,
            },
            "train": [],
            "test": [],
        }
        for index in range(10):
            relative = f"OK/S{index:04d}/transistor1_{index:04d}_OK_C1_train.jpg"
            annotation["train"].append(self.entry("OK", relative, None))
            self.put_image(relative, f"train-{index}".encode())
        good = "OK/S1000/transistor1_1000_OK_C1_test.jpg"
        annotation["test"].append(self.entry("OK", good, None))
        self.put_image(good, b"test-good")
        anomaly = "NG/BX/S2000/transistor1_2000_NG_BX_C1_test.jpg"
        mask = "NG/BX/S2000/transistor1_2000_NG_BX_C1_test.png"
        annotation["test"].append(self.entry("BX", anomaly, mask))
        self.put_image(anomaly, b"test-anomaly")
        self.put_image(mask, b"mask")
        annotation_path = (
            self.root
            / "real_iad"
            / "annotations"
            / "realiad_jsons_sv"
            / "transistor1.json"
        )
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    def entry(self, anomaly_class, image_path, mask_path):
        return {
            "category": self.object_name,
            "anomaly_class": anomaly_class,
            "image_path": image_path,
            "mask_path": mask_path,
        }

    def put_image(self, relative, content):
        path = self.root / "real_iad" / "images" / self.object_name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_builds_disjoint_portable_manifest(self):
        manifest = build_realiad_manifest(self.root, "realiad_transistor1")
        self.assertEqual(manifest["dataset"], "real_iad")
        self.assertEqual(manifest["annotation_variant"], "realiad_jsons_sv")
        self.assertEqual(sum(row["split"] == "train" for row in manifest["records"]), 8)
        self.assertEqual(
            sum(row["split"] == "calibration" for row in manifest["records"]), 2
        )
        self.assertEqual(sum(row["split"] == "test" for row in manifest["records"]), 2)
        self.assertTrue(all(row["view"] == "C1" for row in manifest["records"]))
        verify_manifest(self.root, manifest)

    def test_changed_file_is_rejected(self):
        manifest = build_realiad_manifest(self.root, "realiad_transistor1")
        first = self.root / manifest["records"][0]["image_path"]
        first.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_split_tampering_is_rejected(self):
        manifest = copy.deepcopy(build_realiad_manifest(self.root, "realiad_transistor1"))
        row = next(record for record in manifest["records"] if record["split"] == "train")
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(self.root, manifest)

    def test_non_c1_path_is_rejected(self):
        annotation_path = (
            self.root
            / "real_iad"
            / "annotations"
            / "realiad_jsons_sv"
            / "transistor1.json"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        old = annotation["train"][0]["image_path"]
        new = old.replace("_C1_", "_C2_")
        annotation["train"][0]["image_path"] = new
        self.put_image(new, b"camera-two")
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "camera C1"):
            build_realiad_manifest(self.root, "realiad_transistor1")


if __name__ == "__main__":
    unittest.main()
