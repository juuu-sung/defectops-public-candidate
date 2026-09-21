import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectops.data_audit import audit_dataset
from defectops.realiad_protocol import build_realiad_manifest


class RealIADAuditTests(unittest.TestCase):
    def test_jpeg_and_resized_grayscale_masks_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = {
                "meta": {
                    "prefix": "transistor1/",
                    "normal_class": "OK",
                    "pre_transform": False,
                },
                "train": [],
                "test": [],
            }

            def entry(anomaly_class, image_path, mask_path=None):
                return {
                    "category": "transistor1",
                    "anomaly_class": anomaly_class,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }

            def save(relative, image, format_name):
                path = root / "real_iad" / "images" / "transistor1" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format=format_name)

            for index in range(5):
                relative = f"OK/S{index:04d}/transistor1_{index:04d}_OK_C1_train.jpg"
                annotation["train"].append(entry("OK", relative))
                save(relative, Image.new("RGB", (12, 12), (index * 20, 1, 2)), "JPEG")
            good = "OK/S1000/transistor1_1000_OK_C1_test.jpg"
            annotation["test"].append(entry("OK", good))
            save(good, Image.new("RGB", (12, 12), (1, 90, 2)), "JPEG")

            for index, empty in ((2000, False), (2001, True)):
                image_path = f"NG/BX/S{index}/transistor1_{index}_NG_BX_C1_test.jpg"
                mask_path = f"NG/BX/S{index}/transistor1_{index}_NG_BX_C1_test.png"
                annotation["test"].append(entry("BX", image_path, mask_path))
                color = (210, 10, 10) if empty else (10, 10, 210)
                save(image_path, Image.new("RGB", (12, 12), color), "JPEG")
                pixels = np.zeros((12, 12), dtype=np.uint8)
                if not empty:
                    pixels[4:8, 4:8] = 255
                    pixels[3, 3] = 127
                    pixels[3, 4] = 128
                save(mask_path, Image.fromarray(pixels), "PNG")

            annotation_path = (
                root
                / "real_iad"
                / "annotations"
                / "realiad_jsons_sv"
                / "transistor1.json"
            )
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            report = audit_dataset(
                root, build_realiad_manifest(root, "realiad_transistor1")
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["images_decoded"], 8)
            self.assertEqual(report["masks_decoded"], 2)
            self.assertEqual(report["errors"], [])
            self.assertEqual(len(report["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
