"""Pure checks for the locked VisA external-transfer test; no GPU required."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from defectops.visa_pcb_external_evaluation import (
    _binary_mask, _pixel_sha, _read_mask, audit_final_payloads, checked_preflight,
    load_evaluation_config, paired_macro_escape_difference,
)
from defectops.visa_pcb_external_protocol import canonical


def _image(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), color).save(stream, format="JPEG")
    return stream.getvalue()


def _mask() -> bytes:
    stream = io.BytesIO()
    image = Image.new("L", (12, 10), 0)
    for x in range(2, 5):
        for y in range(3, 7):
            image.putpixel((x, y), 255)
    image.save(stream, format="PNG")
    return stream.getvalue()


class VisaPcbExternalEvaluationTests(unittest.TestCase):
    def test_checked_in_evaluation_lock(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_evaluation_config(root / "configs" / "visa-pcb-external-evaluation-v1.toml")
        self.assertEqual(config["preflight_git_commit"], "1489ccd8e78bcda3fc10bcccda721417d711e482")
        self.assertFalse(config["automatic_pass_allowed"])
        self.assertEqual(config["paired_bootstrap_replicates"], 5000)
        corrected = load_evaluation_config(root / "configs" / "visa-pcb-external-evaluation-v2.toml",
                                           mask_correction_v2=True)
        self.assertEqual(corrected["mask_binarization"], "nonzero_before_nearest")
        self.assertEqual(corrected["baseline_map_top_fraction"], config["baseline_map_top_fraction"])
        self.assertEqual(corrected["challenger_map_top_fraction"], config["challenger_map_top_fraction"])
        self.assertFalse(corrected["automatic_pass_allowed"])
        with self.assertRaisesRegex(ValueError, "selected locked values"):
            load_evaluation_config(root / "configs" / "visa-pcb-external-evaluation-v2.toml")

    def test_original_class_id_one_is_preserved_by_corrected_audit_and_scoring(self) -> None:
        image = Image.new("L", (256, 256), 0)
        image.putpixel((20, 30), 1)
        self.assertEqual(int(_binary_mask(image, 256, 128).sum()), 0)
        self.assertEqual(int(_binary_mask(image, 256, 128, "nonzero_before_nearest").sum()), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tar"
            mask_name = "pcb1/Data/Masks/Anomaly/000.png"
            image_name = "pcb1/Data/Images/Anomaly/000.JPG"
            cal_name = "pcb1/Data/Images/Normal/001.JPG"
            mask_stream = io.BytesIO()
            image.save(mask_stream, format="PNG")
            mask_data = mask_stream.getvalue()

            def jpeg(color: tuple[int, int, int]) -> bytes:
                stream = io.BytesIO()
                Image.new("RGB", (256, 256), color).save(stream, format="JPEG")
                return stream.getvalue()

            cal_data, bad_data = jpeg((20, 30, 40)), jpeg((90, 110, 140))
            with tarfile.open(path, "w") as archive:
                for name, data in ((cal_name, cal_data), (image_name, bad_data), (mask_name, mask_data)):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            with Image.open(io.BytesIO(cal_data)) as cal:
                cal_sha = _pixel_sha(cal)
            calibration = {"records": [{"image": cal_name, "image_sha256": hashlib.sha256(cal_data).hexdigest(),
                                         "decoded_pixel_sha256": cal_sha}]}
            row = {"category": "pcb1", "label": 1, "source_image_member": image_name,
                   "source_mask_member": mask_name}
            config = {"external_final_normal_total": 0, "external_final_anomaly_total": 1,
                      "pixel_ground_truth_threshold": 128, "near_duplicate_dhash_max_distance": 4}
            with self.assertRaisesRegex(ValueError, "Empty anomaly mask"):
                audit_final_payloads(path, [row], calibration, config)
            audit = audit_final_payloads(path, [row], calibration,
                                         {**config, "mask_binarization": "nonzero_before_nearest"})
            self.assertEqual(audit["records"][0]["mask_nonzero_256"], 1)
            with tarfile.open(path, "r:") as archive:
                read = _read_mask(archive, row, (256, 256), 256, 128,
                                  hashlib.sha256(mask_data).hexdigest(), "nonzero_before_nearest")
            self.assertEqual(int(read.sum()), 1)

    def test_preflight_zip_requires_original_calibration_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handoff = Path(directory) / "preflight.zip"
            plan = {"plan_sha256": "plan", "config_sha256": "protocol", "records": [{"category": "pcb1", "source_image_member": "cal.JPG",
                                                          "protocol_split": "calibration"}]}
            audit = {"records": [{"category": "pcb1", "image": "cal.JPG"}],
                     "status": "calibration_jpeg_decoded_no_exact_pixel_duplicates",
                     "external_final_payloads_decoded": False, "exact_pixel_duplicate_count": 0,
                     "count": 1}
            report = {"status": "preflight_passed_external_final_unopened",
                      "git": {"commit": "commit"}, "plan_sha256": "plan", "protocol_config_sha256": "protocol",
                      "registration_handoff_sha256": "registration", "preflight_config_sha256": "pre",
                      "realiad_checkpoint_sha256": "checkpoint", "source_archive_sha256": "archive",
                      "calibration_audit_sha256": hashlib.sha256(canonical(audit)).hexdigest(),
                      "calibration_images_decoded": 224, "external_final_images_decoded": 0,
                      "external_final_scores_computed": 0, "automatic_pass_allowed": False}
            # This synthetic plan intentionally has one calibration image: the hard count must reject it.
            with zipfile.ZipFile(handoff, "w") as archive:
                archive.writestr("preflight.json", json.dumps(report))
                archive.writestr("calibration-audit.json", json.dumps(audit))
            with patch("defectops.visa_pcb_external_evaluation.file_sha256", return_value="zip"):
                with self.assertRaisesRegex(ValueError, "preflight evidence differs"):
                    checked_preflight(handoff, plan,
                                      {"preflight_handoff_sha256": "zip", "preflight_git_commit": "commit",
                                       "realiad_checkpoint_sha256": "checkpoint"},
                                      {"config_sha256": "pre", "source_archive_sha256": "archive",
                                       "registration_handoff_sha256": "registration"})

    def test_test_payload_audit_reads_masks_and_rejects_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "small.tar"
            cal_name = "pcb1/Data/Images/Normal/001.JPG"
            normal_name = "pcb1/Data/Images/Normal/002.JPG"
            bad_name = "pcb1/Data/Images/Anomaly/000.JPG"
            mask_name = "pcb1/Data/Masks/Anomaly/000.png"
            cal = _image((20, 30, 40))
            normal = _image((60, 70, 80))
            bad = _image((90, 110, 140))
            with tarfile.open(archive_path, "w") as archive:
                for name, data in ((cal_name, cal), (normal_name, normal), (bad_name, bad),
                                   (mask_name, _mask())):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            with Image.open(io.BytesIO(cal)) as image:
                cal_pixel_sha = _pixel_sha(image)
            calibration = {"records": [{"image": cal_name, "image_sha256": hashlib.sha256(cal).hexdigest(),
                                         "decoded_pixel_sha256": cal_pixel_sha}]}
            records = [{"category": "pcb1", "label": 0, "source_image_member": normal_name,
                        "source_mask_member": ""},
                       {"category": "pcb1", "label": 1, "source_image_member": bad_name,
                        "source_mask_member": mask_name}]
            config = {"external_final_normal_total": 1, "external_final_anomaly_total": 1,
                      "pixel_ground_truth_threshold": 128, "near_duplicate_dhash_max_distance": 4}
            audit = audit_final_payloads(archive_path, records, calibration, config)
            self.assertEqual((audit["normal_count"], audit["anomaly_count"], audit["mask_count"]), (1, 1, 1))
            self.assertGreater(audit["records"][1]["mask_nonzero_256"], 0)
            with self.assertRaisesRegex(ValueError, "Exact calibration/test image duplicate"):
                audit_final_payloads(archive_path,
                                     [{**records[0], "source_image_member": cal_name}, records[1]],
                                     calibration, config)
            with self.assertRaisesRegex(ValueError, "Missing VisA archive member"):
                audit_final_payloads(archive_path,
                                     [records[0], {**records[1], "source_mask_member": "missing.png"}],
                                     calibration, config)

    def test_paired_macro_difference_is_deterministic(self) -> None:
        rows = []
        for category in ("pcb1", "pcb2", "pcb3", "pcb4"):
            for index in range(100):
                rows.append({"category": category, "label": 1,
                             "baseline_decision": "PASS" if index < 50 else "REVIEW_REQUIRED",
                             "challenger_decision": "PASS" if index < 40 else "REVIEW_REQUIRED"})
        config = {"paired_bootstrap_seed": 20260924, "paired_bootstrap_replicates": 200,
                  "confidence_level": 0.95}
        first = paired_macro_escape_difference(rows, config)
        second = paired_macro_escape_difference(rows, config)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["value"], 0.1)
        self.assertGreater(first["confidence_interval"][0], 0)


if __name__ == "__main__":
    unittest.main()
