"""No final payload is needed to audit VisA calibration images."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from defectops.visa_pcb_external_preflight import (
    audit_calibration,
    checked_registration,
    load_preflight_config,
)


def jpeg(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), color).save(stream, format="JPEG")
    return stream.getvalue()


class VisaPcbExternalPreflightTests(unittest.TestCase):
    def test_checked_in_lock(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_preflight_config(root / "configs" / "visa-pcb-external-preflight-v1.toml")
        self.assertFalse(config["external_final_payload_decode_allowed"])
        self.assertEqual(config["calibration_image_count"], 224)

    def test_registration_zip_must_match_plan_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handoff = Path(directory) / "registration.zip"
            plan = {"plan_sha256": "plan"}
            report = {
                "archive_sha256": "archive", "status": "external_archive_registered_metadata_only",
                "test_member_payloads_decoded": False, "test_scores_computed": False,
            }
            with zipfile.ZipFile(handoff, "w") as archive:
                archive.writestr("plan.json", '{"plan_sha256":"plan"}')
                archive.writestr("registration.json", json.dumps(report))
            contract = {"registration_handoff_sha256": "handoff", "source_archive_sha256": "archive"}
            with patch("defectops.visa_pcb_external_preflight.file_sha256", return_value="handoff"), patch(
                "defectops.visa_pcb_external_preflight.inspect_archive", return_value=report
            ):
                self.assertEqual(checked_registration(handoff, Path("archive.tar"), plan, {}, contract), report)
                with self.assertRaisesRegex(ValueError, "plan differs"):
                    checked_registration(handoff, Path("archive.tar"), {"plan_sha256": "changed"}, {}, contract)

    def test_calibration_only_decodes_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "tiny.tar"
            names = ["pcb1/Data/Images/Normal/001.JPG", "pcb1/Data/Images/Normal/002.JPG"]
            with tarfile.open(archive_path, "w") as archive:
                for name, content in (
                    (names[0], jpeg((20, 30, 40))),
                    (names[1], jpeg((50, 60, 70))),
                    ("pcb1/Data/Images/Anomaly/000.JPG", b"not-a-jpeg"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            rows = [
                {"category": "pcb1", "protocol_split": "calibration", "label": 0,
                 "source_image_member": name, "source_mask_member": ""}
                for name in names
            ]
            rows.append({"category": "pcb1", "protocol_split": "external_final", "label": 1,
                         "source_image_member": "pcb1/Data/Images/Anomaly/000.JPG",
                         "source_mask_member": "pcb1/Data/Masks/Anomaly/000.png"})
            result = audit_calibration(archive_path, rows, expected_count=2)
            self.assertEqual(result["count"], 2)
            self.assertFalse(result["external_final_payloads_decoded"])
            self.assertEqual(result["image_sizes"], {"12x10": 2})
            with tarfile.open(archive_path, "w") as archive:
                for name in names:
                    content = jpeg((20, 30, 40))
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            with self.assertRaisesRegex(ValueError, "Exact calibration pixel duplicate"):
                audit_calibration(archive_path, rows, expected_count=2)


if __name__ == "__main__":
    unittest.main()
