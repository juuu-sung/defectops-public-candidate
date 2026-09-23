"""Official-metadata planning must be deterministic and must not open test images."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from defectops.visa_pcb_external_protocol import EXPECTED_COUNTS, build_plan, canonical, load_config
from defectops.visa_pcb_external_registration import CSV_MEMBER, inspect_archive


def synthetic_csv() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["object", "split", "label", "image", "mask"])
    for category, counts in EXPECTED_COUNTS.items():
        for index in range(counts["train_normal"]):
            writer.writerow([category, "train", "normal", f"{category}/Data/Images/Normal/train-{index:04d}.JPG", ""])
        for index in range(counts["test_normal"]):
            writer.writerow([category, "test", "normal", f"{category}/Data/Images/Normal/test-{index:04d}.JPG", ""])
        for index in range(counts["test_anomaly"]):
            writer.writerow([
                category, "test", "anomaly", f"{category}/Data/Images/Anomaly/test-{index:04d}.JPG",
                f"{category}/Data/Masks/Anomaly/test-{index:04d}.png",
            ])
    return stream.getvalue().encode("utf-8")


def synthetic_config(csv_bytes: bytes) -> dict:
    return {
        "source_split_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "categories": list(EXPECTED_COUNTS),
        "calibration_normal_count_per_category": 56,
        "source_dataset": "synthetic_visa",
        "source_url": "https://example.invalid/VisA_20220922.tar",
        "source_archive_bytes": 0,
        "source_repo_commit": "fixed-test-commit",
        "config_sha256": "fixed-test-config",
    }


class VisaPcbExternalProtocolTests(unittest.TestCase):
    def test_repo_config_and_committed_plan_checksum(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "visa-pcb-external-v1.toml")
        plan = json.loads((root / "data" / "manifests" / "visa-pcb-external-plan-v1.json").read_text())
        unsigned = dict(plan)
        self.assertEqual(unsigned.pop("plan_sha256"), hashlib.sha256(canonical(unsigned)).hexdigest())
        self.assertEqual(plan["config_sha256"], config["config_sha256"])
        self.assertEqual(len(plan["records"]), 1026)
        self.assertEqual(sum(row["protocol_split"] == "external_final" for row in plan["records"]), 802)
        self.assertFalse(plan["external_final_images_opened"])

    def test_csv_selection_is_deterministic_and_rejects_changed_source(self) -> None:
        csv_bytes = synthetic_csv()
        config = synthetic_config(csv_bytes)
        first = build_plan(csv_bytes, config)
        second = build_plan(csv_bytes, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["records"]), 1026)
        self.assertEqual(sum(row["protocol_split"] == "calibration" for row in first["records"]), 224)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            build_plan(csv_bytes + b"\n", config)

    def test_registration_reads_only_csv_payload_and_checks_selected_members(self) -> None:
        csv_bytes = synthetic_csv()
        config = synthetic_config(csv_bytes)
        plan = build_plan(csv_bytes, config)
        selected = {row["source_image_member"] for row in plan["records"]}
        selected.update(row["source_mask_member"] for row in plan["records"] if row["source_mask_member"])
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "VisA_20220922.tar"
            with tarfile.open(archive_path, "w") as archive:
                for name in sorted(selected | {CSV_MEMBER}):
                    content = csv_bytes if name == CSV_MEMBER else b"fake image payload, never decoded"
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            config["source_archive_bytes"] = archive_path.stat().st_size
            plan = build_plan(csv_bytes, config)
            report = inspect_archive(archive_path, plan, config)
            self.assertEqual(report["selected_member_count"], len(selected) + 1)
            self.assertFalse(report["test_member_payloads_decoded"])
            self.assertFalse(report["test_scores_computed"])
            self.assertFalse(report["automatic_pass_allowed"])


if __name__ == "__main__":
    unittest.main()
