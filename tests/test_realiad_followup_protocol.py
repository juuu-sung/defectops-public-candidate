import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from defectops.realiad_followup_protocol import (
    build_realiad_followup_plan,
    load_realiad_followup_protocol,
    summarize_realiad_followup_plan,
)


class RealIADFollowupProtocolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.config = self.root / "configs/realiad-followup-preregistration-v1.toml"

    def test_frozen_candidate_and_gate_contract_loads(self):
        config = load_realiad_followup_protocol(self.config)
        self.assertEqual(config["categories"], ["realiad_pcb", "realiad_terminalblock"])
        self.assertEqual(len(config["candidates"]), 4)
        self.assertFalse(config["selection"]["automatic_pass_allowed"])
        self.assertTrue(config["final_gate"]["run_once"])

    def test_scope_threshold_or_final_gate_change_is_rejected(self):
        original = self.config.read_text(encoding="utf-8")
        changed_versions = (
            original.replace("realiad_pcb", "realiad_switch", 1),
            original.replace("image_quantile = 0.95", "image_quantile = 0.99"),
            original.replace("post_final_tuning_allowed = false", "post_final_tuning_allowed = true"),
        )
        for changed in changed_versions:
            with self.subTest(changed=changed[-80:]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "changed.toml"
                path.write_text(changed, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                    load_realiad_followup_protocol(path)

    def test_plan_is_deterministic_stratified_and_group_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "realiad_jsons.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for source in ("pcb", "terminalblock"):
                    archive.writestr(
                        f"realiad_jsons_sv/{source}.json",
                        json.dumps(self.annotation(source)),
                    )
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            with patch.dict(
                "defectops.realiad_followup_protocol.REAL_IAD_SOURCE_SHA256",
                {"realiad_jsons.zip": archive_sha},
            ):
                first = build_realiad_followup_plan(archive_path, self.config)
                second = build_realiad_followup_plan(archive_path, self.config)
            self.assertEqual(first, second)
            summary = summarize_realiad_followup_plan(first)
            self.assertEqual(summary["plan_sha256"], first["plan_sha256"])
            by_group = {}
            for row in first["records"]:
                by_group.setdefault((row["category"], row["sample_group"]), set()).add(
                    row["planned_split"]
                )
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))
            for category in ("realiad_pcb", "realiad_terminalblock"):
                for defect in ("OK", "D1", "D2"):
                    splits = {
                        row["planned_split"]
                        for row in first["records"]
                        if row["category"] == category and row["defect_type"] == defect
                    }
                    if defect == "OK":
                        self.assertTrue({"train", "calibration", "development", "final_holdout"} >= splits)
                    else:
                        self.assertEqual(splits, {"development", "final_holdout"})

    def test_config_digest_is_independent_of_lf_or_crlf_checkout(self):
        original = self.config.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            lf = directory / "lf.toml"
            crlf = directory / "crlf.toml"
            lf.write_bytes(original.replace("\r\n", "\n").encode())
            crlf.write_bytes(original.replace("\r\n", "\n").replace("\n", "\r\n").encode())
            archive_path = directory / "realiad_jsons.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for source in ("pcb", "terminalblock"):
                    archive.writestr(
                        f"realiad_jsons_sv/{source}.json",
                        json.dumps(self.annotation(source)),
                    )
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            with patch.dict(
                "defectops.realiad_followup_protocol.REAL_IAD_SOURCE_SHA256",
                {"realiad_jsons.zip": archive_sha},
            ):
                lf_plan = build_realiad_followup_plan(archive_path, lf)
                crlf_plan = build_realiad_followup_plan(archive_path, crlf)
            self.assertEqual(lf_plan, crlf_plan)

    @staticmethod
    def annotation(source):
        train = []
        test = []
        for group in range(1, 11):
            for image in range(2):
                path = f"OK/S{group:04d}/{source}_{group:04d}_{image}_OK_C1_train.jpg"
                train.append(RealIADFollowupProtocolTests.item(source, "OK", path))
        for defect in ("OK", "D1", "D2"):
            for group in range(20, 24):
                for image in range(2):
                    path = (
                        f"{'OK' if defect == 'OK' else 'NG/' + defect}/S{group:04d}/"
                        f"{source}_{group:04d}_{image}_{defect}_C1_test.jpg"
                    )
                    mask = None if defect == "OK" else path.removesuffix(".jpg") + ".png"
                    test.append(RealIADFollowupProtocolTests.item(source, defect, path, mask))
        return {
            "meta": {"prefix": f"{source}/", "normal_class": "OK", "pre_transform": False},
            "train": train,
            "test": test,
        }

    @staticmethod
    def item(source, defect, image_path, mask_path=None):
        return {
            "category": source,
            "anomaly_class": defect,
            "image_path": image_path,
            "mask_path": mask_path,
        }


if __name__ == "__main__":
    unittest.main()
