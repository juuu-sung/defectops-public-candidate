"""Synthetic contract fixtures only; these are not model-performance observations."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from defectops.review_store import ReviewStore


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ReviewStore(self.root / "review" / "reviews.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def make_bundle(self, inspection_id="inspection-1", parent=None):
        directory = (parent or self.root) / inspection_id
        directory.mkdir(parents=True)
        payloads = {
            "input": b"synthetic input png bytes",
            "heatmap": b"synthetic heatmap png bytes",
            "overlay": b"synthetic overlay png bytes",
            "anomaly_map": b"synthetic map npz bytes",
        }
        filenames = {
            "input": "input.png",
            "heatmap": "heatmap.png",
            "overlay": "overlay.png",
            "anomaly_map": "anomaly-map.npz",
        }
        artifacts = {}
        for key, name in filenames.items():
            (directory / name).write_bytes(payloads[key])
            artifacts[key] = name
            artifacts[f"{key}_sha256"] = _sha(payloads[key])
        record = {
            "schema_version": 1,
            "inspection_id": inspection_id,
            "status": "completed",
            "category": "transistor",
            "model": "patchcore",
            "comparison_status": "exploratory_after_padim_test_observed",
            "decision": "REVIEW_REQUIRED",
            "decision_rule": "score >= threshold",
            "raw_score": 52.7,
            "image_threshold": 40.8,
            "automatic_pass": False,
            "automatic_pass_allowed": False,
            "input": {"sha256": artifacts["input_sha256"]},
            "trace": {
                "fit_run_id": "patchcore-transistor-seed42",
                "checkpoint_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            },
            "artifacts": artifacts,
        }
        (directory / "inspection.json").write_text(json.dumps(record), encoding="utf-8")
        return directory, record

    def test_register_verify_and_append_only_reviews(self):
        bundle, _ = self.make_bundle()
        self.assertEqual(self.store.register(bundle)["status"], "registered")
        self.assertEqual(self.store.register(bundle)["status"], "already_registered")
        self.assertEqual(self.store.verify("inspection-1")["status"], "verified")
        first = self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator-1",
            human_verdict="ANOMALY_VISIBLE",
            note="Left lead is bent in the image.",
        )
        self.assertEqual(first["revision"], 1)
        self.assertFalse(first["automatic_pass_allowed"])
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.store.add_review(
                "inspection-1",
                expected_revision=0,
                reviewer="operator-2",
                human_verdict="UNCERTAIN",
                note="Second look needed.",
            )
        second = self.store.add_review(
            "inspection-1",
            expected_revision=1,
            reviewer="operator-2",
            human_verdict="UNCERTAIN",
            note="Needs closer visual inspection.",
        )
        self.assertEqual(second["revision"], 2)
        item = self.store.get_inspection("inspection-1")
        self.assertEqual(item["model_decision"], "REVIEW_REQUIRED")
        self.assertEqual(item["review_revision"], 2)
        self.assertEqual(item["latest_review"]["human_verdict"], "UNCERTAIN")
        self.assertEqual(len(self.store.review_history("inspection-1")), 2)
        self.assertEqual(self.store.list_inspections()[0]["human_verdict"], "UNCERTAIN")
        with sqlite3.connect(self.store.database_path) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only review"):
                connection.execute("UPDATE review_events SET note = 'changed' WHERE revision = 1")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable inspection"):
                connection.execute("DELETE FROM inspections WHERE inspection_id = 'inspection-1'")

    def test_checksum_tampering_blocks_import_verify_and_review(self):
        bundle, _ = self.make_bundle()
        (bundle / "overlay.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "overlay artifact checksum differs"):
            self.store.register(bundle)
        (bundle / "overlay.png").write_bytes(b"synthetic overlay png bytes")
        self.store.register(bundle)
        (bundle / "overlay.png").write_bytes(b"changed again")
        with self.assertRaisesRegex(ValueError, "overlay artifact checksum differs"):
            self.store.verify("inspection-1")
        with self.assertRaisesRegex(ValueError, "overlay artifact checksum differs"):
            self.store.add_review(
                "inspection-1",
                expected_revision=0,
                reviewer="operator",
                human_verdict="UNCERTAIN",
                note="Needs inspection.",
            )
        self.assertEqual(self.store.review_history("inspection-1"), [])

    def test_changed_json_and_duplicate_id_are_rejected(self):
        bundle, record = self.make_bundle()
        self.store.register(bundle)
        record["raw_score"] = 60.0
        (bundle / "inspection.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after import"):
            self.store.verify("inspection-1")
        other_parent = self.root / "other"
        other_parent.mkdir()
        other_bundle, _ = self.make_bundle(parent=other_parent)
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.store.register(other_bundle)

    def test_fail_closed_contract_and_failed_inspection(self):
        bundle, record = self.make_bundle()
        record["automatic_pass_allowed"] = True
        (bundle / "inspection.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbid automatic PASS"):
            self.store.register(bundle)
        record["automatic_pass_allowed"] = False
        record["decision"] = "PASS"
        (bundle / "inspection.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen score rule"):
            self.store.register(bundle)

        failed = self.root / "failed-1"
        failed.mkdir()
        (failed / "inspection.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "inspection_id": "failed-1",
                    "status": "INSPECTION_FAILED",
                    "automatic_pass": False,
                    "automatic_pass_allowed": False,
                    "error_type": "InputValidationError",
                }
            ),
            encoding="utf-8",
        )
        self.store.register(failed)
        self.assertEqual(self.store.get_inspection("failed-1")["status"], "INSPECTION_FAILED")
        with self.assertRaisesRegex(ValueError, "Failed inspections cannot"):
            self.store.add_review(
                "failed-1",
                expected_revision=0,
                reviewer="operator",
                human_verdict="NO_ANOMALY_VISIBLE",
                note="Cannot evaluate an invalid input.",
            )

    def test_symlink_artifact_is_rejected(self):
        bundle, _ = self.make_bundle()
        (bundle / "overlay.png").unlink()
        (bundle / "overlay.png").symlink_to(bundle / "input.png")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self.store.register(bundle)


if __name__ == "__main__":
    unittest.main()
