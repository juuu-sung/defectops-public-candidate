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

    def mock_sop_result(self, defect_type="bent_lead", document_id="DEMO-TR-BENT"):
        return {
            "status": "found_simulation_only",
            "product": "transistor",
            "defect_type": defect_type,
            "as_of": "2026-09-21",
            "document_id": document_id,
            "revision": 2,
            "catalog_sha256": "c" * 64,
            "sources": [
                {
                    "citation_id": f"{document_id}@r2#OBSERVE",
                    "text": "Synthetic source snapshot.",
                }
            ],
        }

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

    def test_sop_evidence_is_tied_to_review_revision_and_append_only(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator-1",
            human_verdict="ANOMALY_VISIBLE",
            note="Left metal leg is not seated in the hole.",
        )
        result = self.mock_sop_result()
        event = self.store.add_sop_evidence(
            "inspection-1",
            expected_review_revision=1,
            selected_by="operator-1",
            defect_type="bent_lead",
            result=result,
        )
        self.assertEqual(event["status"], "recorded")
        self.assertEqual(event["review_revision"], 1)
        history = self.store.sop_evidence_history("inspection-1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["sources"], result["sources"])
        repeated = self.store.add_sop_evidence(
            "inspection-1",
            expected_review_revision=1,
            selected_by="operator-1",
            defect_type="bent_lead",
            result=result,
        )
        self.assertEqual(repeated["status"], "already_recorded")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.add_sop_evidence(
                "inspection-1",
                expected_review_revision=1,
                selected_by="operator-1",
                defect_type="cut_lead",
                result=self.mock_sop_result("cut_lead", "DEMO-TR-CUT"),
            )
        with sqlite3.connect(self.store.database_path) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only SOP evidence"):
                connection.execute("UPDATE sop_evidence_events SET selected_by = 'changed'")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only SOP evidence"):
                connection.execute("DELETE FROM sop_evidence_events")

    def test_sop_evidence_requires_current_visible_anomaly_review(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.store.add_sop_evidence(
                "inspection-1",
                expected_review_revision=1,
                selected_by="operator",
                defect_type="bent_lead",
                result=self.mock_sop_result(),
            )
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator",
            human_verdict="UNCERTAIN",
            note="Image is inconclusive.",
        )
        with self.assertRaisesRegex(ValueError, "ANOMALY_VISIBLE"):
            self.store.add_sop_evidence(
                "inspection-1",
                expected_review_revision=1,
                selected_by="operator",
                defect_type="bent_lead",
                result=self.mock_sop_result(),
            )

    def test_local_mock_ticket_requires_evidence_draft_and_explicit_approval(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator-1",
            human_verdict="ANOMALY_VISIBLE",
            note="Left metal leg is not seated in the hole.",
        )
        with self.assertRaisesRegex(ValueError, "SOP evidence"):
            self.store.create_action_draft(
                "inspection-1", expected_review_revision=1, created_by="operator-1"
            )
        self.store.add_sop_evidence(
            "inspection-1",
            expected_review_revision=1,
            selected_by="operator-1",
            defect_type="bent_lead",
            result=self.mock_sop_result(),
        )
        draft = self.store.create_action_draft(
            "inspection-1", expected_review_revision=1, created_by="operator-1"
        )
        self.assertEqual(draft["status"], "recorded")
        self.assertIn("제품 합격·폐기·재작업·설비 변경을 수행하지 않는다", draft["action_text"])
        repeated_draft = self.store.create_action_draft(
            "inspection-1", expected_review_revision=1, created_by="operator-1"
        )
        self.assertEqual(repeated_draft["status"], "already_recorded")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.create_action_draft(
                "inspection-1", expected_review_revision=1, created_by="another-operator"
            )
        with self.assertRaisesRegex(ValueError, "APPROVED_FOR_MOCK_TICKET"):
            self.store.issue_mock_ticket(
                "inspection-1",
                expected_review_revision=1,
                approval_event_id=999,
                issued_by="operator-1",
            )
        approval = self.store.decide_action_draft(
            "inspection-1",
            expected_review_revision=1,
            action_draft_event_id=draft["event_id"],
            approver="operator-1",
            decision="APPROVED_FOR_MOCK_TICKET",
            note="Local workflow test only.",
        )
        self.assertEqual(approval["status"], "recorded")
        ticket = self.store.issue_mock_ticket(
            "inspection-1",
            expected_review_revision=1,
            approval_event_id=approval["event_id"],
            issued_by="operator-1",
        )
        self.assertEqual(ticket["status"], "recorded")
        self.assertTrue(ticket["ticket_id"].startswith("MOCK-"))
        self.assertTrue(ticket["payload"]["simulation_only"])
        self.assertFalse(ticket["payload"]["external_delivery"])
        self.assertEqual(ticket["payload"]["review"]["revision"], 1)
        self.assertEqual(ticket["payload"]["sop_evidence"]["document_id"], "DEMO-TR-BENT")
        self.assertEqual(len(ticket["payload_sha256"]), 64)
        repeated_ticket = self.store.issue_mock_ticket(
            "inspection-1",
            expected_review_revision=1,
            approval_event_id=approval["event_id"],
            issued_by="operator-1",
        )
        self.assertEqual(repeated_ticket["status"], "already_recorded")
        self.assertEqual(len(self.store.action_draft_history("inspection-1")), 1)
        self.assertEqual(len(self.store.action_approval_history("inspection-1")), 1)
        self.assertEqual(len(self.store.mock_ticket_history("inspection-1")), 1)
        with sqlite3.connect(self.store.database_path) as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only action draft"):
                connection.execute("UPDATE action_draft_events SET created_by = 'changed'")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only action approval"):
                connection.execute("DELETE FROM action_approval_events")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only mock ticket"):
                connection.execute("UPDATE mock_ticket_events SET issued_by = 'changed'")

    def test_rejected_action_cannot_create_mock_ticket(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator",
            human_verdict="ANOMALY_VISIBLE",
            note="Visible anomaly.",
        )
        self.store.add_sop_evidence(
            "inspection-1",
            expected_review_revision=1,
            selected_by="operator",
            defect_type="bent_lead",
            result=self.mock_sop_result(),
        )
        draft = self.store.create_action_draft(
            "inspection-1", expected_review_revision=1, created_by="operator"
        )
        rejection = self.store.decide_action_draft(
            "inspection-1",
            expected_review_revision=1,
            action_draft_event_id=draft["event_id"],
            approver="operator",
            decision="REJECTED",
            note="Needs a new human review.",
        )
        with self.assertRaisesRegex(ValueError, "APPROVED_FOR_MOCK_TICKET"):
            self.store.issue_mock_ticket(
                "inspection-1",
                expected_review_revision=1,
                approval_event_id=rejection["event_id"],
                issued_by="operator",
            )

    def test_schema_one_database_migrates_without_losing_reviews(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator",
            human_verdict="ANOMALY_VISIBLE",
            note="Visible anomaly.",
        )
        with sqlite3.connect(self.store.database_path) as connection:
            connection.execute("DROP TRIGGER sop_evidence_events_no_update")
            connection.execute("DROP TRIGGER sop_evidence_events_no_delete")
            connection.execute("DROP TABLE sop_evidence_events")
            connection.execute("PRAGMA user_version = 1")
        self.store.initialize()
        self.assertEqual(len(self.store.review_history("inspection-1")), 1)
        self.assertEqual(self.store.sop_evidence_history("inspection-1"), [])
        with sqlite3.connect(self.store.database_path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_schema_two_database_migrates_to_mock_ticket_workflow(self):
        bundle, _ = self.make_bundle()
        self.store.register(bundle)
        self.store.add_review(
            "inspection-1",
            expected_revision=0,
            reviewer="operator",
            human_verdict="ANOMALY_VISIBLE",
            note="Visible anomaly.",
        )
        self.store.add_sop_evidence(
            "inspection-1",
            expected_review_revision=1,
            selected_by="operator",
            defect_type="bent_lead",
            result=self.mock_sop_result(),
        )
        with sqlite3.connect(self.store.database_path) as connection:
            connection.execute("DROP TRIGGER action_draft_events_no_update")
            connection.execute("DROP TRIGGER action_draft_events_no_delete")
            connection.execute("DROP TRIGGER action_approval_events_no_update")
            connection.execute("DROP TRIGGER action_approval_events_no_delete")
            connection.execute("DROP TRIGGER mock_ticket_events_no_update")
            connection.execute("DROP TRIGGER mock_ticket_events_no_delete")
            connection.execute("DROP TABLE mock_ticket_events")
            connection.execute("DROP TABLE action_approval_events")
            connection.execute("DROP TABLE action_draft_events")
            connection.execute("PRAGMA user_version = 2")
        self.store.initialize()
        self.assertEqual(len(self.store.review_history("inspection-1")), 1)
        self.assertEqual(len(self.store.sop_evidence_history("inspection-1")), 1)
        draft = self.store.create_action_draft(
            "inspection-1", expected_review_revision=1, created_by="operator"
        )
        self.assertEqual(draft["status"], "recorded")
        with sqlite3.connect(self.store.database_path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)


if __name__ == "__main__":
    unittest.main()
