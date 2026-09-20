"""Synthetic bundles only; these tests never write human opinions to real cases."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from defectops.review_store import ReviewStore
from defectops.review_ui_data import load_review_case
from scripts.serve_review_ui import main as serve_main


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_ui.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 3), color).save(output, format="PNG")
    return output.getvalue()


class ReviewUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ReviewStore(self.root / "reviews.sqlite3")
        self.store.initialize()
        self.bundle = self.root / "case-1"
        self.bundle.mkdir()
        payloads = {
            "input": _png("red"),
            "overlay": _png("orange"),
            "heatmap": _png("blue"),
            "anomaly_map": b"synthetic map bytes",
        }
        artifacts = {}
        for key, data in payloads.items():
            name = f"{key.replace('_', '-')}.png" if key != "anomaly_map" else "anomaly-map.npz"
            (self.bundle / name).write_bytes(data)
            artifacts[key] = name
            artifacts[f"{key}_sha256"] = _sha(data)
        record = {
            "schema_version": 1,
            "inspection_id": "case-1",
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
                "fit_run_id": "synthetic-run",
                "checkpoint_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            },
            "artifacts": artifacts,
        }
        (self.bundle / "inspection.json").write_text(json.dumps(record), encoding="utf-8")
        self.store.register(self.bundle)

    def tearDown(self):
        self.temp.cleanup()

    def _app(self):
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            return AppTest.from_file(SCRIPT, default_timeout=10).run()

    def test_verified_display_shows_original_overlay_and_model_decision(self):
        case = load_review_case(self.store, "case-1")
        self.assertEqual(set(case["images"]), {"input", "overlay", "heatmap"})
        app = self._app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("image")), 3)
        self.assertEqual(app.metric[0].value, "검토 필요")
        self.assertEqual(app.selectbox[1].value, None)
        self.assertEqual(self.store.review_history("case-1"), [])

    def test_human_verdict_requires_acknowledgement_and_is_append_only(self):
        app = self._app()
        app.text_input[0].set_value("operator-test")
        app.selectbox[1].set_value("ANOMALY_VISIBLE")
        app.text_area[0].set_value("왼쪽 리드 굽힘이 보임")
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[0].click().run()
        self.assertEqual(self.store.review_history("case-1"), [])
        self.assertTrue(any("동의" in message.value for message in app.error))

        app.checkbox[0].set_value(True)
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[0].click().run()
        history = self.store.review_history("case-1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["revision"], 1)
        self.assertEqual(history[0]["human_verdict"], "ANOMALY_VISIBLE")
        self.assertEqual(self.store.get_inspection("case-1")["model_decision"], "REVIEW_REQUIRED")
        self.assertEqual(app.metric[3].value, "1")
        self.assertTrue(any("revision 1" in message.value for message in app.success))

    def test_checksum_mismatch_hides_images_and_review_form(self):
        (self.bundle / "overlay.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            load_review_case(self.store, "case-1")
        app = self._app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("image")), 0)
        self.assertEqual(len(app.button), 0)
        self.assertTrue(any("무결성" in message.value for message in app.error))
        self.assertEqual(self.store.review_history("case-1"), [])

    def test_failed_inspection_cannot_receive_visual_review(self):
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
        app = self._app()
        app.selectbox[0].set_value("failed-1")
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("image")), 0)
        self.assertEqual(len(app.button), 0)
        self.assertTrue(any("검사 실패" in message.value for message in app.error))

    def test_launcher_only_binds_loopback_and_rejects_missing_db(self):
        with patch("scripts.serve_review_ui.subprocess.run") as run:
            self.assertEqual(serve_main(["--db", str(self.root / "missing.sqlite3")]), 2)
            run.assert_not_called()
            run.return_value.returncode = 0
            self.assertEqual(serve_main(["--db", str(self.store.database_path), "--port", "8502"]), 0)
            command = run.call_args.args[0]
            self.assertIn("--server.address=127.0.0.1", command)
            self.assertIn("--server.port=8502", command)
            self.assertIn("--server.enableXsrfProtection=true", command)

    def test_mock_sop_requires_explicit_type_after_human_review(self):
        self.store.add_review(
            "case-1",
            expected_revision=0,
            reviewer="operator-test",
            human_verdict="ANOMALY_VISIBLE",
            note="A visible shape difference needs another look.",
        )
        app = self._app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[-1].value, "UNKNOWN")
        self.assertFalse(any("DEMO-TR-BENT@r2#OBSERVE" in x.value for x in app.caption))
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[-1].click().run()
        self.assertTrue(any("유형을 모르면" in warning.value for warning in app.warning))

        app.selectbox[-1].set_value("bent_lead")
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[-1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("DEMO-TR-BENT@r2#OBSERVE" in x.value for x in app.caption))
        self.assertTrue(any("가상 문서" in x.value for x in app.caption))
        self.assertEqual(len(self.store.review_history("case-1")), 1)
        self.assertEqual(self.store.sop_evidence_history("case-1"), [])

        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[-1].click().run()
        self.assertTrue(any("동의" in message.value for message in app.error))
        self.assertEqual(self.store.sop_evidence_history("case-1"), [])

        app.checkbox[-1].set_value(True)
        with patch.dict(os.environ, {"DEFECTOPS_REVIEW_DB": str(self.store.database_path)}):
            app.button[-1].click().run()
        evidence = self.store.sop_evidence_history("case-1")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["review_revision"], 1)
        self.assertEqual(evidence[0]["defect_type"], "bent_lead")
        self.assertEqual(evidence[0]["document_id"], "DEMO-TR-BENT")
        self.assertTrue(any("연결 기록 완료" in message.value for message in app.success))

    def test_no_sop_form_when_latest_human_review_is_uncertain(self):
        self.store.add_review(
            "case-1",
            expected_revision=0,
            reviewer="operator-test",
            human_verdict="UNCERTAIN",
            note="The image alone is not conclusive.",
        )
        app = self._app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.selectbox), 2)
        self.assertTrue(any("근거를 제시하지 않습니다" in message.value for message in app.info))


if __name__ == "__main__":
    unittest.main()
