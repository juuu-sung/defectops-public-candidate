"""Source-grounded simulation lookup; no real operational instructions."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from defectops.mock_sop import find_mock_sop, load_mock_catalog


CATALOG_PATH = Path(__file__).resolve().parents[1] / "sops" / "mock-catalog-v1.json"


class MockSopTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_mock_catalog(CATALOG_PATH)

    def search(self, defect_type="bent_lead", product="transistor", verdict="ANOMALY_VISIBLE", when=None):
        return find_mock_sop(
            self.catalog,
            product=product,
            defect_type=defect_type,
            human_verdict=verdict,
            as_of=when or date(2026, 9, 19),
        )

    def test_only_active_revision_and_exact_verifiable_excerpts(self):
        result = self.search()
        self.assertEqual(result["status"], "found_simulation_only")
        self.assertEqual((result["document_id"], result["revision"]), ("DEMO-TR-BENT", 2))
        self.assertEqual(len(result["catalog_sha256"]), 64)
        active = next(
            doc for doc in self.catalog["documents"]
            if doc["document_id"] == "DEMO-TR-BENT" and doc["revision"] == 2
        )
        self.assertEqual(
            result["sources"],
            [
                {"citation_id": f"DEMO-TR-BENT@r2#{part['section_id']}", "text": part["text"]}
                for part in active["sections"]
            ],
        )
        self.assertNotIn("폐기된 예시 문구", str(result))

    def test_no_inferred_type_no_review_wrong_product_or_date(self):
        self.assertEqual(self.search(defect_type="UNKNOWN")["status"], "defect_type_required")
        self.assertEqual(self.search(verdict=None)["status"], "human_review_required")
        self.assertEqual(self.search(verdict="UNCERTAIN")["status"], "human_review_required")
        self.assertEqual(self.search(verdict="NO_ANOMALY_VISIBLE")["status"], "human_review_required")
        self.assertEqual(self.search(product="cable")["status"], "not_found")
        self.assertEqual(self.search(when=date(2026, 8, 15))["status"], "not_found")
        with self.assertRaisesRegex(ValueError, "Unknown defect type"):
            self.search(defect_type="free-text inference")

    def test_all_four_explicit_categories_have_mock_sources(self):
        for defect_type in ("bent_lead", "cut_lead", "damaged_case", "misplaced"):
            with self.subTest(defect_type=defect_type):
                self.assertEqual(self.search(defect_type=defect_type)["status"], "found_simulation_only")

    def test_ambiguous_active_revisions_fail_closed(self):
        duplicate = dict(self.catalog["documents"][1])
        duplicate["revision"] = 3
        self.catalog["documents"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Ambiguous active"):
            self.search()

    def test_catalog_rejects_missing_simulation_marker_duplicate_citation_and_bad_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.json"
            altered = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            altered["purpose"] = "real_sop"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "simulation_only"):
                load_mock_catalog(path)

            altered["purpose"] = "simulation_only"
            altered["documents"][1]["sections"].append(altered["documents"][1]["sections"][0])
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate mock SOP section ID"):
                load_mock_catalog(path)

            altered["documents"][1]["sections"].pop()
            altered["documents"][1]["valid_to"] = "2026-08-31"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective dates are reversed"):
                load_mock_catalog(path)

            altered["documents"][1].pop("valid_to")
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid_to field is missing"):
                load_mock_catalog(path)


if __name__ == "__main__":
    unittest.main()
