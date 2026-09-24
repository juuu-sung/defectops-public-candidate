"""No-GPU checks for the post-test PCB3 normal-bank comparison contract."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from defectops.visa_pcb_normal_bank_evaluation import (
    _require_zip, _same_metrics, load_comparison, paired_summary,
)


class VisaPcbNormalBankEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.config = load_comparison(root / "configs" / "visa-pcb-normal-bank-pcb3-evaluation-v1.toml")

    def test_locked_config_names_completed_fit_and_prior_evaluation(self) -> None:
        self.assertEqual(self.config["fit_handoff_sha256"],
                         "93d998fd88a3983c35688aaee850de794eedabbcf2b1d79471eaa50e3c195b07")
        self.assertEqual(self.config["normal_test_count"], 101)
        self.assertEqual(self.config["anomaly_test_count"], 100)
        self.assertEqual(self.config["mask_binarization"], "nonzero_before_nearest")
        self.assertFalse(self.config["automatic_pass_allowed"])
        self.assertFalse(self.config["confirmatory_claim_allowed"])

    def test_paired_changes_have_correct_sign_and_denominators(self) -> None:
        anomaly = ([{"label": 1, "baseline_decision": "PASS", "v2_decision": "REVIEW_REQUIRED"}] * 15
                   + [{"label": 1, "baseline_decision": "REVIEW_REQUIRED", "v2_decision": "PASS"}] * 5
                   + [{"label": 1, "baseline_decision": "PASS", "v2_decision": "PASS"}] * 58
                   + [{"label": 1, "baseline_decision": "REVIEW_REQUIRED", "v2_decision": "REVIEW_REQUIRED"}] * 22)
        normal = ([{"label": 0, "baseline_decision": "PASS", "v2_decision": "REVIEW_REQUIRED"}] * 3
                  + [{"label": 0, "baseline_decision": "REVIEW_REQUIRED", "v2_decision": "PASS"}] * 2
                  + [{"label": 0, "baseline_decision": "PASS", "v2_decision": "PASS"}] * 96)
        result = paired_summary(anomaly + normal, self.config)
        self.assertAlmostEqual(result["escape_rate_improvement_baseline_minus_v2"], 0.10)
        self.assertEqual(result["anomaly_old_missed_new_caught"], 15)
        self.assertEqual(result["anomaly_old_caught_new_missed"], 5)
        self.assertEqual(result["normal_old_pass_new_review"], 3)
        self.assertEqual(result["normal_old_review_new_pass"], 2)
        self.assertEqual(result, paired_summary(anomaly + normal, self.config))
        with self.assertRaisesRegex(ValueError, "denominator"):
            paired_summary(anomaly + normal[:-1], self.config)

    def test_zip_rejects_extra_member_and_bad_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("fit.json", json.dumps({"status": "fit"}))
                archive.writestr("extra.json", "{}")
            from defectops.visa_pcb_external_registration import file_sha256
            with self.assertRaisesRegex(ValueError, "members or CRC"):
                _require_zip(path, ["fit.json"], file_sha256(path))
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                _require_zip(path, ["fit.json", "extra.json"], "0" * 64)

    def test_stable_point_metrics_allow_roundoff_only(self) -> None:
        left = {"image_auroc": 0.6, "image_average_precision": 0.7,
                "fixed_threshold": {"counts": {"tp": 1, "fp": 2, "tn": 3, "fn": 4},
                                    "rates": {"recall": 0.2}}}
        right = {"image_auroc": 0.6, "image_average_precision": 0.7,
                 "fixed_threshold": {"counts": dict(left["fixed_threshold"]["counts"]),
                                     "rates": {"recall": 0.2 + 1e-15}}}
        self.assertTrue(_same_metrics(left, right))
        right["fixed_threshold"]["counts"]["fn"] = 5
        self.assertFalse(_same_metrics(left, right))


if __name__ == "__main__":
    unittest.main()
