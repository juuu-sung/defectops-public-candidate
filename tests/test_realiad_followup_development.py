import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from defectops.manifest import canonical
from defectops.realiad_followup_development import (
    _assert_final_not_extracted,
    candidate_image_score,
    load_followup_development_execution,
    select_followup_candidate,
)
from defectops.realiad_followup_protocol import load_realiad_followup_protocol


class RealIADFollowupDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.execution_path = (
            self.root / "configs/realiad-followup-development-execution-v1.toml"
        )
        self.protocol_path = self.root / "configs/realiad-followup-preregistration-v1.toml"
        self.execution = load_followup_development_execution(self.execution_path)
        self.protocol = load_realiad_followup_protocol(self.protocol_path)
        self.git = {"commit": "a" * 40, "branch": "main"}

    def test_execution_contract_is_locked_to_registered_manifests(self):
        self.assertEqual(self.execution["development_seed"], 42)
        self.assertEqual(self.execution["benchmark"]["warmup"], 5)
        self.assertEqual(self.execution["benchmark"]["count"], 100)
        self.assertEqual(
            self.execution["manifests"]["realiad_pcb"]["development_sha256"],
            "2d16e21b2e665defbf54266c1d5e8b08f41838dfae679fff8f65dff615b49ed4",
        )
        original = self.execution_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.toml"
            changed.write_text(original.replace("warmup = 5", "warmup = 4"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "benchmark"):
                load_followup_development_execution(changed)

    def test_candidate_image_score_uses_exact_top_fraction(self):
        anomaly_map = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        raw = {"image_score": "anomalib_patchcore_raw", "top_fraction": 0.0}
        top_one = {"image_score": "mean_of_top_map_fraction", "top_fraction": 0.25}
        top_two = {"image_score": "mean_of_top_map_fraction", "top_fraction": 0.50}
        self.assertEqual(candidate_image_score(9.0, anomaly_map, raw), 9.0)
        self.assertEqual(candidate_image_score(9.0, anomaly_map, top_one), 4.0)
        self.assertEqual(candidate_image_score(9.0, anomaly_map, top_two), 3.5)

    def test_selection_applies_constraint_then_registered_tie_breakers(self):
        results = self.results()
        report = select_followup_candidate(results, self.protocol, self.execution, self.git)
        self.assertEqual(report["selected_candidate_id"], "map_top_1pct")
        self.assertTrue(report["final_gate_ready"])
        self.assertFalse(report["final_holdout_evaluated"])
        unsigned = dict(report)
        claimed = unsigned.pop("selection_report_sha256")
        self.assertEqual(hashlib.sha256(canonical(unsigned)).hexdigest(), claimed)
        by_id = {row["candidate_id"]: row for row in report["candidate_results"]}
        self.assertFalse(by_id["map_top_5pct"]["eligible"])

    def test_selection_keeps_final_sealed_when_no_candidate_is_eligible(self):
        results = self.results()
        for result in results.values():
            for candidate in result["candidates"].values():
                candidate["metrics"]["fixed_threshold"]["rates"]["overkill_rate"] = 0.2
        report = select_followup_candidate(results, self.protocol, self.execution, self.git)
        self.assertEqual(report["status"], "no_eligible_candidate")
        self.assertIsNone(report["selected_candidate_id"])
        self.assertFalse(report["final_gate_ready"])

    def test_extracted_final_member_is_rejected(self):
        manifest = {
            "sealed": True,
            "images_extracted": False,
            "records": [
                {
                    "source_image_member": "pcb/NG/D1/S0001/final.jpg",
                    "source_mask_member": "pcb/NG/D1/S0001/final.png",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            _assert_final_not_extracted(root, manifest)
            path = root / "real_iad_followup/images/pcb/NG/D1/S0001/final.jpg"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"sealed")
            with self.assertRaisesRegex(RuntimeError, "sealed final member is extracted"):
                _assert_final_not_extracted(root, manifest)

    def results(self):
        candidates = [candidate["id"] for candidate in self.protocol["candidates"]]
        escape = {
            "baseline_raw": 0.10,
            "map_top_1pct": 0.10,
            "map_top_5pct": 0.08,
            "fine_features_top_1pct": 0.20,
        }
        auroc = {
            "baseline_raw": 0.90,
            "map_top_1pct": 0.95,
            "map_top_5pct": 0.96,
            "fine_features_top_1pct": 0.99,
        }
        results = {}
        for category in ("realiad_pcb", "realiad_terminalblock"):
            per_candidate = {}
            for candidate_id in candidates:
                overkill = 0.11 if candidate_id == "map_top_5pct" else 0.05
                per_candidate[candidate_id] = {
                    "metrics": {
                        "image_auroc": auroc[candidate_id],
                        "fixed_threshold": {
                            "rates": {
                                "escape_rate": escape[candidate_id],
                                "overkill_rate": overkill,
                            }
                        },
                    },
                    "warm_latency_ms": {"p95_higher": 40.0},
                }
            results[category] = {
                "status": "evaluated_preregistered_development",
                "category": category,
                "final_holdout_evaluated": False,
                "plan_sha256": self.execution["plan_sha256"],
                "execution_config_sha256": self.execution["config_sha256"],
                "git": copy.deepcopy(self.git),
                "final_manifest_sha256": self.execution["manifests"][category][
                    "final_sha256"
                ],
                "candidates": per_candidate,
            }
        return results


if __name__ == "__main__":
    unittest.main()
