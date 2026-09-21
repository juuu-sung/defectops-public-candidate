import unittest
from pathlib import Path

from defectops.realiad_multiseed_diagnostics import (
    build_consensus_cases,
    select_sheet_cases,
    summarize_cases,
)


def bundle(seed, decisions):
    rows = []
    for index, (label, defect_type, decision, score) in enumerate(decisions):
        rows.append(
            {
                "image_path": f"real_iad/images/example/{index}.jpg",
                "label": label,
                "defect_type": defect_type,
                "decision": decision,
                "raw_score": score,
            }
        )
    return {
        "seed": seed,
        "sample_ids": [row["image_path"] for row in rows],
        "rows": rows,
        "image_threshold": 10.0,
    }


class RealIADMultiseedDiagnosticsTests(unittest.TestCase):
    def test_consensus_and_unstable_groups_are_exact(self):
        decisions = {
            42: [
                (1, "A", "PASS", 8.0),
                (1, "B", "REVIEW_REQUIRED", 12.0),
                (0, "OK", "PASS", 7.0),
                (0, "OK", "REVIEW_REQUIRED", 11.0),
            ],
            43: [
                (1, "A", "PASS", 9.0),
                (1, "B", "PASS", 9.0),
                (0, "OK", "PASS", 8.0),
                (0, "OK", "PASS", 9.0),
            ],
            44: [
                (1, "A", "PASS", 9.5),
                (1, "B", "REVIEW_REQUIRED", 11.0),
                (0, "OK", "PASS", 8.5),
                (0, "OK", "REVIEW_REQUIRED", 12.0),
            ],
        }
        cases = build_consensus_cases([bundle(seed, decisions[seed]) for seed in (42, 43, 44)])
        self.assertEqual(
            [case["group"] for case in cases],
            ["consensus_missed", "unstable_anomaly", "consensus_normal", "unstable_normal"],
        )
        self.assertEqual([case["review_votes"] for case in cases], [0, 2, 0, 2])
        summary = summarize_cases(cases)
        self.assertEqual(summary["groups"]["consensus_missed"], 1)
        self.assertEqual(summary["by_defect_type"]["A"]["consensus_missed"], 1)
        self.assertAlmostEqual(summary["by_defect_type"]["B"]["mean_seed_recall"], 2 / 3)

    def test_sheet_selection_is_bounded_and_includes_reviewed_normals(self):
        cases = []
        for index in range(30):
            cases.append(
                {
                    "index": index,
                    "image_path": f"{index}.jpg",
                    "label": 1,
                    "defect_type": "A" if index % 2 else "B",
                    "group": "consensus_missed",
                    "review_votes": 0,
                    "mean_margin_ratio": -index / 100,
                }
            )
        cases.append(
            {
                "index": 30,
                "image_path": "normal.jpg",
                "label": 0,
                "defect_type": "OK",
                "group": "unstable_normal",
                "review_votes": 2,
                "mean_margin_ratio": 0.1,
            }
        )
        selected = select_sheet_cases(cases)
        self.assertEqual(len(selected["consensus-false-negatives"]), 20)
        self.assertEqual(len(selected["reviewed-normal-cases"]), 1)
        self.assertEqual(
            {case["defect_type"] for case in selected["consensus-false-negatives"]},
            {"A", "B"},
        )

    def test_windows_script_only_runs_diagnostics_and_builds_handoff(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/run_realiad_patchcore_diagnostics.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scripts\\analyze_realiad_multiseed.py"', script)
        self.assertIn('"realiad-patchcore-multiseed-diagnostics-v1.zip"', script)
        self.assertNotIn("train_patchcore.py", script)
        self.assertNotIn("evaluate_patchcore.py", script)


if __name__ == "__main__":
    unittest.main()
