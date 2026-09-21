import unittest
from pathlib import Path


class AdditionalCategoryEvaluationsScriptTests(unittest.TestCase):
    def test_script_locks_evaluations_summary_and_handoff(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/run_additional_category_evaluations.ps1").read_text(
            encoding="utf-8"
        )
        for value in (
            '"scripts\\evaluate_padim.py"',
            '"scripts\\evaluate_patchcore.py"',
            '"configs\\additional-padim-evaluation-v1.toml"',
            '"configs\\additional-patchcore-evaluation-v1.toml"',
            'foreach ($seed in @(42, 43, 44))',
            'foreach ($category in @("transistor", "cable", "metal-nut"))',
            '"scripts\\summarize_three_category.py"',
            '"summarized_preregistered_three_category_mixed_evidence"',
            '"three-category-evaluation-v1.zip"',
            "No post-test model, threshold, ROI, or metric tuning was performed.",
        ):
            self.assertIn(value, text)
        self.assertNotIn("train_padim.py", text)
        self.assertNotIn("train_patchcore.py", text)


if __name__ == "__main__":
    unittest.main()
