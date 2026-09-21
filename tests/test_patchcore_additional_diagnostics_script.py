import unittest
from pathlib import Path


class PatchCoreAdditionalDiagnosticsScriptTests(unittest.TestCase):
    def test_script_uses_fixed_seed42_analysis_and_warm_contract(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/run_patchcore_additional_diagnostics.ps1").read_text(
            encoding="utf-8"
        )
        for value in (
            '@{ Category = "cable" }',
            '@{ Category = "metal-nut" }',
            '"patchcore-$($job.Category)-seed42"',
            '"scripts\\analyze_patchcore_errors.py"',
            '"scripts\\benchmark_patchcore_warm.py"',
            '"--warmup", "5"',
            '"--repeats", "1"',
            '"--count", "100"',
            '"patchcore-additional-diagnostics-v1.zip"',
            "No model, threshold, ROI, or postprocessing setting was changed.",
        ):
            self.assertIn(value, text)
        self.assertNotIn("train_patchcore.py", text)
        self.assertNotIn("evaluate_patchcore.py", text)


if __name__ == "__main__":
    unittest.main()
