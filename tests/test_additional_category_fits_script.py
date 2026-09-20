import unittest
from pathlib import Path


class AdditionalCategoryFitsScriptTests(unittest.TestCase):
    def test_script_locks_inputs_and_does_not_evaluate_test(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/run_additional_category_fits.ps1").read_text(
            encoding="utf-8"
        )
        for value in (
            "f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70",
            "6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5",
            'foreach ($seed in @(42, 43, 44))',
            'foreach ($model in @("padim", "patchcore"))',
            '"scripts\\train_padim.py"',
            '"scripts\\train_patchcore.py"',
            '"scripts\\prepare_data.py", "verify"',
            '"fit_calibrated_preregistered"',
        ):
            self.assertIn(value, text)
        self.assertNotIn("evaluate_padim.py", text)
        self.assertNotIn("evaluate_patchcore.py", text)


if __name__ == "__main__":
    unittest.main()
