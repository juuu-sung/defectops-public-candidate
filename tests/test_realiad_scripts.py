import unittest
from pathlib import Path


class RealIADScriptTests(unittest.TestCase):
    def test_registration_fit_and_evaluation_are_separate(self):
        root = Path(__file__).resolve().parents[1]
        prepare = (root / "scripts/prepare_realiad.ps1").read_text(encoding="utf-8")
        fits = (root / "scripts/run_realiad_patchcore_fits.ps1").read_text(encoding="utf-8")
        evaluations = (root / "scripts/run_realiad_patchcore_evaluations.ps1").read_text(
            encoding="utf-8"
        )
        gitignore = (root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/data/manifests/realiad-*.json", gitignore)
        for value in (
            '"scripts\\register_realiad.py", "register"',
            '"scripts\\register_realiad.py", "verify"',
            '"realiad-data-registration-v1.zip"',
            '"artifacts\\data-audit\\$($job.Name)\\audit.json"',
            '"artifacts\\data-audit\\$($job.Name)\\summary.md"',
        ):
            self.assertIn(value, prepare)
        for value in (
            'foreach ($seed in @(42, 43, 44))',
            '"scripts\\train_patchcore.py"',
            '"fit_calibrated_preregistered"',
        ):
            self.assertIn(value, fits)
        self.assertNotIn("evaluate_patchcore.py", fits)
        for value in (
            'foreach ($seed in @(42, 43, 44))',
            '"scripts\\evaluate_patchcore.py"',
            '"configs\\realiad-patchcore-evaluation-v1.toml"',
            '"realiad-patchcore-evaluations-v1.zip"',
        ):
            self.assertIn(value, evaluations)
        self.assertNotIn("train_patchcore.py", evaluations)


if __name__ == "__main__":
    unittest.main()
