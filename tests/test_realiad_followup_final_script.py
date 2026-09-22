import unittest
from pathlib import Path


class RealIADFollowupFinalScriptTests(unittest.TestCase):
    def test_windows_script_requires_explicit_final_open_switch(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts/run_realiad_followup_final.ps1").read_text(encoding="utf-8")
        guard = "if (-not $OpenFinalHoldout)"
        self.assertIn('param([switch]$OpenFinalHoldout)', script)
        self.assertIn(guard, script)
        self.assertLess(script.index('"fit",'), script.index(guard))
        self.assertGreater(script.index('"evaluate",'), script.index(guard))
        self.assertIn('"summarize"', script)
        self.assertIn('realiad-followup-final-v1.zip', script)
        self.assertNotIn('Remove-Item', script)


if __name__ == "__main__":
    unittest.main()
