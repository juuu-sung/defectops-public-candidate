import unittest
from pathlib import Path


class RealIADFollowupDevelopmentScriptTests(unittest.TestCase):
    def test_windows_script_locks_development_selection_and_handoff(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/run_realiad_followup_development.ps1").read_text(
            encoding="utf-8"
        )
        for value in (
            '"scripts\\register_realiad_followup.py", "verify"',
            '"scripts\\run_realiad_followup_development.py", "category"',
            '"scripts\\run_realiad_followup_development.py", "select"',
            '"realiad_pcb"',
            '"realiad_terminalblock"',
            '"realiad-followup-development-v1.zip"',
            'Final holdout remains sealed and was not scored.',
        ):
            self.assertIn(value, text)
        self.assertNotIn("final_holdout", text.split("Invoke-CheckedPython -Arguments @(")[-1])


if __name__ == "__main__":
    unittest.main()
