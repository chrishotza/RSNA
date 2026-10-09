import unittest
from pathlib import Path
from scripts.validate_notebook_experiment import same_python_statement, validate

ROOT = Path(__file__).resolve().parents[1]

class GateTests(unittest.TestCase):
    def test_inline_comment_is_ignored(self):
        self.assertTrue(same_python_statement("AMP_PREF = 'bf16'  # note", "AMP_PREF = 'bf16'"))

    def test_different_value_is_rejected(self):
        self.assertFalse(same_python_statement("AMP_PREF = 'fp16'  # note", "AMP_PREF = 'bf16'"))

    def test_a1_candidate_matches_frozen_baseline(self):
        result = validate(ROOT / "notebooks/A0_public_0941/a0-submitted-freeze.ipynb", ROOT / "notebooks/experiments/EXP-A1-amp-auto-t4.ipynb", "EXP-A1", 19, "AMP_PREF = 'bf16'", "AMP_PREF = 'auto'", None)
        self.assertEqual(result["changed_code_cells"], [19])
        self.assertEqual(result["status"], "PASS")

if __name__ == "__main__":
    unittest.main(verbosity=2)
