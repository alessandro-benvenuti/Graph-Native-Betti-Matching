"""Tests for representative loop-patch selection."""

import unittest

from scripts.visualize_loop_patch_examples import select_examples


def _row(sample_id, outcome, control_false, betti_false):
    return {
        "source_sample_id": sample_id,
        "outcome": outcome,
        "control_false_cycle_rank": control_false,
        "betti_false_cycle_rank": betti_false,
        "control_predicted_beta1": control_false,
        "betti_predicted_beta1": betti_false,
    }


class VisualizeLoopPatchExamplesTests(unittest.TestCase):
    def test_prioritizes_fixed_regressed_then_largest_suppression(self):
        rows = [
            _row("ordinary", "both_inexact", 2, 1),
            _row("suppressed", "both_inexact", 8, 1),
            _row("fixed", "betti_fixed", 1, 0),
            _row("regressed", "betti_regressed", 0, 1),
        ]
        selected = select_examples(rows, limit=4)
        self.assertEqual(
            [item["source_sample_id"] for item in selected],
            ["fixed", "regressed", "suppressed", "ordinary"],
        )


if __name__ == "__main__":
    unittest.main()
