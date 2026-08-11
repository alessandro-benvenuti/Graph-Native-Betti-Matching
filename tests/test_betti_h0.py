"""Regression tests for combined matched-node H0 + H1 losses."""

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.losses.betti_h0 import h0_betti_matching_loss  # noqa: E402
from training.losses.betti_h1 import cycle_space_matching_loss  # noqa: E402


EDGES = tuple((u, v) for u in range(4) for v in range(u + 1, 4))
EDGE_TENSOR = torch.tensor(EDGES, dtype=torch.long)


class GraphBettiH0H1Tests(unittest.TestCase):
    def test_h0_missing_bridge_is_strengthened(self):
        probabilities = torch.tensor(
            [0.95, 0.0, 0.0, 0.10, 0.0, 0.90],
            dtype=torch.float64,
            requires_grad=True,
        )
        truth = torch.tensor(
            [[0, 1], [1, 2], [2, 3]],
            dtype=torch.long,
        )
        loss, matching = h0_betti_matching_loss(
            probabilities,
            EDGE_TENSOR,
            truth,
            num_vertices=4,
        )
        loss.backward()
        bridge_index = EDGES.index((1, 2))
        self.assertEqual(matching.false_prediction_rank, 1)
        self.assertLess(float(probabilities.grad[bridge_index]), 0.0)

    def test_h0_false_bridge_is_weakened(self):
        probabilities = torch.tensor(
            [0.95, 0.0, 0.0, 0.80, 0.0, 0.90],
            dtype=torch.float64,
            requires_grad=True,
        )
        truth = torch.tensor(
            [[0, 1], [2, 3]],
            dtype=torch.long,
        )
        loss, matching = h0_betti_matching_loss(
            probabilities,
            EDGE_TENSOR,
            truth,
            num_vertices=4,
        )
        loss.backward()
        bridge_index = EDGES.index((1, 2))
        self.assertEqual(matching.matched_rank, 1)
        self.assertGreater(float(probabilities.grad[bridge_index]), 0.0)

    def test_combined_loss_reaches_relation_probabilities(self):
        probabilities = torch.tensor(
            [0.95, 0.85, 0.0, 0.90, 0.75, 0.20],
            dtype=torch.float64,
            requires_grad=True,
        )
        truth = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 3]],
            dtype=torch.long,
        )
        h0_loss, _ = h0_betti_matching_loss(
            probabilities,
            EDGE_TENSOR,
            truth,
            num_vertices=4,
        )
        h1_loss, _ = cycle_space_matching_loss(
            probabilities,
            EDGE_TENSOR,
            truth,
            num_vertices=4,
        )
        combined = 0.01 * h0_loss + 0.01 * h1_loss
        combined.backward()
        self.assertTrue(torch.isfinite(combined.detach()))
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertGreater(float(probabilities.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
