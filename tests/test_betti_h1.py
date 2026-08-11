"""Regression tests for the matched-node graph H1 loss.

Run from the 3d directory:
    python tests/test_graph_betti_h1.py
"""

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.losses.betti_h1 import (  # noqa: E402
    compute_cycle_space_matching,
    cycle_space_matching_loss,
)


def complete_edges(num_vertices):
    return tuple(
        (left, right)
        for left in range(num_vertices)
        for right in range(left + 1, num_vertices)
    )


class GraphBettiH1Tests(unittest.TestCase):
    def test_true_cycle_and_false_cycle_are_separated(self):
        edges = complete_edges(4)
        probabilities = {
            (0, 1): 0.95,
            (0, 2): 0.85,
            (0, 3): 0.0,
            (1, 2): 0.90,
            (1, 3): 0.75,
            (2, 3): 0.80,
        }
        truth = {(0, 1), (0, 2), (1, 2), (2, 3)}
        result = compute_cycle_space_matching(
            [probabilities[edge] for edge in edges],
            edges,
            truth,
            num_vertices=4,
        )
        self.assertEqual(result.shared_rank, 1)
        self.assertEqual(result.false_prediction_rank, 1)
        self.assertEqual(result.missed_target_rank, 0)
        self.assertEqual(result.union_only_rank, 0)

    def test_matching_is_independent_of_fundamental_basis(self):
        edges = complete_edges(4)
        active = {(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)}
        left = [
            0.55 + 0.07 * index if edge in active else 0.0
            for index, edge in enumerate(edges)
        ]
        # The binary target uses canonical tie-breaking, while the prediction
        # confidence order produces a different spanning-forest basis.
        result = compute_cycle_space_matching(
            left,
            edges,
            active,
            num_vertices=4,
        )
        self.assertEqual(result.shared_rank, 2)
        self.assertEqual(result.false_prediction_rank, 0)
        self.assertEqual(result.missed_target_rank, 0)

    def test_union_only_cycle_is_not_matched(self):
        edges = complete_edges(4)
        prediction = {(0, 1): 0.9, (2, 3): 0.8}
        target = {(1, 2), (0, 3)}
        result = compute_cycle_space_matching(
            [prediction.get(edge, 0.0) for edge in edges],
            edges,
            target,
            num_vertices=4,
        )
        self.assertEqual(result.shared_rank, 0)
        self.assertEqual(result.false_prediction_rank, 0)
        self.assertEqual(result.missed_target_rank, 0)
        self.assertEqual(result.union_only_rank, 1)

    def test_gradients_strengthen_true_and_weaken_false_birth_edges(self):
        edges = complete_edges(4)
        probabilities = torch.tensor(
            [0.95, 0.85, 0.0, 0.90, 0.75, 0.80],
            dtype=torch.float64,
            requires_grad=True,
        )
        edge_tensor = torch.tensor(edges, dtype=torch.long)
        truth = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 3]],
            dtype=torch.long,
        )
        loss, result = cycle_space_matching_loss(
            probabilities,
            edge_tensor,
            truth,
            num_vertices=4,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        for match in result.matches:
            birth_index = result.prediction_classes[
                match.prediction_index
            ].birth_edge_index
            self.assertLess(float(probabilities.grad[birth_index]), 0.0)
        for index in result.unmatched_prediction_indices:
            birth_index = result.prediction_classes[index].birth_edge_index
            self.assertGreater(float(probabilities.grad[birth_index]), 0.0)


if __name__ == "__main__":
    unittest.main()

