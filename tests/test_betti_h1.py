"""Regression tests for the matched-node graph H1 loss.

Run from the 3d directory:
    python tests/test_graph_betti_h1.py
"""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.losses.betti_h1 import (  # noqa: E402
    compute_cycle_space_matching,
    cycle_space_matching_loss,
)
from training.losses.focal import softmax_focal_loss  # noqa: E402


def complete_edges(num_vertices):
    return tuple(
        (left, right)
        for left in range(num_vertices)
        for right in range(left + 1, num_vertices)
    )


class GraphBettiH1Tests(unittest.TestCase):
    def test_missing_true_cycle_strengthens_critical_edge(self):
        edges = complete_edges(3)
        probabilities = torch.tensor(
            [0.9, 0.8, 0.0], dtype=torch.float64, requires_grad=True
        )
        truth = torch.tensor(edges, dtype=torch.long)
        loss, result = cycle_space_matching_loss(
            probabilities,
            torch.tensor(edges, dtype=torch.long),
            truth,
            num_vertices=3,
            normalize=False,
        )
        loss.backward()
        self.assertEqual(result.missed_target_rank, 1)
        critical = result.target_classes[
            result.unmatched_target_indices[0]
        ].birth_edge_index
        self.assertLess(float(probabilities.grad[critical]), 0.0)

    def test_h1_weight_and_ramp_scale_only_h1_gradient(self):
        edges = complete_edges(4)
        base = torch.tensor(
            [0.95, 0.85, 0.0, 0.90, 0.75, 0.80], dtype=torch.float64
        )
        truth = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 3]], dtype=torch.long
        )

        def gradient(weight):
            probabilities = base.clone().requires_grad_(True)
            topology, _ = cycle_space_matching_loss(
                probabilities,
                torch.tensor(edges, dtype=torch.long),
                truth,
                num_vertices=4,
            )
            classification = (probabilities - 0.5).pow(2).mean()
            (classification + weight * topology).backward()
            return probabilities.grad

        zero = gradient(0.0)
        half = gradient(0.5)
        full = gradient(1.0)
        self.assertTrue(torch.allclose(half - zero, 0.5 * (full - zero)))

    def test_betti_ramp_does_not_scale_focal_or_edge_ce_gradients(self):
        edges = complete_edges(4)
        base = torch.tensor(
            [0.95, 0.85, 0.0, 0.90, 0.75, 0.80], dtype=torch.float64
        )
        truth = torch.tensor(
            [[0, 1], [0, 2], [1, 2], [2, 3]], dtype=torch.long
        )

        def gradients(multiplier):
            probabilities = base.clone().requires_grad_(True)
            node_logits = torch.tensor(
                [[0.2, 0.8], [0.7, 0.3]],
                dtype=torch.float64,
                requires_grad=True,
            )
            edge_logits = torch.tensor(
                [[0.4, 0.6], [0.9, 0.1]],
                dtype=torch.float64,
                requires_grad=True,
            )
            topology, _ = cycle_space_matching_loss(
                probabilities,
                torch.tensor(edges, dtype=torch.long),
                truth,
                num_vertices=4,
            )
            focal = softmax_focal_loss(
                node_logits, torch.tensor([1, 0]), [1.0, 1.0], gamma=2.0
            )
            edge_ce = F.cross_entropy(edge_logits, torch.tensor([1, 0]))
            (focal + edge_ce + multiplier * topology).backward()
            return probabilities.grad, node_logits.grad, edge_logits.grad

        half = gradients(0.5)
        full = gradients(1.0)
        self.assertTrue(torch.allclose(half[0], 0.5 * full[0]))
        self.assertTrue(torch.allclose(half[1], full[1]))
        self.assertTrue(torch.allclose(half[2], full[2]))
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
