"""Focused tests for focal classification and hard-negative helpers."""

import unittest

import torch
import torch.nn.functional as F

from training.losses.focal import (
    build_unmatched_relation_pairs,
    linear_progress_schedule,
    select_active_unmatched_queries,
    select_hard_unmatched_relation_logits,
    softmax_focal_loss,
)


class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_matches_weighted_cross_entropy(self):
        logits = torch.tensor([[1.0, -0.5], [-1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        weights = [0.25, 0.75]
        observed = softmax_focal_loss(logits, targets, weights, gamma=0.0)
        expected = F.cross_entropy(logits, targets, weight=torch.tensor(weights))
        self.assertTrue(torch.allclose(observed, expected))

    def test_hard_selection_retains_gradient_connection(self):
        logits = torch.tensor(
            [[2.0, -1.0], [0.0, 2.0], [-1.0, 3.0]], requires_grad=True
        )
        selected, indices = select_hard_unmatched_relation_logits(logits, 1)
        self.assertEqual(indices.tolist(), [2])
        selected.sum().backward()
        self.assertEqual(int((logits.grad.abs().sum(1) > 0).sum()), 1)

    def test_unmatched_candidates_and_curriculum(self):
        logits = torch.tensor([[3.0, -2.0], [-1.0, 2.0], [0.0, 1.0]])
        _, _, active, _ = select_active_unmatched_queries(
            logits, torch.tensor([0]), object_threshold=0.5, max_active_unmatched=2
        )
        pairs = build_unmatched_relation_pairs(torch.tensor([0]), active)
        self.assertEqual(pairs.shape, (3, 2))
        self.assertEqual(linear_progress_schedule(20, 2.0, 10, 30), 1.0)


if __name__ == "__main__":
    unittest.main()
