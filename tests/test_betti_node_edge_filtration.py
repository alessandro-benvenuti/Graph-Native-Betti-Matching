"""Tests for node-aware graph-native Betti filtrations."""

import unittest

import torch

from training.losses.betti_filtration import node_edge_confidences
from training.losses.betti_h0 import h0_betti_matching_loss
from training.losses.betti_h1 import cycle_space_matching_loss


class NodeEdgeFiltrationTests(unittest.TestCase):
    def test_rules_have_expected_values_and_respect_faces(self):
        nodes = torch.tensor([0.3, 0.8], dtype=torch.float64)
        edges = torch.tensor([0.7], dtype=torch.float64)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)

        product = node_edge_confidences(
            nodes, edges, pairs, aggregation="product"
        )
        minimum = node_edge_confidences(
            nodes, edges, pairs, aggregation="min"
        )
        hybrid = node_edge_confidences(
            nodes, edges, pairs, aggregation="hybrid", alpha=0.5
        )

        self.assertTrue(
            torch.allclose(product, torch.tensor([0.168], dtype=torch.float64))
        )
        self.assertTrue(
            torch.allclose(minimum, torch.tensor([0.3], dtype=torch.float64))
        )
        self.assertTrue(
            torch.allclose(hybrid, torch.tensor([0.234], dtype=torch.float64))
        )
        self.assertLessEqual(float(hybrid), float(nodes.min()))
        self.assertLessEqual(float(hybrid), float(edges[0]))

    def test_matched_endpoints_recover_raw_edge_probability(self):
        nodes = torch.ones(2, dtype=torch.float64)
        edge = torch.tensor([0.4], dtype=torch.float64)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        for aggregation in ("product", "min", "hybrid"):
            observed = node_edge_confidences(
                nodes, edge, pairs, aggregation=aggregation, alpha=0.37
            )
            self.assertTrue(torch.allclose(observed, edge))

    def test_hybrid_reaches_all_inputs_away_from_ties(self):
        nodes = torch.tensor([0.3, 0.8], dtype=torch.float64, requires_grad=True)
        edge = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        score = node_edge_confidences(
            nodes, edge, pairs, aggregation="hybrid", alpha=0.5
        )
        score.sum().backward()
        self.assertTrue((nodes.grad > 0).all())
        self.assertGreater(float(edge.grad[0]), 0.0)
        self.assertGreater(float(nodes.grad[0]), float(nodes.grad[1]))

    def test_h0_isolated_unmatched_node_is_suppressed(self):
        unmatched = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        nodes = torch.stack((unmatched.new_tensor(1.0), unmatched))
        effective_edges = torch.tensor(
            [0.0], dtype=torch.float64, requires_grad=True
        )
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        truth = torch.empty((0, 2), dtype=torch.long)
        presence = torch.tensor([1.0, 0.0], dtype=torch.float64)

        loss, matching = h0_betti_matching_loss(
            effective_edges,
            pairs,
            truth,
            num_vertices=2,
            node_probabilities=nodes,
            target_node_presence=presence,
            normalize=False,
        )
        loss.backward()

        self.assertEqual(matching.false_prediction_rank, 1)
        self.assertGreater(float(unmatched.grad), 0.0)

    def test_h1_false_loop_reaches_unmatched_node_and_edge(self):
        node_logits = torch.tensor(
            [1.0, 1.0, 1.0, 0.6], dtype=torch.float64, requires_grad=True
        )
        raw_edges = torch.tensor(
            [0.95, 0.90, 0.80, 0.75],
            dtype=torch.float64,
            requires_grad=True,
        )
        pairs = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long
        )
        effective = node_edge_confidences(
            node_logits,
            raw_edges,
            pairs,
            aggregation="hybrid",
            alpha=0.5,
        )
        truth = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        loss, matching = cycle_space_matching_loss(
            effective,
            pairs,
            truth,
            num_vertices=4,
            normalize=False,
        )
        loss.backward()

        self.assertEqual(matching.false_prediction_rank, 1)
        self.assertGreater(float(node_logits.grad[3]), 0.0)
        self.assertGreater(float(raw_edges.grad[3]), 0.0)

    def test_h1_node_gradient_matches_finite_difference(self):
        pairs = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long
        )
        truth = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        def evaluate(value, *, gradient=False):
            q = torch.tensor(value, dtype=torch.float64, requires_grad=gradient)
            nodes = torch.cat((q.new_ones(3), q.unsqueeze(0)))
            raw = q.new_tensor([0.95, 0.90, 0.80, 0.75])
            effective = node_edge_confidences(
                nodes, raw, pairs, aggregation="hybrid", alpha=0.5
            )
            loss, _ = cycle_space_matching_loss(
                effective,
                pairs,
                truth,
                num_vertices=4,
                normalize=False,
            )
            return q, loss

        q, loss = evaluate(0.6, gradient=True)
        loss.backward()
        step = 1.0e-6
        _, high = evaluate(0.6 + step)
        _, low = evaluate(0.6 - step)
        finite_difference = float((high - low) / (2.0 * step))
        self.assertAlmostEqual(float(q.grad), finite_difference, places=5)

    def test_h0_duplicate_bridge_can_be_reinforced(self):
        duplicate = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        nodes = torch.stack(
            (
                duplicate.new_tensor(1.0),
                duplicate.new_tensor(1.0),
                duplicate,
                duplicate.new_tensor(1.0),
            )
        )
        raw_edges = torch.tensor(
            [0.95, 0.85, 0.90, 0.05],
            dtype=torch.float64,
            requires_grad=True,
        )
        pairs = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [1, 3]], dtype=torch.long
        )
        effective = node_edge_confidences(
            nodes,
            raw_edges,
            pairs,
            aggregation="hybrid",
            alpha=0.5,
        )
        truth = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
        presence = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        loss, _ = h0_betti_matching_loss(
            effective,
            pairs,
            truth,
            num_vertices=4,
            node_probabilities=nodes,
            target_node_presence=presence,
            normalize=False,
        )
        loss.backward()

        # The duplicate provides a much stronger route than the direct 1-3
        # relation. Connectivity supervision therefore increases its score;
        # node focal/cardinality supervision must oppose an unwanted duplicate.
        self.assertLess(float(duplicate.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
