"""Tests for baseline-compatible RelationFormer graph inference."""

import unittest

import torch
from torch import nn

from training.evaluation import infer_graphs


class AsymmetricRelationHead(nn.Module):
    def forward(self, features):
        # D=2, no relation token: make each direction different so the test
        # proves that inference averages logits before applying softmax.
        return torch.stack((features[:, 0], features[:, 2]), dim=-1)


class InferenceTests(unittest.TestCase):
    def _inputs(self):
        tokens = torch.tensor(
            [[[4.0, 1.0], [1.0, 5.0], [2.0, 3.0], [0.0, 0.0]]]
        )
        predictions = {
            "pred_logits": torch.tensor(
                [[[0.0, 4.0], [0.0, 3.0], [4.0, 0.0]]]
            ),
            "pred_nodes": torch.tensor(
                [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
            ),
        }
        return tokens, predictions

    def test_argmax_nodes_and_bidirectional_relation_logits(self):
        tokens, predictions = self._inputs()
        graphs = infer_graphs(
            tokens,
            predictions,
            AsymmetricRelationHead(),
            object_queries=3,
            relation_tokens=0,
            edge_threshold=0.49,
        )

        graph = graphs[0]
        self.assertEqual(graph["query_ids"].tolist(), [0, 1])
        self.assertEqual(graph["boxes"].shape, (2, 3))
        self.assertTrue(torch.equal(graph["boxes"], graph["nodes"]))
        self.assertEqual(graph["edges"].tolist(), [[0, 1]])
        # Mean logits are [2.5, 2.5], hence P(edge)=0.5 exactly.
        self.assertAlmostEqual(float(graph["edge_scores"][0]), 0.5, places=6)

    def test_probability_thresholds_and_empty_graph(self):
        tokens, predictions = self._inputs()
        graph = infer_graphs(
            tokens,
            predictions,
            AsymmetricRelationHead(),
            object_queries=3,
            relation_tokens=0,
            node_threshold=0.97,
            edge_threshold=0.9,
        )[0]

        self.assertEqual(graph["nodes"].shape, (1, 3))
        self.assertEqual(graph["edges"].shape, (0, 2))
        self.assertEqual(graph["edge_scores"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
