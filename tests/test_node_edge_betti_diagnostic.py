"""Focused tests for the frozen node/edge Betti diagnostic."""

import unittest

import torch

from models.matcher import HungarianMatcher
from scripts.diagnose_node_edge_betti import (
    _evaluate_mode,
    _original_target_cycle_diagnostics,
)
from tests.test_graph_losses import CountingRelationHead, _batch, _config
from training.losses import GraphCriterion


class NodeEdgeBettiDiagnosticTests(unittest.TestCase):
    def test_original_target_cycle_reports_missing_matched_node(self):
        edges = torch.tensor(
            [[0, 1], [1, 2], [0, 2], [1, 0]], dtype=torch.long
        )
        represented = _original_target_cycle_diagnostics(
            edges,
            target_node_count=3,
            matched_target_nodes=[0, 1, 2],
        )
        blocked = _original_target_cycle_diagnostics(
            edges,
            target_node_count=3,
            matched_target_nodes=[0, 1],
        )
        self.assertEqual(len(represented), 1)
        self.assertTrue(represented[0]["represented_in_local_target"])
        self.assertEqual(blocked[0]["missing_matched_target_nodes"], [2])
        self.assertFalse(blocked[0]["represented_in_local_target"])

    def test_both_complexes_produce_serializable_gradient_records(self):
        config = _config()
        config["topology"]["complex"].update(
            mode="node_aware",
            aggregation="hybrid",
            alpha=0.5,
            unmatched_object_threshold=0.0,
            max_active_unmatched=1,
        )
        matcher_config = config["model"]["matcher"]
        criterion = GraphCriterion(
            config,
            HungarianMatcher(
                matcher_config["class_cost"], matcher_config["node_cost"]
            ),
            CountingRelationHead(),
        )
        tokens, predictions, targets = _batch()
        source = torch.tensor([0, 1, 2])
        target = torch.tensor([0, 1, 2])

        matched = _evaluate_mode(
            criterion,
            tokens[0],
            predictions["pred_logits"][0],
            predictions["pred_nodes"][0],
            targets["edges"][0],
            source,
            target,
            mode="matched_only",
            aggregation="hybrid",
            alpha=0.5,
        )
        augmented = _evaluate_mode(
            criterion,
            tokens[0],
            predictions["pred_logits"][0],
            predictions["pred_nodes"][0],
            targets["edges"][0],
            source,
            target,
            mode="node_aware",
            aggregation="hybrid",
            alpha=0.5,
        )

        self.assertEqual(matched["active_unmatched_count"], 0)
        self.assertEqual(augmented["active_unmatched_count"], 1)
        self.assertEqual(len(matched["nodes"]), 3)
        self.assertEqual(len(augmented["nodes"]), 4)
        self.assertTrue(torch.isfinite(torch.tensor(augmented["h0_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(augmented["h1_loss"])))
        unmatched = [node for node in augmented["nodes"] if not node["matched"]]
        self.assertEqual(len(unmatched), 1)
        self.assertTrue(unmatched[0]["topology_can_update_node"])


if __name__ == "__main__":
    unittest.main()
