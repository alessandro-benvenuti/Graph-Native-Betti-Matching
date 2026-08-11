"""Contract tests for baseline, Betti, and focal/HNM loss modes."""

import copy
from pathlib import Path
import unittest

import torch
from torch import nn

from configs import load_config
from models.matcher import HungarianMatcher
from training.losses import GraphCriterion
from training.losses.criterion import _ratio_upsample


def _config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pretrain_mixed.yaml",
        environment={
            "GNBM_OUTPUT_DIR": "/outputs",
            "PLANTS_DATASET": "/plants",
            "SYNTHETIC_MRI_DATASET": "/synthetic",
        },
    )
    config = copy.deepcopy(config)
    config["model"]["decoder"].update(
        hidden_dim=8, object_queries=4, relation_tokens=1
    )
    return config


class CountingRelationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(24, 2)
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return self.linear(features)


def _batch():
    torch.manual_seed(13)
    tokens = torch.randn((1, 5, 8), requires_grad=True)
    logits = torch.tensor(
        [[[0.0, 4.0], [0.0, 3.0], [0.0, 2.0], [3.0, 0.0]]],
        requires_grad=True,
    )
    nodes = torch.tensor(
        [[[0.15, 0.20, 0.25, 0.2, 0.2, 0.2],
          [0.45, 0.50, 0.55, 0.2, 0.2, 0.2],
          [0.75, 0.70, 0.65, 0.2, 0.2, 0.2],
          [0.05, 0.90, 0.10, 0.2, 0.2, 0.2]]],
        requires_grad=True,
    )
    targets = {
        "nodes": [torch.tensor([[0.15, 0.20, 0.25], [0.45, 0.50, 0.55], [0.75, 0.70, 0.65]])],
        "edges": [torch.tensor([[0, 1], [1, 2]], dtype=torch.long)],
    }
    return tokens, {"pred_logits": logits, "pred_nodes": nodes}, targets


class HungarianMatcherTests(unittest.TestCase):
    def test_expected_assignment_and_empty_target(self):
        matcher = HungarianMatcher(class_cost=2.0, node_cost=5.0)
        _, predictions, targets = _batch()
        source, target = matcher(predictions, targets)[0]
        self.assertEqual(source.tolist(), [0, 1, 2])
        self.assertEqual(target.tolist(), [0, 1, 2])

        empty = matcher(
            {key: value[:, :2] for key, value in predictions.items()},
            {"nodes": [torch.empty((0, 3))]},
        )[0]
        self.assertEqual(empty[0].numel(), 0)


class GraphCriterionTests(unittest.TestCase):
    def test_ratio_upsampling_backward_avoids_zero_repeat(self):
        # This needs one extra positive, fewer than the two-item source pool.
        # PyTorch 1.5 crashes in RepeatBackward if implemented as repeat(0, 1).
        logits = torch.randn((7, 2), requires_grad=True)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.long)

        expanded_logits, expanded_labels = _ratio_upsample(
            logits, labels, ratio=0.75, tolerance=0.0
        )
        expanded_logits.sum().backward()

        expected_gradient = torch.ones_like(logits)
        expected_gradient[0] = 2.0
        self.assertEqual(expanded_logits.shape, (8, 2))
        self.assertEqual(int((expanded_labels == 1).sum()), 3)
        self.assertTrue(torch.equal(logits.grad, expected_gradient))

    def _criterion(self, config):
        relation = CountingRelationHead()
        matcher_config = config["model"]["matcher"]
        matcher = HungarianMatcher(
            matcher_config["class_cost"], matcher_config["node_cost"]
        )
        return GraphCriterion(config, matcher, relation), relation

    def test_baseline_is_finite_differentiable_and_skips_extensions(self):
        config = _config()
        criterion, relation = self._criterion(config)
        tokens, predictions, targets = _batch()
        losses = criterion(tokens, predictions, targets)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["betti_h0"].detach()), 0.0)
        self.assertEqual(float(losses["betti_h1"].detach()), 0.0)
        self.assertEqual(relation.calls, 1)
        losses["total"].backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(predictions["pred_logits"].grad)
        self.assertIsNotNone(predictions["pred_nodes"].grad)
        self.assertIsNotNone(relation.linear.weight.grad)

    def test_betti_terms_reach_relation_head(self):
        config = _config()
        for name in ("betti_h0", "betti_h1"):
            config["topology"][name].update(enabled=True, log_only=False, weight=0.2)
        criterion, relation = self._criterion(config)
        tokens, predictions, targets = _batch()
        losses = criterion(tokens, predictions, targets)
        self.assertTrue(torch.isfinite(losses["betti_h0"]))
        self.assertTrue(torch.isfinite(losses["betti_h1"]))
        self.assertGreater(relation.calls, 1)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(relation.linear.weight.grad).all())

    def test_focal_hard_negative_mode_is_finite_and_differentiable(self):
        config = _config()
        config["loss"]["node"]["classification"]["name"] = "focal"
        config["loss"]["edge"]["classification"]["name"] = "focal"
        config["loss"]["edge"]["balancing"]["mode"] = "none"
        config["loss"]["edge"]["candidates"].update(
            include_unmatched=True,
            # Guarantee that the unmatched query enters the candidate pool;
            # the shared fixture intentionally predicts it as background.
            unmatched_object_threshold=0.0,
            max_active_unmatched=4,
            max_unmatched_pairs_per_graph=8,
        )
        criterion, relation = self._criterion(config)
        tokens, predictions, targets = _batch()
        losses = criterion(tokens, predictions, targets)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreaterEqual(relation.calls, 2)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(relation.linear.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
