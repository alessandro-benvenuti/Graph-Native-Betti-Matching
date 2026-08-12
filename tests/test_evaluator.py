"""End-to-end tests for dataset-level inference and export."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from training.evaluation import calibrate_batch_norm, evaluate_model


class ConstantEvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(1)
        self.relation_embed = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.relation_embed.weight.zero_()
            self.relation_embed.weight[1].fill_(1.0)

    def forward(self, volumes):
        normalized = self.batch_norm(volumes)
        batch = len(volumes)
        tokens = normalized.new_tensor(
            [[[1.0, 1.0], [1.0, 1.0]]]
        ).expand(batch, -1, -1)
        predictions = {
            "pred_logits": normalized.new_tensor(
                [[[0.0, 4.0], [0.0, 3.0]]]
            ).expand(batch, -1, -1),
            "pred_nodes": normalized.new_tensor(
                [
                    [
                        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                        [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
                    ]
                ]
            ).expand(batch, -1, -1),
        }
        return tokens, predictions, normalized


def _config():
    return {
        "training": {"input": "image"},
        "model": {"decoder": {"object_queries": 2, "relation_tokens": 0}},
        "evaluation": {"node_threshold": None, "edge_threshold": None},
    }


def _batch(value=1.0):
    volume = torch.full((1, 1, 4, 4, 4), float(value))
    nodes = [torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])]
    edges = [torch.tensor([[0, 1]])]
    return [volume, volume.clone(), nodes, edges, [None], torch.ones(1).long()]


class EvaluatorTests(unittest.TestCase):
    def test_evaluation_exports_summary_and_graphs(self):
        model = ConstantEvaluationModel()
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = evaluate_model(
                model,
                [_batch()],
                _config(),
                torch.device("cpu"),
                output_dir=Path(directory),
            )
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(len(rows), 1)
            self.assertTrue((Path(directory) / "summary.json").is_file())
            metrics_csv = Path(directory) / "per-patch-metrics.csv"
            self.assertTrue(metrics_csv.is_file())
            self.assertIn("source_sample_id", metrics_csv.read_text(encoding="utf-8"))
            self.assertIn("beta0_absolute_error", metrics_csv.read_text(encoding="utf-8"))
            predictions = json.loads(
                (Path(directory) / "predictions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(predictions[0]["edges"], [[0, 1]])

    def test_bn_calibration_updates_running_statistics_without_weights(self):
        model = ConstantEvaluationModel()
        relation_before = model.relation_embed.weight.detach().clone()
        running_before = model.batch_norm.running_mean.detach().clone()
        consumed = calibrate_batch_norm(
            model, [_batch(3.0)], _config(), torch.device("cpu"), batches=1
        )
        self.assertEqual(consumed, 1)
        self.assertFalse(torch.equal(model.batch_norm.running_mean, running_before))
        self.assertTrue(torch.equal(model.relation_embed.weight, relation_before))
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
