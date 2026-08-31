"""F1 is micro-aggregated from actual matches, not AP and AR."""

import unittest
import torch

from training.evaluation.metrics import evaluate_graph, summarize_metrics


def prediction(nodes):
    nodes = torch.tensor(nodes, dtype=torch.float32).reshape(-1, 3)
    return {
        "nodes": nodes,
        "boxes": torch.cat((nodes, torch.full_like(nodes, 0.2)), dim=1),
        "node_scores": torch.linspace(0.99, 0.51, len(nodes)),
        "edges": torch.empty(0, 2, dtype=torch.long),
        "edge_scores": torch.empty(0),
    }


class F1MetricTests(unittest.TestCase):
    def test_duplicates_and_all_predictions_beyond_ap_cap(self):
        pred = prediction([[0.5, 0.5, 0.5]] * 45)
        row = evaluate_graph(pred, [[0.5, 0.5, 0.5]], [], protocol={"max_detections": 1})
        summary = summarize_metrics([row])
        self.assertEqual(summary["node_mAP"], 1.0)
        self.assertEqual(summary["node_tp_total"], 1)
        self.assertEqual(summary["node_fp_total"], 44)
        self.assertAlmostEqual(summary["node_precision"], 1 / 45)
        self.assertEqual(summary["node_recall"], 1.0)
        self.assertAlmostEqual(summary["node_f1"], 2 / 46)

    def test_micro_not_mean_per_patch_and_empty_targets_count_false_positives(self):
        correct = evaluate_graph(prediction([[0.5, 0.5, 0.5]]), [[0.5, 0.5, 0.5]], [])
        incorrect = evaluate_graph(prediction([[0.5, 0.5, 0.5]] * 9), [], [])
        summary = summarize_metrics([correct, incorrect])
        self.assertAlmostEqual(summary["node_precision"], 0.1)
        self.assertAlmostEqual(summary["node_f1"], 2 / 11)

    def test_empty_predictions_and_empty_dataset_targets_do_not_reward_collapse(self):
        row = evaluate_graph(prediction([]), [[0.5, 0.5, 0.5]], [])
        self.assertEqual(row["node_fn"], 1)
        self.assertEqual(summarize_metrics([row])["node_f1"], 0.0)
        empty = evaluate_graph(prediction([]), [], [])
        self.assertEqual(summarize_metrics([empty])["edge_f1"], 0.0)

    def test_perfect_edge_and_wrong_edge(self):
        nodes = [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]
        pred = prediction(nodes)
        pred.update(edges=torch.tensor([[0, 1], [1, 2]]), edge_scores=torch.tensor([0.9, 0.8]))
        row = evaluate_graph(pred, nodes, [[0, 1]])
        self.assertEqual(row["edge_tp"], 1)
        self.assertEqual(row["edge_fp"], 1)
        self.assertAlmostEqual(summarize_metrics([row])["edge_f1"], 2 / 3)

    def test_iou_is_configurable_and_independent_of_ap_thresholds(self):
        pred = prediction([[0.55, 0.5, 0.5]])  # IoU approximately 0.6
        loose = evaluate_graph(pred, [[0.5, 0.5, 0.5]], [], protocol={"f1_iou_threshold": 0.5})
        strict = evaluate_graph(pred, [[0.5, 0.5, 0.5]], [], protocol={"f1_iou_threshold": 0.75})
        self.assertEqual(loose["node_tp"], 1)
        self.assertEqual(strict["node_tp"], 0)
        self.assertEqual(loose["node_mAP"], strict["node_mAP"])


if __name__ == "__main__":
    unittest.main()
