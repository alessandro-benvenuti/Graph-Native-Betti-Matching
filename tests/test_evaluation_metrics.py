"""Contracts for corrected baseline graph evaluation metrics."""

import math
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.evaluation.metrics import (
    aggregate_detection_ap_ar,
    center_size_to_corners,
    detection_ap_ar,
    detection_match_state,
    evaluate_graph,
    graph_point_cloud,
    summarize_metrics,
)
from training.evaluation.visualization import (
    normalized_dhw_to_plot_xyz,
    save_graph_comparison,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_center_size_boxes_are_converted_exactly_once(self):
        corners = center_size_to_corners(
            np.asarray([[0.5, 0.4, 0.3, 0.2, 0.4, 0.6]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            corners,
            [[0.4, 0.2, 0.6, 0.6, 0.0, 0.6]],
            atol=1.0e-6,
        )

    def test_perfect_detections_have_unit_ap_and_recall(self):
        boxes = center_size_to_corners(
            np.asarray(
                [
                    [0.25, 0.25, 0.25, 0.2, 0.2, 0.2],
                    [0.75, 0.75, 0.75, 0.2, 0.2, 0.2],
                ],
                dtype=np.float32,
            )
        )
        average_precision, average_recall = detection_ap_ar(
            boxes, np.asarray([0.9, 0.8]), boxes
        )
        self.assertAlmostEqual(average_precision, 1.0, places=6)
        self.assertAlmostEqual(average_recall, 1.0, places=6)

    def test_high_confidence_false_positive_reduces_ap(self):
        target = center_size_to_corners(
            np.asarray([[0.25, 0.25, 0.25, 0.2, 0.2, 0.2]])
        )
        false = center_size_to_corners(
            np.asarray([[0.75, 0.75, 0.75, 0.2, 0.2, 0.2]])
        )
        predicted = np.concatenate((false, target), axis=0)
        average_precision, average_recall = detection_ap_ar(
            predicted, np.asarray([0.99, 0.9]), target
        )
        self.assertLess(average_precision, 1.0)
        self.assertAlmostEqual(average_recall, 1.0, places=6)

    def test_dataset_ap_pools_scores_across_images(self):
        target = center_size_to_corners(
            np.asarray([[0.25, 0.25, 0.25, 0.2, 0.2, 0.2]])
        )
        false = center_size_to_corners(
            np.asarray([[0.75, 0.75, 0.75, 0.2, 0.2, 0.2]])
        )
        states = [
            detection_match_state(target, np.asarray([0.9]), target),
            detection_match_state(false, np.asarray([0.8]), target),
        ]
        average_precision, average_recall = aggregate_detection_ap_ar(states)
        self.assertAlmostEqual(average_recall, 0.5, places=6)
        self.assertGreater(average_precision, 0.45)
        self.assertLess(average_precision, 0.55)

    def test_point_cloud_uses_the_declared_edge_endpoints(self):
        nodes = torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
        )
        cloud = graph_point_cloud(nodes, torch.tensor([[0, 2]]), num_points=5)
        self.assertIsNotNone(cloud)
        self.assertTrue(torch.allclose(cloud[:, :2], torch.zeros((5, 2))))
        self.assertTrue(torch.allclose(cloud[:, 2], torch.linspace(0, 1, 5)))

    def test_perfect_graph_has_perfect_detection_and_topology_metrics(self):
        nodes = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
        edges = torch.tensor([[0, 1]])
        prediction = {
            "nodes": nodes,
            "boxes": torch.cat((nodes, torch.full_like(nodes, 0.2)), dim=1),
            "node_scores": torch.tensor([0.9, 0.8]),
            "edges": edges,
            "edge_scores": torch.tensor([0.7]),
        }
        metrics = evaluate_graph(prediction, nodes, edges)
        for name in ("node_mAP", "node_mAR", "edge_mAP", "edge_mAR"):
            self.assertAlmostEqual(metrics[name], 1.0, places=6)
        self.assertAlmostEqual(metrics["smd"], 0.0, places=6)
        self.assertEqual(metrics["beta0_absolute_error"], 0.0)
        self.assertEqual(metrics["beta1_absolute_error"], 0.0)
        self.assertEqual(metrics["target_beta0"], 1.0)
        self.assertEqual(metrics["predicted_beta0"], 1.0)
        self.assertEqual(metrics["target_beta1"], 0.0)
        self.assertEqual(metrics["predicted_beta1"], 0.0)
        self.assertEqual(metrics["node_count_absolute_error"], 0.0)
        self.assertEqual(metrics["edge_count_absolute_error"], 0.0)

    def test_summary_includes_every_sample_when_folds_are_uneven(self):
        rows = [{"metric": float(index)} for index in range(6)]
        summary = summarize_metrics(rows, folds=5)
        self.assertEqual(summary["samples"], 6)
        self.assertAlmostEqual(summary["metric"], 2.5)
        self.assertTrue(math.isfinite(summary["metric_std"]))

    def test_visualization_mapping_respects_dhw_coordinate_order(self):
        xyz = normalized_dhw_to_plot_xyz(
            torch.tensor([[0.5, 0.25, 0.75]]), (10, 20, 40)
        )
        np.testing.assert_allclose(xyz, [[30.0, 5.0, 5.0]])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib is required for rendering",
    )
    def test_headless_plot_is_written(self):
        segmentation = torch.zeros((1, 8, 8, 8))
        segmentation[0, 2:6, 3:5, 3:5] = 1
        nodes = torch.tensor([[0.25, 0.5, 0.5], [0.625, 0.5, 0.5]])
        edges = torch.tensor([[0, 1]])
        prediction = {"nodes": nodes, "edges": edges}
        with tempfile.TemporaryDirectory() as directory:
            output = save_graph_comparison(
                segmentation,
                nodes,
                edges,
                prediction,
                Path(directory) / "comparison.png",
            )
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
