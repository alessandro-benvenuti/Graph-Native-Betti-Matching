"""Unit tests for CPU-only detailed corruption inspection helpers."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.inspect_validation_corruptions import (
    Distribution,
    _analyse_raw_epoch,
    _cooccurrence,
    _probabilities,
    _segment_distances,
)


class DetailedCorruptionInspectionTests(unittest.TestCase):
    def test_distribution_has_exact_mean_and_fine_quantiles(self):
        distribution = Distribution(0, 1, 1000)
        distribution.add([0.0, 0.25])
        distribution.add([0.5, 1.0])
        summary = distribution.summary()
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 0.4375)
        self.assertLess(abs(summary["q50"] - 0.3755), 0.001)

    def test_probabilities_average_logits_before_softmax(self):
        forward = np.array([[0.0, 2.0]])
        reverse = np.array([[2.0, 0.0]])
        np.testing.assert_allclose(_probabilities((forward + reverse) * 0.5), [0.5])

    def test_segment_distance_uses_clipped_projection(self):
        distance, segment = _segment_distances(
            np.array([[0.5, 0.2, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0, 1]]),
        )
        np.testing.assert_allclose(distance, [0.2, 1.0])
        self.assertEqual(segment.tolist(), [0, 0])

    def test_cooccurrence_reports_exact_patterns_and_conditionals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["epoch"] + ["t0.5_" + name for name in (
                "extra_nodes", "missing_nodes", "false_edges", "missing_edges",
                "false_loop_edges", "duplicate_candidates",
                "subdivision_node_candidates", "split_node_candidates",
            )]
            import csv
            with (root / "checkpoint_comparison.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"epoch": 500, "t0.5_extra_nodes": 1, "t0.5_false_edges": 2})
                writer.writerow({"epoch": 500, "t0.5_extra_nodes": 1})
            pairs, patterns = _cooccurrence(root, 0.5)
            row = next(item for item in pairs if item["left"] == "extra_nodes" and item["right"] == "false_edges")
            self.assertEqual(row["patches_with_both"], 1)
            self.assertEqual(row["p_right_given_left"], 0.5)
            self.assertEqual(sum(item["patches"] for item in patterns), 2)

    def test_raw_epoch_finds_subdivision_and_complete_edge_groups(self):
        import torch

        def logits(probabilities):
            probabilities = np.asarray(probabilities, dtype=np.float32)
            return torch.from_numpy(
                np.stack((np.log1p(-probabilities), np.log(probabilities)), axis=-1)
            )

        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "epoch_0500"
            (folder / "raw").mkdir(parents=True)
            record = {
                "sample_id": "sample",
                "nodes": torch.tensor([
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.5, 0.01, 0.0], [0.5, 0.5, 0.5],
                ]),
                "gt_nodes": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "gt_edges": torch.tensor([[0, 1]]),
                "matched_queries": torch.tensor([0, 1]),
                "matched_gt": torch.tensor([0, 1]),
                "model_input_shape": (64, 64, 64),
                "node_logits": logits([0.99, 0.99, 0.9, 0.01]),
                "edge_logits_forward": logits([0.1, 0.9, 0.01, 0.9, 0.01, 0.01]),
                "edge_logits_reverse": logits([0.1, 0.9, 0.01, 0.9, 0.01, 0.01]),
            }
            torch.save([record], folder / "raw" / "shard_00000.pt")
            distributions, _, subdivision, patches = _analyse_raw_epoch(
                folder, pairs, threshold=0.5, tolerance=0.03
            )
            self.assertEqual(patches, 1)
            self.assertEqual(distributions[("gt_edges", "raw_probability")].count, 1)
            self.assertEqual(distributions[("matched_nonedges", "raw_probability")].count, 0)
            self.assertEqual(distributions[("unmatched_endpoint_edges", "raw_probability")].count, 5)
            self.assertEqual(len(subdivision), 1)
            self.assertEqual(subdivision[0]["query"], 2)


if __name__ == "__main__":
    unittest.main()
