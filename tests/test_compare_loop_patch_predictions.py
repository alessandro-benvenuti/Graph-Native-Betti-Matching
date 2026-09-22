"""Tests for spatial cycle-space comparison on GT-loop patches."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.loaders.common import SamplePaths
from scripts.compare_loop_patch_predictions import (
    compare_predictions,
    spatial_cycle_result,
    summarize,
    write_report,
)


NODES = np.asarray(
    [[0.1, 0.1, 0.1], [0.5, 0.1, 0.1], [0.3, 0.5, 0.1]], dtype=np.float32
)
TRIANGLE = [[0, 1], [1, 2], [0, 2]]
PATH = [[0, 1], [1, 2]]


def _prediction(edges, *, all_scores=None):
    prediction = {
        "source_sample_id": "loop_patch",
        "nodes_dhw": NODES.tolist(),
        "edges": edges,
    }
    if all_scores is not None:
        prediction["all_candidate_edges"] = TRIANGLE
        prediction["all_candidate_edge_scores"] = all_scores
    return prediction


class CompareLoopPatchPredictionsTests(unittest.TestCase):
    def test_spatial_matching_distinguishes_preserved_and_missed_loop(self):
        preserved = spatial_cycle_result(
            _prediction(TRIANGLE), NODES, TRIANGLE, max_node_distance=0.01
        )
        missed = spatial_cycle_result(
            _prediction(PATH), NODES, TRIANGLE, max_node_distance=0.01
        )
        self.assertEqual(preserved["shared_cycle_rank"], 1)
        self.assertTrue(preserved["spatial_cycle_exact"])
        self.assertEqual(missed["missed_cycle_rank"], 1)
        self.assertFalse(missed["spatial_cycle_complete"])

    def test_equal_beta1_at_different_location_is_not_spatially_correct(self):
        nodes = np.concatenate(
            (NODES, np.asarray([[0.8, 0.5, 0.1]], dtype=np.float32)), axis=0
        )
        prediction = {
            "source_sample_id": "loop_patch",
            "nodes_dhw": nodes.tolist(),
            "edges": [[0, 1], [1, 3], [0, 3]],
        }
        result = spatial_cycle_result(
            prediction, nodes, TRIANGLE, max_node_distance=0.01
        )
        self.assertTrue(result["beta1_count_exact"])
        self.assertEqual(result["shared_cycle_rank"], 0)
        self.assertEqual(result["false_cycle_rank"], 1)
        self.assertEqual(result["missed_cycle_rank"], 1)
        self.assertFalse(result["spatial_cycle_exact"])

    def test_reports_low_confidence_bottleneck_for_hard_missing_cycle(self):
        result = spatial_cycle_result(
            _prediction(PATH, all_scores=[0.9, 0.8, 0.49]),
            NODES,
            TRIANGLE,
            max_node_distance=0.01,
        )
        cycle = result["target_cycle_diagnostics"][0]
        self.assertEqual(cycle["failure_mode"], "edge_below_decision")
        self.assertAlmostEqual(cycle["weakest_edge_score"], 0.49)
        self.assertTrue(cycle["closed_above_0.25"])
        self.assertFalse(cycle["closed_above_0.5"])

    def test_reports_target_cycle_blocked_by_missing_node(self):
        prediction = {
            "source_sample_id": "loop_patch",
            "nodes_dhw": NODES[:2].tolist(),
            "edges": [[0, 1]],
            "all_candidate_edges": [[0, 1]],
            "all_candidate_edge_scores": [0.9],
        }
        result = spatial_cycle_result(
            prediction, NODES, TRIANGLE, max_node_distance=0.01
        )
        cycle = result["target_cycle_diagnostics"][0]
        self.assertEqual(cycle["failure_mode"], "missing_node")
        self.assertEqual(cycle["missing_gt_nodes"], [2])

    def test_comparison_labels_betti_fix_and_writes_lists(self):
        dataset = [
            SamplePaths(Path("raw"), Path("seg"), Path("graph"), "loop_patch")
        ]

        def graph_reader(_):
            return NODES, np.asarray(TRIANGLE, dtype=np.int64)

        rows = compare_predictions(
            [_prediction(PATH)],
            [_prediction(TRIANGLE)],
            dataset,
            graph_reader=graph_reader,
            max_node_distance=0.01,
        )
        summary = summarize(rows, max_node_distance=0.01)
        self.assertEqual(rows[0]["outcome"], "betti_fixed")
        self.assertEqual(summary["methods"]["control"]["cycle_recall"], 0.0)
        self.assertEqual(summary["methods"]["betti"]["cycle_recall"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            write_report(Path(directory), rows, summary)
            fixed = (Path(directory) / "betti_fixed_sample_ids.txt").read_text()
            csv_text = (Path(directory) / "per-patch.csv").read_text()
        self.assertEqual(fixed, "loop_patch\n")
        self.assertIn("control_shared_cycle_rank", csv_text)


if __name__ == "__main__":
    unittest.main()
