"""Focused tests for the portable SyntheticMRI graph visualizer."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.evaluation.interactive_visualization import (
    classify_visualization_errors,
    filter_prediction,
    find_patch_triplet,
    normalized_dhw_to_plot_xyz,
    resolve_prediction_record,
    validate_graph_endpoints,
)


class InteractiveVisualizationTests(unittest.TestCase):
    def test_resolves_source_and_evaluation_ids_independently(self):
        records = [
            {"sample_id": "sample_000000", "source_sample_id": "sample_patient_0001"},
            {"sample_id": "sample_000001", "source_sample_id": "sample_patient_0002"},
        ]
        self.assertEqual(
            resolve_prediction_record(records, sample_id="sample_patient_0002")["sample_id"],
            "sample_000001",
        )
        self.assertEqual(
            resolve_prediction_record(records, evaluation_id="sample_000000")["source_sample_id"],
            "sample_patient_0001",
        )
        with self.assertRaises(ValueError):
            resolve_prediction_record(records, sample_id="sample_patient_0001", evaluation_id="sample_000000")

    def test_matches_complete_patch_triplet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("raw", "seg", "vtp"):
                (root / "test" / folder).mkdir(parents=True)
            (root / "test/raw/example_data.nii.gz").touch()
            (root / "test/seg/example_seg.nii.gz").touch()
            (root / "test/vtp/example_graph.vtp").touch()
            result = find_patch_triplet(root, "test", "example")
            self.assertEqual(result.sample_id, "example")
            self.assertEqual(result.graph.name, "example_graph.vtp")

    def test_incomplete_patch_triplet_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("raw", "seg", "vtp"):
                (root / "test" / folder).mkdir(parents=True)
            (root / "test/raw/example_data.nii.gz").touch()
            (root / "test/seg/example_seg.nii.gz").touch()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete sample"):
                find_patch_triplet(root, "test", "example")

    def test_dhw_to_xyz_conversion_does_not_swap_semantics_silently(self):
        xyz = normalized_dhw_to_plot_xyz([[0.25, 0.5, 0.75]], (20, 40, 80))
        np.testing.assert_allclose(xyz, [[60.0, 20.0, 5.0]])

    def test_threshold_filtering_remaps_edge_endpoints(self):
        record = {
            "nodes_dhw": [[0, 0, 0], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
            "node_scores": [0.4, 0.9, 0.8],
            "edges": [[0, 1], [1, 2], [0, 2]],
            "edge_scores": [0.95, 0.75, 0.2],
        }
        result = filter_prediction(record, node_threshold=0.5, edge_threshold=0.7)
        np.testing.assert_array_equal(result.original_node_indices, [1, 2])
        np.testing.assert_array_equal(result.edges, [[0, 1]])
        np.testing.assert_allclose(result.edge_scores, [0.75])

    def test_graph_endpoint_validation_rejects_invalid_indices_and_self_loops(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_graph_endpoints([[0, 2]], 2)
        with self.assertRaisesRegex(ValueError, "self-loop"):
            validate_graph_endpoints([[1, 1]], 2)

    def test_visualization_error_classification_uses_mapped_endpoints(self):
        gt_nodes = [[0, 0, 0], [0, 0, 0.5], [0, 0, 1.0]]
        gt_edges = [[0, 1], [1, 2]]
        predicted_nodes = [[0.01, 0, 0], [0, 0.01, 0.5], [0.8, 0.8, 0.8]]
        predicted_edges = [[0, 1], [1, 2]]
        result = classify_visualization_errors(
            predicted_nodes, predicted_edges, gt_nodes, gt_edges, max_distance=0.05
        )
        self.assertEqual(result.predicted_to_gt, {0: 0, 1: 1})
        self.assertEqual(result.unmatched_predicted_nodes, (2,))
        self.assertEqual(result.unmatched_gt_nodes, (2,))
        self.assertEqual(result.matched_predicted_edges, ((0, 1),))
        self.assertEqual(result.incident_to_unmatched_edges, ((1, 2),))
        self.assertEqual(result.missing_gt_edges, ((1, 2),))

    def test_edge_between_matched_wrong_endpoints_is_likely_false_positive(self):
        result = classify_visualization_errors(
            [[0, 0, 0], [0, 0.5, 0], [0, 1, 0]],
            [[0, 2]],
            [[0, 0, 0], [0, 0.5, 0], [0, 1, 0]],
            [[0, 1], [1, 2]],
            max_distance=0.01,
        )
        self.assertEqual(result.false_positive_edges, ((0, 2),))
        self.assertEqual(result.missing_gt_edges, ((0, 1), (1, 2)))


if __name__ == "__main__":
    unittest.main()
