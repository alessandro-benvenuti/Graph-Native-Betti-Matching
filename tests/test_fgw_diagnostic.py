"""Unit tests for the offline Hungarian-versus-FGW diagnostic."""

import math
import unittest

import numpy as np
import torch

from scripts.diagnose_fgw_matching import (
    assignment_metrics,
    summarize_rows,
    transport_metrics,
)


class FGWDiagnosticTests(unittest.TestCase):
    def test_assignment_metrics_align_queries_to_target_order(self):
        predicted_nodes = torch.tensor(
            [[0.9, 0.9, 0.9], [0.8, 0.8, 0.8], [0.1, 0.1, 0.1]]
        )
        predicted_logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
        target_nodes = torch.tensor([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]])
        target_edges = torch.tensor([[0, 1]])
        candidates = torch.tensor([2, 1, 0])
        candidate_structure = torch.tensor(
            [[0.0, 0.9, 0.1], [0.9, 0.0, 0.2], [0.1, 0.2, 0.0]]
        )
        assignment = (torch.tensor([1, 2]), torch.tensor([1, 0]))

        metrics = assignment_metrics(
            predicted_nodes,
            predicted_logits,
            target_nodes,
            target_edges,
            candidates,
            candidate_structure,
            assignment,
            dimensions=3,
        )

        self.assertEqual(metrics["matched_query_ids"], [2, 1])
        self.assertAlmostEqual(metrics["coordinate_l1_mean"], 0.0)
        self.assertAlmostEqual(metrics["gt_edge_probability_mean"], 0.9)
        self.assertTrue(metrics["hard_unique"])

    def test_transport_metrics_detect_collisions_and_diffusion(self):
        concentrated = np.array([[0.5, 0.0], [0.5, 0.0]])
        metrics = transport_metrics(concentrated)
        self.assertEqual(metrics["soft_argmax_collisions"], 1)
        self.assertAlmostEqual(metrics["normalized_row_entropy"], 0.0)

        diffuse = transport_metrics(np.full((2, 2), 0.25))
        self.assertAlmostEqual(diffuse["normalized_row_entropy"], 1.0)

    def test_summary_ignores_nonfinite_and_nonnumeric_values(self):
        rows = [
            {"method": "fgw", "sample_id": "a", "metric": 1.0, "other": math.nan},
            {"method": "fgw", "sample_id": "b", "metric": 3.0, "other": 2.0},
            {"method": "hungarian", "sample_id": "a", "metric": 5.0},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["fgw"]["samples"], 2)
        self.assertAlmostEqual(summary["fgw"]["metric"], 2.0)
        self.assertAlmostEqual(summary["fgw"]["other"], 2.0)


if __name__ == "__main__":
    unittest.main()
