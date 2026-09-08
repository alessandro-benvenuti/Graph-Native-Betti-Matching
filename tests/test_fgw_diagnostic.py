"""Unit tests for the offline Hungarian-versus-FGW diagnostic."""

import math
import unittest

import numpy as np
import torch

from scripts.diagnose_fgw_matching import (
    assignment_metrics,
    format_console_summary,
    objective_diagnostics,
    summarize_by_target_count,
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
        self.assertAlmostEqual(metrics["gt_edge_squared_error"], 0.01)
        self.assertEqual(metrics["gt_edge_pair_count"], 1)
        self.assertEqual(metrics["gt_nonedge_pair_count"], 0)
        self.assertEqual(metrics["nonedge_metrics_undefined"], 1)
        self.assertTrue(metrics["hard_unique"])

    def test_assignment_metrics_counts_edgeless_and_tiny_undefined_cases(self):
        metrics = assignment_metrics(
            torch.zeros((1, 3)),
            torch.zeros((1, 2)),
            torch.zeros((1, 3)),
            torch.empty((0, 2), dtype=torch.long),
            torch.tensor([0]),
            torch.zeros((1, 1)),
            (torch.tensor([0]), torch.tensor([0])),
            dimensions=3,
        )
        self.assertEqual(metrics["graph_too_small_for_pair_metrics"], 1)
        self.assertEqual(metrics["edge_metrics_undefined"], 1)
        self.assertEqual(metrics["nonedge_metrics_undefined"], 1)

    def test_objective_diagnostics_reports_projection_gap(self):
        feature = np.array([[0.0, 1.0], [1.0, 0.0]])
        structure = np.array([[0.0, 1.0], [1.0, 0.0]])
        soft = np.full((2, 2), 0.25)
        observed = objective_diagnostics(
            feature,
            structure,
            structure,
            soft,
            (torch.tensor([0, 1]), torch.tensor([0, 1])),
            torch.tensor([0, 1]),
            0.5,
        )
        self.assertAlmostEqual(observed["initial_objective"], 0.0)
        self.assertGreater(observed["soft_objective_change_vs_hungarian"], 0.0)
        self.assertAlmostEqual(observed["hard_objective_change_vs_hungarian"], 0.0)
        self.assertLess(observed["hardening_objective_gap"], 0.0)

    def test_transport_metrics_detect_collisions_and_diffusion(self):
        concentrated = np.array([[0.5, 0.0], [0.5, 0.0]])
        metrics = transport_metrics(concentrated)
        self.assertEqual(metrics["soft_argmax_collisions"], 1)
        self.assertAlmostEqual(metrics["normalized_row_entropy"], 0.0)
        self.assertAlmostEqual(metrics["max_prediction_capacity_ratio"], 2.0)
        self.assertAlmostEqual(metrics["transport_total_mass"], 1.0)

        diffuse = transport_metrics(np.full((2, 2), 0.25))
        self.assertAlmostEqual(diffuse["normalized_row_entropy"], 1.0)
        self.assertAlmostEqual(diffuse["max_prediction_capacity_ratio"], 1.0)

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

    def test_summary_groups_by_graph_size_and_counts_undefined_metrics(self):
        rows = [
            {"method": "fgw", "target_count": 1, "edge_metrics_undefined": 1},
            {"method": "fgw", "target_count": 2, "edge_metrics_undefined": 0},
        ]
        grouped = summarize_by_target_count(rows)
        self.assertEqual(set(grouped), {"1", "2"})
        self.assertEqual(
            grouped["1"]["fgw"]["undefined_or_failure_counts"][
                "edge_metrics_undefined"
            ],
            1,
        )

    def test_console_summary_is_compact_and_omits_graph_size_dump(self):
        summary = {
            "dataset": "synthetic_mri",
            "split": "val",
            "methods": {
                "hungarian": {"samples": 50},
                "fgw_alpha_0": {
                    "samples": 50,
                    "changed_any": 0.0,
                    "changed_target_fraction": 0.0,
                    "coordinate_l1_mean_delta_vs_hungarian": 0.0,
                    "structural_mse_delta_vs_hungarian": 0.0,
                    "edge_nonedge_separation_delta_vs_hungarian": 0.0,
                    "soft_objective_change_vs_hungarian": 0.0,
                    "hard_objective_change_vs_hungarian": 0.0,
                    "hardening_objective_gap": 0.0,
                    "solver_iterations": 1.0,
                    "solver_seconds": 0.0005,
                    "alpha_zero_unary_cost_delta": 0.0,
                    "max_prediction_capacity_ratio": 1.0,
                    "undefined_or_failure_counts": {"invariant_failure": 0},
                },
            },
            "by_target_count": {"16": {"large": "detail"}},
        }
        report = format_console_summary(summary)
        self.assertIn("fgw_alpha_0", report)
        self.assertIn("0/50", report)
        self.assertIn("alpha-zero check", report)
        self.assertNotIn("by_target_count", report)
        self.assertLess(len(report.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
