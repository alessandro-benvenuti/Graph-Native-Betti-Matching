"""Contracts for the mechanistic node-focal topology-overfit experiment."""

import copy
from pathlib import Path
import unittest

from configs import ConfigError, load_config, validate_config
from scripts.select_topology_overfit_patches import select_rows
from scripts.select_h1_gradient_diagnostic_patches import (
    select_rows as select_h1_diagnostic_rows,
)
from scripts.summarize_topology_overfit import paired_outcomes
from scripts.summarize_topology_overfit_pool import summarize_pool_rows
from scripts.visualize_topology_overfit import choose_examples


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "GNBM_OUTPUT_DIR": "/outputs",
    "SYNTHETIC_MRI_DATASET": "/synthetic",
    "GNBM_OVERFIT_SAMPLE_LIST": "/selection/source_sample_ids.txt",
}


def _row(sample_id, **updates):
    row = {
        "source_sample_id": sample_id,
        "target_nodes": 4,
        "predicted_nodes": 4,
        "matched_nodes": 4,
        "matched_gt_fraction": 1.0,
        "unmatched_target_nodes": 0,
        "unmatched_predicted_nodes": 0,
        "target_edges": 3,
        "predicted_edges": 3,
        "missing_gt_edges": 0,
        "false_positive_edges": 0,
        "target_beta0": 1,
        "predicted_beta0": 1,
        "target_beta1": 0,
        "predicted_beta1": 0,
        "severity": 0.0,
    }
    row.update(updates)
    return row


class TopologyOverfitExperimentTests(unittest.TestCase):
    def test_h1_diagnostic_selection_excludes_optimized_from_transfer_cohort(self):
        pool = [
            {"source_sample_id": "selected", "target_beta1": 1},
            {"source_sample_id": "loop_a", "target_beta1": 1},
            {"source_sample_id": "loop_b", "target_beta1": 2},
            {"source_sample_id": "tree", "target_beta1": 0},
        ]
        selection = [
            {
                "source_sample_id": "selected",
                "selection_category": "genuine_loop_edge_break",
                "target_beta1": 1,
            }
        ]
        rows = select_h1_diagnostic_rows(
            pool, selection, unselected_limit=1, seed=364505
        )
        self.assertEqual(rows[0]["source_sample_id"], "selected")
        self.assertEqual(rows[0]["cohort"], "selected_genuine_loop_edge_break")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[1]["source_sample_id"], "selected")
        self.assertEqual(rows[1]["cohort"], "unselected_target_loop")

    def test_paired_outcomes_respect_metric_direction_and_ties(self):
        rows = []
        for beta_delta, edge_f1_delta in [(-1.0, 0.1), (0.0, 0.0), (2.0, -0.2)]:
            row = {}
            for metric in (
                "node_count_absolute_error",
                "edge_count_absolute_error",
                "beta0_absolute_error",
                "beta1_absolute_error",
                "node_f1",
                "edge_f1",
            ):
                row[f"betti_minus_control_{metric}"] = 0.0
            row["betti_minus_control_beta1_absolute_error"] = beta_delta
            row["betti_minus_control_edge_f1"] = edge_f1_delta
            rows.append(row)
        outcomes = paired_outcomes(rows)
        self.assertEqual(
            (
                outcomes["beta1_absolute_error"]["betti_better"],
                outcomes["beta1_absolute_error"]["tie"],
                outcomes["beta1_absolute_error"]["betti_worse"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(outcomes["edge_f1"]["betti_better"], 1)

    def test_pool_summary_separates_optimized_and_unselected_patches(self):
        rows = []
        for selected, beta1 in [(True, 1), (False, 1), (False, 0)]:
            row = {
                "selected_for_optimization": selected,
                "target_beta1": beta1,
            }
            for method in ("before", "control", "betti"):
                for metric in (
                    "node_count_absolute_error",
                    "edge_count_absolute_error",
                    "beta0_absolute_error",
                    "beta1_absolute_error",
                    "node_f1",
                    "edge_f1",
                ):
                    row[f"{method}_{metric}"] = 0.0
            for metric in (
                "node_count_absolute_error",
                "edge_count_absolute_error",
                "beta0_absolute_error",
                "beta1_absolute_error",
                "node_f1",
                "edge_f1",
            ):
                row[f"betti_minus_control_{metric}"] = 0.0
            rows.append(row)
        cohorts = summarize_pool_rows(rows)["cohorts"]
        self.assertEqual(cohorts["all_pool"]["patches"], 3)
        self.assertEqual(cohorts["selected_for_optimization"]["patches"], 1)
        self.assertEqual(cohorts["unselected_pool"]["patches"], 2)
        self.assertEqual(cohorts["unselected_target_loop"]["patches"], 1)
        self.assertEqual(cohorts["unselected_target_nonloop"]["patches"], 1)

    def test_launcher_uses_current_recipe_not_historical_resolved_config(self):
        launcher = (
            ROOT / "cluster/jean_zay/submit_topology_overfit_node_focal.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "configs/experiments/full_dataset_node_focal/finetune.yaml",
            launcher,
        )
        self.assertNotIn("$source_run/resolved-config.yaml", launcher)

    def test_paired_configs_only_change_name_and_topology(self):
        root = ROOT / "configs/experiments/topology_overfit_node_focal"
        control = load_config(root / "control.yaml", environment=ENVIRONMENT)
        betti = load_config(root / "betti.yaml", environment=ENVIRONMENT)
        self.assertEqual(
            control["loss"]["node"]["classification"]["name"], "focal"
        )
        self.assertEqual(
            control["loss"]["edge"]["classification"]["name"], "cross_entropy"
        )
        self.assertFalse(control["data"]["train_augmentation"])
        self.assertEqual(control["training"]["epochs"], 100)
        normalized = copy.deepcopy(control)
        normalized["experiment"]["name"] = betti["experiment"]["name"]
        normalized["topology"] = copy.deepcopy(betti["topology"])
        self.assertEqual(normalized, betti)

    def test_explicit_sample_file_requires_uncapped_training_data(self):
        config = load_config(
            ROOT / "configs/experiments/topology_overfit_node_focal/control.yaml",
            environment=ENVIRONMENT,
        )
        config["data"]["datasets"]["synthetic_mri"]["train_samples"] = 25
        with self.assertRaisesRegex(ConfigError, "train_samples must be null"):
            validate_config(config)

    def test_selection_covers_each_mechanistic_category(self):
        rows = [
            _row(
                "loop_break",
                target_edges=4,
                predicted_edges=3,
                target_beta1=1,
                predicted_beta1=0,
                missing_gt_edges=1,
                severity=4,
            ),
            _row("false_loop", predicted_beta1=1, severity=3),
            _row("broken_h0", predicted_beta0=2, severity=2),
            _row("false_node", unmatched_predicted_nodes=1, severity=1),
            _row("correct"),
        ]
        selected, summary = select_rows(rows, total=5)
        self.assertEqual(summary["selected"], 5)
        self.assertEqual(
            {row["selection_category"] for row in selected},
            {
                "genuine_loop_edge_break",
                "false_loop",
                "broken_h0",
                "false_positive_node",
                "correct_control",
            },
        )

    def test_gallery_takes_each_category_not_only_first_rows(self):
        selection = [
            {"source_sample_id": "a", "selection_category": "one"},
            {"source_sample_id": "b", "selection_category": "one"},
            {"source_sample_id": "c", "selection_category": "two"},
        ]
        chosen = choose_examples(selection, per_category=1)
        self.assertEqual(
            [row["source_sample_id"] for row in chosen], ["a", "c"]
        )


if __name__ == "__main__":
    unittest.main()
