"""Contracts for the mechanistic node-focal topology-overfit experiment."""

import copy
from pathlib import Path
import unittest

from configs import ConfigError, load_config, validate_config
from scripts.select_topology_overfit_patches import select_rows
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
