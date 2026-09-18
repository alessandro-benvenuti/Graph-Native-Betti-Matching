"""Configuration contracts for node-aware graph Betti matching."""

import copy
from pathlib import Path
import unittest

from configs import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "GNBM_OUTPUT_DIR": "/outputs",
    "PLANTS_DATASET": "/plants",
    "SYNTHETIC_MRI_DATASET": "/synthetic",
}


class NodeEdgeBettiConfigTests(unittest.TestCase):
    def test_smoke_overlay_enables_node_aware_hybrid_complex(self):
        config = load_config(
            ROOT / "configs" / "smoke_mixed_focal_betti_node_edge.yaml",
            environment=ENVIRONMENT,
        )
        complex_config = config["topology"]["complex"]
        self.assertEqual(complex_config["mode"], "node_aware")
        self.assertEqual(complex_config["aggregation"], "hybrid")
        self.assertEqual(complex_config["alpha"], 0.5)
        self.assertEqual(complex_config["max_active_unmatched"], 8)
        self.assertEqual(
            config["topology"]["betti_h0"]["normalization"], "matched_mean"
        )
        self.assertEqual(
            config["topology"]["betti_h0"]["unmatched_node_weight"], 1.0
        )
        self.assertEqual(
            config["topology"]["betti_h1"]["normalization"], "matched_mean"
        )

    def test_invalid_complex_configuration_is_rejected(self):
        config = load_config(
            ROOT / "configs" / "pretrain_mixed.yaml",
            environment=ENVIRONMENT,
        )
        for key, value in (
            ("mode", "unknown"),
            ("aggregation", "mean"),
            ("alpha", 1.1),
            ("unmatched_object_threshold", -0.1),
        ):
            invalid = copy.deepcopy(config)
            invalid["topology"]["complex"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_config(invalid)

    def test_invalid_topology_normalization_is_rejected(self):
        config = load_config(
            ROOT / "configs" / "pretrain_mixed.yaml",
            environment=ENVIRONMENT,
        )
        config["topology"]["betti_h0"]["normalization"] = "unsafe"
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_pilot_arms_differ_only_in_name_and_topology(self):
        experiment_root = (
            ROOT / "configs" / "experiments" / "node_edge_betti_pilot_4000"
        )
        control = load_config(
            experiment_root / "control.yaml", environment=ENVIRONMENT
        )
        betti = load_config(
            experiment_root / "node_aware_betti.yaml", environment=ENVIRONMENT
        )
        self.assertEqual(
            control["data"]["datasets"]["synthetic_mri"]["train_samples"],
            4000,
        )
        self.assertEqual(
            control["data"]["datasets"]["synthetic_mri"]["validation_samples"],
            500,
        )
        normalized_control = copy.deepcopy(control)
        normalized_betti = copy.deepcopy(betti)
        normalized_control["experiment"]["name"] = normalized_betti["experiment"][
            "name"
        ]
        normalized_control["topology"] = copy.deepcopy(
            normalized_betti["topology"]
        )
        self.assertEqual(normalized_control, normalized_betti)


if __name__ == "__main__":
    unittest.main()
