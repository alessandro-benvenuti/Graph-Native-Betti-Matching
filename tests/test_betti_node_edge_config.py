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


if __name__ == "__main__":
    unittest.main()
