"""Contracts for the controlled MRI finetuning ablation matrix."""

from pathlib import Path
import unittest

from configs import load_config


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "GNBM_OUTPUT_DIR": "/outputs",
    "PLANTS_DATASET": "/plants",
    "SYNTHETIC_MRI_DATASET": "/synthetic",
}


def _load(name):
    return load_config(
        ROOT / "configs" / "experiments" / "finetune_mri" / f"{name}.yaml",
        environment=ENVIRONMENT,
    )


class ExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = _load("baseline")
        cls.focal = _load("focal")
        cls.betti = _load("betti")
        cls.combined = _load("focal_betti")

    def test_matrix_holds_non_loss_settings_constant(self):
        for section in (
            "schema_version",
            "runtime",
            "data",
            "augmentation",
            "model",
            "training",
            "evaluation",
        ):
            expected = self.baseline[section]
            for config in (self.focal, self.betti, self.combined):
                self.assertEqual(config[section], expected, msg=section)
        seeds = {
            config["experiment"]["seed"]
            for config in (self.baseline, self.focal, self.betti, self.combined)
        }
        self.assertEqual(seeds, {364505})

    def test_each_axis_changes_only_its_intended_objective(self):
        self.assertEqual(self.baseline["loss"], self.betti["loss"])
        self.assertEqual(self.focal["loss"], self.combined["loss"])
        self.assertEqual(self.baseline["topology"], self.focal["topology"])
        self.assertEqual(self.betti["topology"], self.combined["topology"])

        self.assertEqual(
            self.baseline["loss"]["edge"]["classification"]["name"],
            "cross_entropy",
        )
        self.assertEqual(
            self.focal["loss"]["edge"]["classification"]["name"], "focal"
        )
        self.assertTrue(
            self.focal["loss"]["edge"]["candidates"]["include_unmatched"]
        )
        for name in ("betti_h0", "betti_h1"):
            topology = self.betti["topology"][name]
            self.assertTrue(topology["enabled"])
            self.assertFalse(topology["log_only"])
            self.assertEqual(topology["weight"], 0.1)
            self.assertEqual(topology["warmup_epochs"], 2)
            self.assertEqual(topology["ramp_epochs"], 5)

    def test_pilots_are_short_and_store_only_the_best_checkpoint(self):
        for config in (self.baseline, self.focal, self.betti, self.combined):
            dataset = config["data"]["datasets"]["synthetic_mri"]
            self.assertEqual(dataset["train_samples"], 256)
            self.assertEqual(dataset["validation_samples"], 32)
            self.assertEqual(config["training"]["epochs"], 10)
            self.assertEqual(
                config["training"]["checkpoint"]["policy"], "best_only"
            )
            self.assertEqual(config["evaluation"]["interval_epochs"], 1)

    def test_run_names_are_unique(self):
        names = [
            config["experiment"]["name"]
            for config in (self.baseline, self.focal, self.betti, self.combined)
        ]
        self.assertEqual(len(names), len(set(names)))

    def test_launcher_supports_exported_directories_without_git(self):
        launcher = (
            ROOT / "scripts" / "run_finetune_mri_experiment.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Refusing to launch from a dirty repository", launcher)
        self.assertIn("source-manifest.sha256", launcher)
        self.assertIn("exported directory without Git metadata", launcher)

    def test_combined_smoke_keeps_mixed_data_and_enables_all_extensions(self):
        config = load_config(
            ROOT / "configs" / "smoke_mixed_focal_betti.yaml",
            environment=ENVIRONMENT,
        )
        self.assertEqual(set(config["data"]["datasets"]), {"plants", "synthetic_mri"})
        self.assertEqual(config["runtime"]["workers"], 0)
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertEqual(
            config["loss"]["edge"]["classification"]["name"], "focal"
        )
        self.assertTrue(config["topology"]["betti_h0"]["enabled"])
        self.assertTrue(config["topology"]["betti_h1"]["enabled"])

    def test_overfit_configuration_is_tiny_fixed_and_full_strength(self):
        config = load_config(
            ROOT / "configs" / "overfit_synthetic_mri_focal_betti.yaml",
            environment=ENVIRONMENT,
        )
        self.assertEqual(set(config["data"]["datasets"]), {"synthetic_mri"})
        self.assertFalse(config["data"]["train_augmentation"])
        self.assertEqual(config["data"]["batch_size"], 2)
        self.assertEqual(
            config["data"]["datasets"]["synthetic_mri"]["train_samples"], 8
        )
        self.assertEqual(
            config["data"]["datasets"]["synthetic_mri"]["validation_samples"], 2
        )
        self.assertEqual(config["training"]["epochs"], 10)
        self.assertEqual(config["training"]["checkpoint"]["policy"], "best_only")
        self.assertEqual(
            config["loss"]["edge"]["candidates"]["unmatched_ramp_epochs"], 0
        )
        for name in ("betti_h0", "betti_h1"):
            self.assertTrue(config["topology"][name]["enabled"])
            self.assertEqual(config["topology"][name]["warmup_epochs"], 0)
            self.assertEqual(config["topology"][name]["ramp_epochs"], 0)


if __name__ == "__main__":
    unittest.main()
