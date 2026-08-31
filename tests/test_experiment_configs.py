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
            "tracking",
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

    def test_all_experiments_share_wandb_defaults(self):
        for config in (self.baseline, self.focal, self.betti, self.combined):
            tracking = config["tracking"]
            self.assertTrue(tracking["enabled"])
            self.assertEqual(tracking["project"], "focal-loss")
            self.assertIsNone(tracking["mode"])

    def test_seven_recipe_matrix_has_paired_pretraining_and_finetuning(self):
        paths = ROOT / "configs" / "experiments" / "focal_matrix_600"
        recipes = (
            "baseline",
            "node_immediate",
            "node_curriculum",
            "edge_immediate",
            "edge_curriculum",
            "combined_immediate",
            "combined_curriculum",
        )
        names = set()
        for recipe in recipes:
            pretrain = load_config(
                paths / f"pretrain_{recipe}.yaml", environment=ENVIRONMENT
            )
            finetune = load_config(
                paths / f"finetune_{recipe}.yaml", environment=ENVIRONMENT
            )
            self.assertEqual(set(pretrain["data"]["datasets"]), {"plants", "synthetic_mri"})
            self.assertEqual(set(finetune["data"]["datasets"]), {"synthetic_mri"})
            for dataset in pretrain["data"]["datasets"].values():
                self.assertIsNone(dataset["train_samples"])
                self.assertIsNone(dataset["validation_samples"])
            self.assertIsNone(
                finetune["data"]["datasets"]["synthetic_mri"]["train_samples"]
            )
            self.assertIsNone(
                finetune["data"]["datasets"]["synthetic_mri"]["validation_samples"]
            )
            self.assertIsNone(
                pretrain["evaluation"]["training_metrics"]["max_samples"]
            )
            self.assertIsNone(
                finetune["evaluation"]["training_metrics"]["max_samples"]
            )
            self.assertEqual(pretrain["training"]["epochs"], 100)
            self.assertEqual(finetune["training"]["epochs"], 600)
            self.assertEqual(pretrain["training"]["checkpoint"]["policy"], "best_only")
            self.assertEqual(finetune["training"]["checkpoint"]["policy"], "best_only")
            self.assertEqual(pretrain["loss"], finetune["loss"])
            self.assertEqual(pretrain["topology"], finetune["topology"])
            names.update((pretrain["experiment"]["name"], finetune["experiment"]["name"]))
        self.assertEqual(len(names), 2 * len(recipes))

    def test_focal_matrix_recipe_contract(self):
        paths = ROOT / "configs" / "experiments" / "focal_matrix_600"
        configs = {
            recipe: load_config(
                paths / f"finetune_{recipe}.yaml", environment=ENVIRONMENT
            )
            for recipe in (
                "baseline",
                "node_immediate",
                "node_curriculum",
                "edge_immediate",
                "edge_curriculum",
                "combined_immediate",
                "combined_curriculum",
            )
        }
        baseline = configs["baseline"]
        self.assertEqual(
            baseline["loss"]["node"]["classification"]["class_weights"],
            [0.25, 0.75],
        )
        self.assertEqual(
            baseline["loss"]["edge"]["balancing"]["mode"], "ratio_upsample"
        )
        self.assertFalse(
            baseline["loss"]["edge"]["candidates"]["include_unmatched"]
        )

        for recipe, config in configs.items():
            node_focal = recipe.startswith("node_") or recipe.startswith("combined_")
            edge_focal = recipe.startswith("edge_") or recipe.startswith("combined_")
            node = config["loss"]["node"]["classification"]
            edge = config["loss"]["edge"]
            self.assertEqual(node["name"], "focal" if node_focal else "weighted_cross_entropy")
            if node_focal:
                self.assertEqual(node["class_weights"], [1.0, 1.0])
            self.assertEqual(edge["classification"]["name"], "focal" if edge_focal else "cross_entropy")
            self.assertEqual(edge["balancing"]["mode"], "none" if edge_focal else "ratio_upsample")
            self.assertEqual(edge["candidates"]["include_unmatched"], edge_focal)
            if edge_focal:
                self.assertEqual(edge["classification"]["class_weights"], [1.0, 1.0])
                self.assertEqual(edge["candidates"]["unmatched_weight"], 1.0)
                self.assertEqual(edge["candidates"]["unmatched_warmup_epochs"], 0)
                self.assertEqual(edge["candidates"]["unmatched_ramp_epochs"], 0)

            curriculum = recipe.endswith("curriculum")
            if node_focal:
                self.assertEqual(node["curriculum"]["enabled"], curriculum)
                self.assertEqual(node["curriculum"]["start_percent"], 40.0)
                self.assertEqual(node["curriculum"]["end_percent"], 70.0)
            if edge_focal:
                edge_schedule = edge["classification"]["curriculum"]
                self.assertEqual(edge_schedule["enabled"], curriculum)
                self.assertEqual(edge_schedule["start_percent"], 40.0)
                self.assertEqual(edge_schedule["end_percent"], 70.0)

    def test_focal_matrix_smoke_exercises_mixed_combined_recipe(self):
        config = load_config(
            ROOT
            / "configs"
            / "experiments"
            / "focal_matrix_600"
            / "smoke_combined_immediate.yaml",
            environment=ENVIRONMENT,
        )
        self.assertEqual(set(config["data"]["datasets"]), {"plants", "synthetic_mri"})
        self.assertEqual(config["runtime"]["workers"], 0)
        self.assertEqual(config["data"]["batch_size"], 2)
        self.assertEqual(config["training"]["epochs"], 1)
        self.assertTrue(config["tracking"]["enabled"])
        self.assertEqual(config["loss"]["node"]["classification"]["name"], "focal")
        self.assertEqual(config["loss"]["edge"]["classification"]["name"], "focal")
        self.assertTrue(config["loss"]["edge"]["candidates"]["include_unmatched"])

    def test_focal_matrix_launcher_uses_wandb_run_group(self):
        launcher = (
            ROOT / "cluster" / "jean_zay" / "submit_focal_matrix_600.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("WANDB_RUN_GROUP", launcher)
        self.assertNotIn("WANDB_GROUP=", launcher)
        self.assertIn("focal-matrix-600-seed364505", launcher)

    def test_full_dataset_node_focal_uses_limited_mri_pretraining(self):
        paths = ROOT / "configs" / "experiments" / "full_dataset_node_focal"
        pretrain = load_config(paths / "pretrain.yaml", environment=ENVIRONMENT)
        finetune = load_config(paths / "finetune.yaml", environment=ENVIRONMENT)
        self.assertTrue(pretrain["data"]["mixed_sampling"]["balance_source_target"])
        self.assertEqual(
            pretrain["data"]["datasets"]["plants"]["train_samples"], 25900
        )
        self.assertEqual(
            pretrain["data"]["datasets"]["synthetic_mri"]["train_samples"],
            4000,
        )
        self.assertEqual(
            pretrain["data"]["datasets"]["synthetic_mri"]["validation_samples"],
            200,
        )
        self.assertEqual(
            pretrain["data"]["datasets"]["synthetic_mri"][
                "sample_cap_selection"
            ],
            "seeded_random",
        )
        self.assertEqual(
            pretrain["data"]["datasets"]["synthetic_mri"]["sample_cap_seed"],
            364505,
        )
        self.assertEqual(
            pretrain["evaluation"]["training_metrics"]["max_samples"], 200
        )
        self.assertIsNone(
            finetune["data"]["datasets"]["synthetic_mri"]["train_samples"]
        )
        self.assertIsNone(
            finetune["data"]["datasets"]["synthetic_mri"]["validation_samples"]
        )
        self.assertEqual(pretrain["training"]["epochs"], 50)
        self.assertEqual(finetune["training"]["epochs"], 250)
        self.assertEqual(finetune["training"]["early_stopping"]["min_epochs"], 0)
        self.assertEqual(finetune["training"]["early_stopping"]["patience_epochs"], 50)
        for config in (pretrain, finetune):
            self.assertEqual(config["loss"]["node"]["classification"]["name"], "focal")
            self.assertEqual(config["loss"]["edge"]["classification"]["name"], "cross_entropy")
            self.assertEqual(config["training"]["checkpoint"]["latest_interval_epochs"], 2)

    def test_full_dataset_comparisons_limit_only_pretraining_mri(self):
        paths = ROOT / "configs" / "experiments" / "full_dataset_comparison"
        expectations = {
            "baseline": ("weighted_cross_entropy", "cross_entropy", "ratio_upsample"),
            "nodefocal_edgefocal_mm": ("focal", "focal", "none"),
        }
        names = set()
        for recipe, expected in expectations.items():
            pretrain = load_config(
                paths / f"pretrain_{recipe}.yaml", environment=ENVIRONMENT
            )
            finetune = load_config(
                paths / f"finetune_{recipe}.yaml", environment=ENVIRONMENT
            )
            node_name, edge_name, balancing = expected
            self.assertTrue(
                pretrain["data"]["mixed_sampling"]["balance_source_target"]
            )
            self.assertEqual(
                pretrain["data"]["datasets"]["plants"]["train_samples"], 25900
            )
            self.assertEqual(
                pretrain["data"]["datasets"]["synthetic_mri"]["train_samples"],
                4000,
            )
            self.assertEqual(
                pretrain["data"]["datasets"]["synthetic_mri"][
                    "validation_samples"
                ],
                200,
            )
            self.assertEqual(
                pretrain["data"]["datasets"]["synthetic_mri"][
                    "sample_cap_selection"
                ],
                "seeded_random",
            )
            self.assertEqual(
                pretrain["data"]["datasets"]["synthetic_mri"][
                    "sample_cap_seed"
                ],
                364505,
            )
            self.assertEqual(
                pretrain["evaluation"]["training_metrics"]["max_samples"],
                200,
            )
            self.assertIsNone(
                finetune["data"]["datasets"]["synthetic_mri"]["train_samples"]
            )
            self.assertIsNone(
                finetune["data"]["datasets"]["synthetic_mri"][
                    "validation_samples"
                ]
            )
            self.assertEqual(pretrain["training"]["epochs"], 50)
            self.assertEqual(finetune["training"]["epochs"], 250)
            self.assertEqual(finetune["training"]["early_stopping"]["min_epochs"], 0)
            self.assertEqual(finetune["training"]["early_stopping"]["patience_epochs"], 50)
            for config in (pretrain, finetune):
                self.assertEqual(
                    config["loss"]["node"]["classification"]["name"], node_name
                )
                edge = config["loss"]["edge"]
                self.assertEqual(edge["classification"]["name"], edge_name)
                self.assertEqual(edge["balancing"]["mode"], balancing)
                self.assertFalse(edge["candidates"]["include_unmatched"])
                self.assertEqual(
                    config["training"]["checkpoint"]["latest_interval_epochs"],
                    2,
                )
            names.update(
                (pretrain["experiment"]["name"], finetune["experiment"]["name"])
            )
        self.assertEqual(len(names), 2 * len(expectations))

    def test_full_dataset_comparison_launcher_contract(self):
        launcher = (
            ROOT
            / "cluster"
            / "jean_zay"
            / "submit_full_dataset_comparisons.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("WANDB_RUN_GROUP", launcher)
        self.assertIn("afterok:", launcher)
        self.assertIn("GNBM_PRETRAIN_GPUS", launcher)
        self.assertIn("GNBM_FINETUNE_GPUS", launcher)
        self.assertIn("GNBM_FINETUNE_WALLTIME", launcher)
        self.assertIn("baseline", launcher)
        self.assertIn("nodefocal_edgefocal_mm", launcher)

    def test_edge_candidate_ablation_completes_missing_recipe_cells(self):
        paths = ROOT / "configs" / "experiments" / "edge_candidate_ablation_600"
        expectations = {
            "nodece_edgefocal_mm": (False, "focal", False, "none"),
            "nodefocal_edgefocal_mm": (True, "focal", False, "none"),
            "nodece_edgece_all": (False, "cross_entropy", True, "ratio_upsample"),
            "nodefocal_edgece_all": (True, "cross_entropy", True, "ratio_upsample"),
        }
        names = set()
        for recipe, expected in expectations.items():
            pretrain = load_config(
                paths / f"pretrain_{recipe}.yaml", environment=ENVIRONMENT
            )
            finetune = load_config(
                paths / f"finetune_{recipe}.yaml", environment=ENVIRONMENT
            )
            node_focal, edge_name, include_unmatched, balancing = expected
            self.assertEqual(pretrain["training"]["epochs"], 100)
            self.assertEqual(finetune["training"]["epochs"], 600)
            self.assertEqual(pretrain["loss"], finetune["loss"])
            self.assertEqual(pretrain["topology"], finetune["topology"])
            self.assertEqual(
                finetune["loss"]["node"]["classification"]["name"],
                "focal" if node_focal else "weighted_cross_entropy",
            )
            edge = finetune["loss"]["edge"]
            self.assertEqual(edge["classification"]["name"], edge_name)
            self.assertEqual(
                edge["candidates"]["include_unmatched"], include_unmatched
            )
            self.assertEqual(edge["balancing"]["mode"], balancing)
            self.assertEqual(edge["candidates"]["unmatched_weight"], 1.0)
            self.assertFalse(finetune["topology"]["betti_h0"]["enabled"])
            self.assertFalse(finetune["topology"]["betti_h1"]["enabled"])
            names.update((pretrain["experiment"]["name"], finetune["experiment"]["name"]))
        self.assertEqual(len(names), 2 * len(expectations))

    def test_edge_candidate_ablation_launcher_contract(self):
        launcher = (
            ROOT
            / "cluster"
            / "jean_zay"
            / "submit_edge_candidate_ablation_600.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("WANDB_RUN_GROUP", launcher)
        self.assertIn("edge-candidate-ablation-600-seed364505", launcher)
        for recipe in (
            "nodece_edgefocal_mm",
            "nodefocal_edgefocal_mm",
            "nodece_edgece_all",
            "nodefocal_edgece_all",
        ):
            self.assertIn(recipe, launcher)

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
