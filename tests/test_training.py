"""CPU smoke tests for optimization, training steps, and resume state."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from configs import load_config
from models.matcher import build_matcher
from training.checkpoint import load_training_checkpoint, save_training_checkpoint
from training.engine import Trainer, train_step
from training.losses import GraphCriterion
from training.optim import build_optimizer, build_scheduler


def _config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pretrain_mixed.yaml",
        environment={
            "GNBM_OUTPUT_DIR": "/outputs",
            "PLANTS_DATASET": "/plants",
            "SYNTHETIC_MRI_DATASET": "/synthetic",
        },
    )
    config = copy.deepcopy(config)
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["model"]["decoder"].update(
        hidden_dim=8, object_queries=4, relation_tokens=1
    )
    return config


class TinyGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(1, 8)
        self.class_embed = nn.Linear(8, 2)
        self.coord_embed = nn.Linear(8, 6)
        self.relation_embed = nn.Linear(24, 2)

    def forward(self, volumes):
        pooled = volumes.mean(dim=(2, 3, 4)).unsqueeze(1)
        base = self.feature(pooled)
        objects = base.expand(-1, 4, -1)
        relation = (base + 0.1).expand(-1, 1, -1)
        tokens = torch.cat((objects, relation), dim=1)
        predictions = {
            "pred_logits": self.class_embed(objects),
            "pred_nodes": self.coord_embed(objects).sigmoid(),
        }
        return tokens, predictions, base


def _batch():
    images = torch.rand((1, 1, 4, 4, 4))
    segmentations = (images > 0.5).float()
    nodes = [torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.6, 0.5]])]
    edges = [torch.tensor([[0, 1]], dtype=torch.long)]
    return [images, segmentations, nodes, edges, [0], torch.tensor([1])]


class TrainingTests(unittest.TestCase):
    def test_training_step_updates_model_and_scheduler(self):
        config = _config()
        config["training"]["warmup_epochs"] = 0
        model = TinyGraphModel()
        criterion = GraphCriterion(config, build_matcher(config), model.relation_embed)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, iterations_per_epoch=2)
        before = model.feature.weight.detach().clone()

        losses = train_step(
            model,
            criterion,
            optimizer,
            scheduler,
            _batch(),
            config,
            torch.device("cpu"),
            epoch=1,
            iteration=1,
            total_iterations=4,
        )

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertFalse(torch.equal(before, model.feature.weight.detach()))
        self.assertEqual(scheduler.last_epoch, 1)

    def test_checkpoint_round_trip_restores_all_training_state(self):
        config = _config()
        model = TinyGraphModel()
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, iterations_per_epoch=2)
        model.feature.weight.data.fill_(3.0)
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(path, model, optimizer, scheduler, 4, 17)
            restored = TinyGraphModel()
            restored_optimizer = build_optimizer(config, restored)
            restored_scheduler = build_scheduler(
                config, restored_optimizer, iterations_per_epoch=2
            )
            epoch, iteration = load_training_checkpoint(
                path, restored, restored_optimizer, restored_scheduler
            )

        self.assertEqual((epoch, iteration), (4, 17))
        self.assertTrue(torch.equal(model.feature.weight, restored.feature.weight))
        self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())

    def test_trainer_writes_interval_and_best_checkpoints(self):
        config = _config()
        config["training"]["epochs"] = 1
        config["training"]["warmup_epochs"] = 0
        config["training"]["checkpoint"]["interval_epochs"] = 1
        config["training"]["checkpoint"]["policy"] = "interval_and_best"
        config["evaluation"]["interval_epochs"] = 1
        model = TinyGraphModel()
        criterion = GraphCriterion(config, build_matcher(config), model.relation_embed)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, iterations_per_epoch=1)

        with tempfile.TemporaryDirectory() as directory:
            config["experiment"]["output_dir"] = directory
            config["experiment"]["name"] = "checkpoint-test"
            Trainer(
                model,
                criterion,
                criterion,
                optimizer,
                scheduler,
                [_batch()],
                [_batch()],
                config,
                torch.device("cpu"),
            ).fit()
            models = Path(directory) / "checkpoint-test" / "models"
            self.assertTrue((models / "checkpoint_epoch=1.pt").is_file())
            self.assertTrue((models / "best_checkpoint.pt").is_file())

    def test_best_only_keeps_exactly_one_checkpoint(self):
        config = _config()
        config["training"]["epochs"] = 2
        config["training"]["warmup_epochs"] = 0
        config["training"]["checkpoint"]["policy"] = "best_only"
        config["training"]["checkpoint"]["interval_epochs"] = 1
        config["evaluation"]["interval_epochs"] = 1
        model = TinyGraphModel()
        criterion = GraphCriterion(config, build_matcher(config), model.relation_embed)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, iterations_per_epoch=1)

        with tempfile.TemporaryDirectory() as directory:
            config["experiment"]["output_dir"] = directory
            config["experiment"]["name"] = "best-only-test"
            Trainer(
                model,
                criterion,
                criterion,
                optimizer,
                scheduler,
                [_batch()],
                [_batch()],
                config,
                torch.device("cpu"),
            ).fit()
            checkpoints = list(
                (Path(directory) / "best-only-test" / "models").glob("*.pt")
            )

        self.assertEqual([path.name for path in checkpoints], ["best_checkpoint.pt"])

    def test_none_policy_writes_no_checkpoint(self):
        config = _config()
        config["training"]["epochs"] = 1
        config["training"]["warmup_epochs"] = 0
        config["training"]["checkpoint"]["policy"] = "none"
        config["evaluation"]["interval_epochs"] = 1
        model = TinyGraphModel()
        criterion = GraphCriterion(config, build_matcher(config), model.relation_embed)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, iterations_per_epoch=1)

        with tempfile.TemporaryDirectory() as directory:
            config["experiment"]["output_dir"] = directory
            config["experiment"]["name"] = "no-checkpoint-test"
            Trainer(
                model,
                criterion,
                criterion,
                optimizer,
                scheduler,
                [_batch()],
                [_batch()],
                config,
                torch.device("cpu"),
            ).fit()
            models = Path(directory) / "no-checkpoint-test" / "models"
            checkpoints = list(models.glob("*.pt")) if models.exists() else []

        self.assertEqual(checkpoints, [])


if __name__ == "__main__":
    unittest.main()
