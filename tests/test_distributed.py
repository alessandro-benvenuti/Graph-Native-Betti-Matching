"""CPU-safe structural tests for distributed model preparation."""

import unittest

import torch
from torch import nn

from training.distributed import prepare_model_for_distributed


class DistributedPreparationTests(unittest.TestCase):
    def test_sync_batch_norm_conversion_disables_nested_inplace_relu(self) -> None:
        model = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=1),
            nn.BatchNorm3d(2),
            nn.Sequential(nn.ReLU(inplace=True)),
        )
        state_keys = tuple(model.state_dict())

        converted = prepare_model_for_distributed(model)

        self.assertTrue(
            any(isinstance(module, nn.SyncBatchNorm) for module in converted.modules())
        )
        relus = [
            module for module in converted.modules() if isinstance(module, nn.ReLU)
        ]
        self.assertTrue(relus)
        self.assertTrue(all(not module.inplace for module in relus))
        self.assertEqual(tuple(converted.state_dict()), state_keys)


if __name__ == "__main__":
    unittest.main()
