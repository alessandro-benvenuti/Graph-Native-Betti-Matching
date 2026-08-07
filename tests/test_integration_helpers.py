from __future__ import annotations

import unittest

import torch

from data.augmentations import normalize_voxel_coordinates
from tests.integration_helpers import (
    node_foreground_hit_rate,
    node_foreground_neighbourhood_hit_rate,
    select_evenly_spaced,
)


class IntegrationHelperTests(unittest.TestCase):
    def test_evenly_spaced_selection_is_deterministic(self) -> None:
        self.assertEqual(select_evenly_spaced(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(select_evenly_spaced([4, 5], 8), [4, 5])
        with self.assertRaises(ValueError):
            select_evenly_spaced([1], 0)

    def test_neighbourhood_metric_distinguishes_exact_and_adjacent_nodes(self) -> None:
        segmentation = torch.zeros((1, 5, 5, 5), dtype=torch.float32)
        segmentation[0, 2, 2, 2] = 1.0
        adjacent_node = normalize_voxel_coordinates(
            torch.tensor([[2.0, 3.0, 3.0]]), (5, 5, 5)
        )
        self.assertEqual(node_foreground_hit_rate(segmentation, adjacent_node), 0.0)
        self.assertEqual(
            node_foreground_neighbourhood_hit_rate(
                segmentation, adjacent_node, radius_voxels=1
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
