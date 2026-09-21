"""Tests for GT loop-patch discovery."""

from pathlib import Path
import tempfile
import unittest

import torch

from data.loaders.common import SamplePaths
from scripts.find_gt_loop_patches import find_loop_records, write_selection


class FindGtLoopPatchesTests(unittest.TestCase):
    def test_selects_only_positive_cycle_rank_and_writes_manifest(self):
        records = [
            SamplePaths(Path("a"), Path("a"), Path("path"), "tree"),
            SamplePaths(Path("b"), Path("b"), Path("triangle"), "loop"),
        ]

        def graph_reader(path):
            nodes = torch.zeros((3, 3))
            edges = (
                torch.tensor([[0, 1], [1, 2], [0, 2]])
                if path.name == "triangle"
                else torch.tensor([[0, 1], [1, 2]])
            )
            return nodes, edges

        selected = find_loop_records(records, graph_reader=graph_reader)
        self.assertEqual([item["source_sample_id"] for item in selected], ["loop"])
        self.assertEqual(selected[0]["target_beta1"], 1)

        with tempfile.TemporaryDirectory() as directory:
            summary = write_selection(
                Path(directory), selected, total_samples=2, split="val"
            )
            sample_list = (Path(directory) / "source_sample_ids.txt").read_text()
        self.assertEqual(sample_list, "loop\n")
        self.assertEqual(summary["loop_patches"], 1)
        self.assertEqual(summary["loop_patch_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
