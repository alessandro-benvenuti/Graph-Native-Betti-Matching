from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np

from scripts.generate_synthetic_mri_dataset import (
    PatientFeatures,
    balanced_patient_split,
    endpoint_axis_starts,
    endpoint_grid_positions,
    main,
    split_sizes,
)


class EndpointGridTests(unittest.TestCase):
    def test_endpoint_starts_average_the_remainder(self) -> None:
        self.assertEqual(
            endpoint_axis_starts(304, 54, 40),
            [0, 36, 71, 107, 143, 179, 214, 250],
        )
        starts = endpoint_axis_starts(325, 54, 40)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 271)
        self.assertLessEqual(max(np.diff(starts)), 40)

    def test_typical_volume_still_has_960_candidates(self) -> None:
        positions = endpoint_grid_positions(
            (325, 304, 600), (54, 54, 54), (40, 40, 40)
        )
        self.assertEqual(len(positions), 960)
        self.assertEqual(positions[0], (0, 0, 0))
        self.assertEqual(positions[-1], (271, 250, 546))


class BalancedSplitTests(unittest.TestCase):
    def test_split_is_exact_and_deterministic(self) -> None:
        features = [
            PatientFeatures(
                patient_id=str(index),
                foreground_fraction=0.01 + index * 0.0001,
                node_count=3800 + index,
                edge_count=3900 + 2 * index,
                bifurcation_count=100 + index % 7,
                betti_0=1,
                betti_1=100 + index % 11,
            )
            for index in range(1, 137)
        ]
        first, first_score = balanced_patient_split(features, seed=42, trials=100)
        second, second_score = balanced_patient_split(features, seed=42, trials=100)
        self.assertEqual(first, second)
        self.assertEqual(first_score, second_score)
        self.assertEqual(
            {split: list(first.values()).count(split) for split in ("train", "val", "test")},
            split_sizes(136),
        )
        self.assertEqual(set(first), {str(index) for index in range(1, 137)})


class GeneratorIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_patient(root: Path, patient_id: int) -> None:
        graph = root / "graphs" / str(patient_id)
        graph.mkdir(parents=True)
        raw = (np.arange(64, dtype=np.int16).reshape(4, 4, 4) + patient_id)
        segmentation = np.ones((4, 4, 4), dtype=np.uint8)
        nib.save(nib.Nifti1Image(raw, np.eye(4)), root / "raw" / f"{patient_id}.nii.gz")
        nib.save(
            nib.Nifti1Image(segmentation, np.eye(4)),
            root / "seg" / f"{patient_id}.nii.gz",
        )
        (graph / "nodes.csv").write_text(
            "id;pos_x;pos_y;pos_z\n"
            "1;1;1;1\n"
            "2;2;2;2\n"
        )
        (graph / "edges.csv").write_text("id;node1id;node2id\n1;1;2\n")
        (graph / "graph.vvg").write_text(
            json.dumps(
                {
                    "graph": {
                        "edges": [
                            {
                                "id": 1,
                                "node1": 1,
                                "node2": 2,
                                "skeletonVoxels": [
                                    {"pos": [1, 1, 1]},
                                    {"pos": [2, 2, 2]},
                                ],
                            }
                        ]
                    }
                }
            )
        )

    def test_generation_writes_compatible_triplets_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "raw").mkdir(parents=True)
            (root / "seg").mkdir()
            (root / "graphs").mkdir()
            for patient_id in range(1, 8):
                self._write_patient(root, patient_id)

            common = [
                "--root",
                str(root),
                "--patch-size",
                "4",
                "4",
                "4",
                "--pad",
                "0",
                "0",
                "0",
                "--maximum-stride",
                "4",
                "4",
                "4",
                "--split-trials",
                "20",
            ]
            self.assertEqual(main(common), 0)

            split_path = root / "new_split.csv"
            output = root / "new_patches"
            with split_path.open(newline="") as handle:
                split_rows = list(csv.DictReader(handle))
            self.assertEqual(len(split_rows), 7)
            self.assertEqual(
                {split: sum(row["split"] == split for row in split_rows) for split in ("train", "val", "test")},
                {"train": 4, "val": 1, "test": 2},
            )

            raw_paths = sorted(output.glob("*/raw/*_data.nii.gz"))
            seg_paths = sorted(output.glob("*/seg/*_seg.nii.gz"))
            graph_paths = sorted(output.glob("*/vtp/*_graph.vtp"))
            self.assertEqual((len(raw_paths), len(seg_paths), len(graph_paths)), (7, 7, 7))
            self.assertEqual(nib.load(raw_paths[0]).shape, (4, 4, 4))
            self.assertEqual(nib.load(seg_paths[0]).shape, (4, 4, 4))

            vtk = ET.parse(graph_paths[0]).getroot()
            piece = vtk.find("./PolyData/Piece")
            self.assertIsNotNone(piece)
            self.assertEqual(piece.attrib["NumberOfPoints"], "2")
            self.assertEqual(piece.attrib["NumberOfLines"], "1")

            with (output / "patch_index.csv").open(newline="") as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual(len(index_rows), 7)
            summary = json.loads((output / "generation_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["patches_indexed"], 7)

            self.assertEqual(main([*common, "--resume"]), 0)
            self.assertEqual(len(list(output.glob("*/raw/*_data.nii.gz"))), 7)


if __name__ == "__main__":
    unittest.main()
