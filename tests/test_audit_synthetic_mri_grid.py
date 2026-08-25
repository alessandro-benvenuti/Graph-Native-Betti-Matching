import unittest
import csv
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from scripts.audit_synthetic_mri_grid import (
    CenterlineEdge,
    SourceGraph,
    complete_axis_starts,
    complete_grid_positions,
    inherited_rejection_reason,
    main,
)


class GridAuditTests(unittest.TestCase):
    def test_complete_axis_starts_include_both_boundaries(self):
        self.assertEqual(complete_axis_starts(325, 54, 40), [0, 40, 80, 120, 160, 200, 240, 271])
        self.assertEqual(complete_axis_starts(304, 54, 40), [0, 40, 80, 120, 160, 200, 240, 250])
        self.assertEqual(complete_axis_starts(600, 54, 40)[-2:], [520, 546])

    def test_typical_volume_has_960_full_grid_candidates(self):
        positions = complete_grid_positions((325, 304, 600), (54, 54, 54), (40, 40, 40))
        self.assertEqual(len(positions), 960)
        self.assertEqual(positions[0], (0, 0, 0))
        self.assertEqual(positions[-1], (271, 250, 546))

    def test_graph_crop_keeps_inside_edge(self):
        graph = SourceGraph(
            nodes={
                1: np.asarray((2.0, 2.0, 2.0)),
                2: np.asarray((8.0, 8.0, 8.0)),
            },
            edges=[(1, 2)],
            centerlines=[],
        )
        cropped = graph.crop(np.asarray(((0.0, 10.0),) * 3))
        self.assertEqual(len(cropped.positions), 2)
        self.assertEqual(cropped.edge_count, 1)
        np.testing.assert_array_equal(cropped.edges, np.asarray(((0, 1),)))

    def test_graph_crop_adds_boundary_node_for_truncated_edge(self):
        graph = SourceGraph(
            nodes={
                1: np.asarray((2.0, 2.0, 2.0)),
                2: np.asarray((20.0, 2.0, 2.0)),
            },
            edges=[(1, 2)],
            centerlines=[
                CenterlineEdge(
                    1,
                    2,
                    tuple(np.asarray(item, dtype=float) for item in ((2, 2, 2), (8, 2, 2), (12, 2, 2))),
                )
            ],
        )
        cropped = graph.crop(np.asarray(((0.0, 10.0),) * 3))
        self.assertEqual(len(cropped.positions), 2)
        self.assertEqual(cropped.edge_count, 1)
        np.testing.assert_array_equal(cropped.positions[-1], np.asarray((8.0, 2.0, 2.0)))
        np.testing.assert_array_equal(cropped.edges, np.asarray(((1, 0),)))

    def test_graph_crop_adds_segment_when_both_endpoints_are_outside(self):
        graph = SourceGraph(
            nodes={
                1: np.asarray((-5.0, 2.0, 2.0)),
                2: np.asarray((20.0, 2.0, 2.0)),
            },
            edges=[(1, 2)],
            centerlines=[
                CenterlineEdge(
                    1,
                    2,
                    tuple(
                        np.asarray(item, dtype=float)
                        for item in ((-5, 2, 2), (1, 2, 2), (8, 2, 2), (12, 2, 2))
                    ),
                )
            ],
        )
        cropped = graph.crop(np.asarray(((0.0, 10.0),) * 3))
        self.assertEqual(len(cropped.positions), 2)
        self.assertEqual(cropped.edge_count, 1)
        np.testing.assert_array_equal(cropped.positions[0], np.asarray((1.0, 2.0, 2.0)))
        np.testing.assert_array_equal(cropped.positions[1], np.asarray((8.0, 2.0, 2.0)))
        np.testing.assert_array_equal(cropped.edges, np.asarray(((0, 1),)))

    def test_inherited_rejection_priority(self):
        common = dict(
            foreground_count=100,
            background_count=100,
            snr=2.0,
            foreground_ratio=0.1,
            segmentation_sum=100.0,
            node_count=3,
            coordinate_valid=True,
        )
        self.assertEqual(inherited_rejection_reason(**common), "accepted")
        self.assertEqual(
            inherited_rejection_reason(**{**common, "foreground_count": 0}),
            "no_foreground",
        )
        self.assertEqual(
            inherited_rejection_reason(**{**common, "node_count": 2}),
            "fewer_than_3_nodes",
        )

    def test_read_only_audit_writes_reports_for_empty_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            output = Path(directory) / "reports"
            for child in ("raw", "seg", "graphs/1"):
                (root / child).mkdir(parents=True, exist_ok=True)
            raw = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
            segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
            nib.save(nib.Nifti1Image(raw, np.eye(4)), root / "raw/1.nii.gz")
            nib.save(
                nib.Nifti1Image(segmentation, np.eye(4)), root / "seg/1.nii.gz"
            )
            (root / "graphs/1/nodes.csv").write_text("id;pos_x;pos_y;pos_z\n")
            (root / "graphs/1/edges.csv").write_text("id;node1id;node2id\n")
            (root / "graphs/1/graph.vvg").write_text(
                json.dumps({"graph": {"edges": []}})
            )
            splits = root / "splits.csv"
            splits.write_text("patient_id,split\n1,train\n")

            result = main(
                [
                    "--root",
                    str(root),
                    "--splits",
                    str(splits),
                    "--output-dir",
                    str(output),
                    "--patch-size",
                    "4",
                    "4",
                    "4",
                    "--pad",
                    "0",
                    "0",
                    "0",
                    "--overlap",
                    "0",
                ]
            )

            self.assertEqual(result, 0)
            with (output / "candidate_audit.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["segmentation_empty"], "True")
            self.assertEqual(rows[0]["graph_empty"], "True")
            self.assertEqual(rows[0]["under_3_nodes"], "True")
            self.assertEqual(rows[0]["old_rejection_reason"], "no_foreground")
            self.assertTrue((output / "scan_inventory.csv").is_file())
            self.assertTrue((output / "audit_summary_by_split.csv").is_file())
            self.assertTrue((output / "audit_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
