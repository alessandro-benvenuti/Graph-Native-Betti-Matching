import unittest

import numpy as np

from scripts.audit_synthetic_mri_grid import CenterlineEdge, SourceGraph
from scripts.diagnose_synthetic_mri_boundaries import (
    clip_polyline_to_box,
    clip_segment_to_box,
    inherited_edge_crop,
    orient_edge_polyline,
)


class BoundaryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.bounds = np.asarray(((0.0, 10.0),) * 3)

    def test_segment_is_clipped_at_exact_box_intersections(self):
        clipped = clip_segment_to_box(
            np.asarray((-2.0, 2.0, 2.0)),
            np.asarray((12.0, 2.0, 2.0)),
            self.bounds,
        )
        self.assertIsNotNone(clipped)
        np.testing.assert_allclose(clipped[0], (0.0, 2.0, 2.0))
        np.testing.assert_allclose(clipped[1], (10.0, 2.0, 2.0))

    def test_reversed_samples_expose_inherited_missing_boundary(self):
        edge = CenterlineEdge(
            1,
            2,
            tuple(
                np.asarray(point, dtype=float)
                for point in ((12, 2, 2), (8, 2, 2), (2, 2, 2))
            ),
        )
        graph = SourceGraph(
            {1: np.asarray((2.0, 2.0, 2.0)), 2: np.asarray((20.0, 2.0, 2.0))},
            [(1, 2)],
            [edge],
        )
        points, orientation, _, _ = orient_edge_polyline(edge, graph.nodes)
        self.assertEqual(orientation, "reversed")
        self.assertEqual(len(clip_polyline_to_box(points, self.bounds)), 1)
        current_count, current_boundary = inherited_edge_crop(graph, edge, self.bounds)
        self.assertEqual(current_count, 0)
        self.assertEqual(current_boundary, [])

    def test_sparse_outside_samples_expose_missed_crossing(self):
        edge = CenterlineEdge(
            1,
            2,
            (np.asarray((-2.0, 2.0, 2.0)), np.asarray((12.0, 2.0, 2.0))),
        )
        graph = SourceGraph(
            {1: np.asarray((-2.0, 2.0, 2.0)), 2: np.asarray((12.0, 2.0, 2.0))},
            [(1, 2)],
            [edge],
        )
        points, _, _, _ = orient_edge_polyline(edge, graph.nodes)
        self.assertEqual(len(clip_polyline_to_box(points, self.bounds)), 1)
        self.assertEqual(inherited_edge_crop(graph, edge, self.bounds)[0], 0)

    def test_empty_samples_are_repaired_by_true_endpoints(self):
        edge = CenterlineEdge(1, 2, ())
        nodes = {
            1: np.asarray((-2.0, 2.0, 2.0)),
            2: np.asarray((12.0, 2.0, 2.0)),
        }
        points, orientation, _, _ = orient_edge_polyline(edge, nodes)
        self.assertEqual(orientation, "empty")
        components = clip_polyline_to_box(points, self.bounds)
        self.assertEqual(len(components), 1)
        np.testing.assert_allclose(components[0][0], (0.0, 2.0, 2.0))
        np.testing.assert_allclose(components[0][-1], (10.0, 2.0, 2.0))

    def test_tangent_contact_is_one_zero_dimensional_component(self):
        points = np.asarray(
            ((-1.0, -1.0, 5.0), (0.0, 0.0, 5.0), (1.0, -1.0, 5.0))
        )
        components = clip_polyline_to_box(points, self.bounds)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].shape, (1, 3))
        np.testing.assert_array_equal(components[0][0], (0.0, 0.0, 5.0))


if __name__ == "__main__":
    unittest.main()
