"""Test diagnostic math without importing GPU/model dependencies."""
import ast
from pathlib import Path
import numpy as np

source = Path(__file__).resolve().parents[1] / 'analyze_validation_predictions.py'
tree = ast.parse(source.read_text())
pure_functions = {
    'segment_distances',
    '_edge_matrix',
    'topology',
    'bridge_edges',
    'connected_components',
    '_quantiles',
    'diagnostics',
    'edge_event_rows',
}
functions = ast.Module(
    body=[
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in pure_functions
    ],
    type_ignores=[],
)
namespace = {'np': np}
exec(compile(functions, str(source), 'exec'), namespace)


def test_segment_projection_and_empty_graph():
    fn = namespace['segment_distances']
    d, i, t = fn(np.array([[.5, .2, 0.]]), np.array([[0., 0., 0.], [1., 0., 0.]]), np.array([[0, 1]]))
    np.testing.assert_allclose(d, [.2])
    np.testing.assert_allclose(t, [.5])
    assert i.tolist() == [0]
    d, i, t = fn(np.zeros((1, 3)), np.empty((0, 3)), np.empty((0, 2), dtype=int))
    assert np.isnan(d[0]) and i[0] == -1


def test_triangle_and_inactive_nodes():
    fn = namespace['topology']
    adjacency = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(adjacency, False)
    assert fn(np.ones(3, dtype=bool), adjacency)['cycle_rank'] == 1
    assert fn(np.array([True, True, False]), adjacency)['cycle_rank'] == 0
    assert fn(np.zeros(3, dtype=bool), adjacency)['components'] == 0


def test_bridge_detection_distinguishes_cycles():
    fn = namespace['bridge_edges']
    chain = np.array(
        [[False, True, False], [True, False, True], [False, True, False]]
    )
    assert fn(np.ones(3, dtype=bool), chain) == {(0, 1), (1, 2)}
    triangle = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(triangle, False)
    assert fn(np.ones(3, dtype=bool), triangle) == set()


def test_extra_node_has_no_coordinate_cost():
    fn = namespace['diagnostics']
    x = np.array([[0., 0., 0.], [1., 0., 0.], [50., 50., 50.]])
    args = (x, np.array([1., 1., .2]), np.array([[0, 1], [0, 2], [1, 2]]),
            np.array([1., 0., 0.]), x[:2], np.array([[0, 1]]),
            np.array([0, 1]), np.array([0, 1]), [.5], .03)
    nodes, summary, y, displacement = fn(*args)
    assert summary['coordinate_cost'] == 0
    np.testing.assert_allclose(summary['node_cost'], .2)
    assert summary['edge_raw_cost'] == 0
    assert summary['t0.5_extra_nodes'] == 0
    assert nodes[2]['coordinate_error'] is None


def test_edge_event_marks_gt_shortcut_as_loop_closing():
    fn = namespace['edge_event_rows']
    x = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    target = np.array(
        [[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]
    )
    rows = fn(
        x,
        np.ones(3),
        pairs,
        np.array([.9, .8, .9]),
        np.array([0, 1, 2]),
        np.array([[0, 1], [1, 2]]),
        target,
        {},
        .3,
    )
    shortcut = next(row for row in rows if row['left'] == 0 and row['right'] == 2)
    assert shortcut['gt_loop_closing_candidate']
