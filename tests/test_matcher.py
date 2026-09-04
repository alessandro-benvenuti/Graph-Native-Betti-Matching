"""Contracts for structure-aware node matching."""

from pathlib import Path
import unittest

import numpy as np
import torch

from configs import load_config
from models.matcher import FusedGromovWassersteinMatcher, build_matcher


class FGWMatcherTests(unittest.TestCase):
    def setUp(self):
        self.matcher = FusedGromovWassersteinMatcher(
            class_cost=2.0,
            node_cost=5.0,
            structure_weight=0.5,
            max_iter=50,
        )

    def test_target_structure_is_symmetric_and_ignores_loops(self):
        observed = self.matcher.target_structure(
            3,
            torch.tensor([[0, 1], [2, 1], [2, 2]]),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.assertTrue(torch.equal(observed, expected))

    def test_hardening_is_global_one_to_one_and_preserves_orientation(self):
        # Both rows prefer query 1 independently, so row-wise argmax would
        # duplicate it. The global projection instead selects distinct queries.
        transport = np.array([[0.10, 0.80, 0.10], [0.35, 0.60, 0.05]])
        source, target = self.matcher.harden_transport(transport)
        self.assertEqual(source.tolist(), [1, 0])
        self.assertEqual(target.tolist(), [0, 1])
        self.assertEqual(len(source.unique()), 2)
        self.assertEqual(len(target.unique()), 2)

    def test_forward_validates_transport_and_returns_all_targets(self):
        matcher = FusedGromovWassersteinMatcher(
            class_cost=1.0,
            node_cost=1.0,
            structure_weight=0.5,
        )
        matcher._solve_transport = lambda *args: np.array(
            [[0.05, 0.45, 0.0], [0.40, 0.05, 0.05]]
        )
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 2.0], [0.0, 1.0], [1.0, 0.0]]]),
            "pred_nodes": torch.tensor(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]]]
            ),
        }
        targets = {
            "nodes": [torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
            "edges": [torch.tensor([[0, 1]])],
        }
        source, target = matcher(
            outputs,
            targets,
            predicted_structure=[torch.zeros((3, 3))],
            candidate_indices=[torch.arange(3)],
        )[0]
        self.assertEqual(len(source), 2)
        self.assertEqual(len(source.unique()), 2)
        self.assertEqual(sorted(target.tolist()), [0, 1])

    def test_real_solver_returns_valid_fixed_marginal(self):
        feature_cost = np.array(
            [[0.0, 2.0, 4.0], [2.0, 0.0, 3.0]], dtype=np.float64
        )
        target_structure = np.array([[0.0, 1.0], [1.0, 0.0]])
        predicted_structure = np.array(
            [[0.0, 0.9, 0.1], [0.9, 0.0, 0.2], [0.1, 0.2, 0.0]]
        )
        transport = self.matcher._solve_transport(
            feature_cost, target_structure, predicted_structure
        )
        self.assertEqual(transport.shape, (2, 3))
        self.assertTrue(np.isfinite(transport).all())
        self.assertTrue((transport >= -1e-8).all())
        self.assertTrue(np.allclose(transport.sum(axis=1), [0.5, 0.5]))
        source, target = self.matcher.harden_transport(transport)
        self.assertEqual(len(source.unique()), 2)
        self.assertEqual(sorted(target.tolist()), [0, 1])

    def test_candidate_pool_retains_unary_matches_and_is_bounded(self):
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0,
            node_cost=1.0,
            structure_weight=0.5,
            candidate_count=3,
        )
        outputs = {
            "pred_logits": torch.tensor(
                [[[0.0, 1.0], [5.0, 0.0], [0.0, 4.0], [0.0, 3.0]]]
            ),
            "pred_nodes": torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [0.8, 0.8, 0.8],
                        [0.7, 0.7, 0.7],
                    ]
                ]
            ),
        }
        targets = {"nodes": [torch.tensor([[1.0, 1.0, 1.0]])]}
        candidates = matcher.matching_candidates(outputs, targets)[0]
        self.assertEqual(len(candidates), 3)
        self.assertIn(1, candidates.tolist())

    def test_empty_target_does_not_invoke_solver(self):
        self.matcher._solve_transport = lambda *args: self.fail("solver was called")
        assignment = self.matcher(
            {
                "pred_logits": torch.zeros((1, 2, 2)),
                "pred_nodes": torch.zeros((1, 2, 3)),
            },
            {
                "nodes": [torch.empty((0, 3))],
                "edges": [torch.empty((0, 2), dtype=torch.long)],
            },
            predicted_structure=[torch.empty((0, 0))],
            candidate_indices=[torch.empty(0, dtype=torch.long)],
        )[0]
        self.assertEqual(assignment[0].numel(), 0)

    def test_fgw_config_builds_structure_aware_matcher(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "finetune_synthetic_mri_focal_fgw.yaml",
            environment={
                "GNBM_OUTPUT_DIR": "/outputs",
                "PLANTS_DATASET": "/plants",
                "SYNTHETIC_MRI_DATASET": "/synthetic",
            },
        )
        self.assertIsInstance(build_matcher(config), FusedGromovWassersteinMatcher)


if __name__ == "__main__":
    unittest.main()
