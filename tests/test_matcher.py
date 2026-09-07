"""Contracts for structure-aware node matching."""

from pathlib import Path
import unittest

import numpy as np
import torch

from configs import load_config
from models.matcher import (
    FusedGromovWassersteinMatcher,
    HungarianMatcher,
    build_matcher,
    fgw_objective_terms,
)


class FGWMatcherTests(unittest.TestCase):
    def setUp(self):
        self.matcher = FusedGromovWassersteinMatcher(
            class_cost=2.0,
            node_cost=5.0,
            structure_weight=0.5,
            max_iter=50,
        )

    @staticmethod
    def _candidate_fixture(target_count=3):
        outputs = {
            # Query zero is most confident but deliberately far from every GT.
            "pred_logits": torch.tensor(
                [[[0.0, 9.0], [0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.0]]]
            ),
            "pred_nodes": torch.tensor(
                [[[9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]]]
            ),
        }
        targets = {
            "nodes": [torch.arange(target_count, dtype=torch.float32)[:, None].repeat(1, 3)
                      * torch.tensor([1.0, 0.0, 0.0])]
        }
        return outputs, targets

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
        assignments, transports = matcher(
            outputs,
            targets,
            predicted_structure=[torch.zeros((3, 3))],
            candidate_indices=[torch.arange(3)],
            return_transport=True,
        )
        source, target = assignments[0]
        self.assertEqual(len(source), 2)
        self.assertEqual(len(source.unique()), 2)
        self.assertEqual(sorted(target.tolist()), [0, 1])
        self.assertEqual(transports[0].shape, (2, 3))

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
        self.assertTrue((transport.sum(axis=0) <= 0.5 + 1e-8).all())
        source, target = self.matcher.harden_transport(transport)
        self.assertEqual(len(source.unique()), 2)
        self.assertEqual(sorted(target.tolist()), [0, 1])

    def test_solver_diagnostics_report_only_available_convergence_evidence(self):
        feature_cost = np.array([[0.0, 2.0], [2.0, 0.0]])
        structure = np.array([[0.0, 1.0], [1.0, 0.0]])
        transport, diagnostics = self.matcher._solve_transport(
            feature_cost, structure, structure, return_diagnostics=True
        )
        self.matcher.validate_transport(transport, 2, 2)
        self.assertGreaterEqual(diagnostics["iterations"], 1)
        self.assertIn(
            diagnostics["termination_evidence"],
            {"tolerance_met", "iteration_limit", "not_exposed_by_pot"},
        )
        self.assertNotIn("converged", diagnostics)
        self.assertIsInstance(diagnostics["pot_objective_history"], list)

    def test_zero_structure_weight_reproduces_hungarian(self):
        outputs = {
            "pred_logits": torch.zeros((1, 3, 2)),
            "pred_nodes": torch.tensor(
                [[[0.04, 0.0, 0.0], [0.20, 0.0, 0.0], [0.90, 0.0, 0.0]]]
            ),
        }
        targets = {
            "nodes": [
                torch.tensor([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
            ],
            "edges": [torch.tensor([[0, 1]])],
        }
        hungarian = HungarianMatcher(class_cost=0.0, node_cost=1.0)
        partial_fgw = FusedGromovWassersteinMatcher(
            class_cost=0.0,
            node_cost=1.0,
            structure_weight=0.0,
            candidate_count=3,
        )
        expected = hungarian(outputs, targets)[0]
        observed = partial_fgw(
            outputs,
            targets,
            predicted_structure=[torch.zeros((3, 3))],
            candidate_indices=[torch.arange(3)],
        )[0]
        expected_by_target = dict(zip(expected[1].tolist(), expected[0].tolist()))
        observed_by_target = dict(zip(observed[1].tolist(), observed[0].tolist()))
        self.assertEqual(expected_by_target, observed_by_target)
        feature = partial_fgw._feature_cost(outputs, targets["nodes"][0], 0)
        expected_cost = sum(feature[target, source] for target, source in expected_by_target.items())
        observed_cost = sum(feature[target, source] for target, source in observed_by_target.items())
        self.assertAlmostEqual(float(expected_cost), float(observed_cost))

    def test_zero_weight_tied_optimum_requires_cost_not_mapping_equality(self):
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0, node_cost=1.0, structure_weight=0.0,
            candidate_count=2,
        )
        outputs = {
            "pred_logits": torch.zeros((1, 2, 2)),
            "pred_nodes": torch.zeros((1, 2, 3)),
        }
        targets = {
            "nodes": [torch.zeros((2, 3))],
            "edges": [torch.empty((0, 2), dtype=torch.long)],
        }
        # This alternate permutation is a valid optimum because all unary costs tie.
        matcher._solve_transport = lambda *args: np.array([[0.0, 0.5], [0.5, 0.0]])
        observed = matcher(
            outputs,
            targets,
            predicted_structure=[torch.zeros((2, 2))],
            candidate_indices=[torch.arange(2)],
        )[0]
        observed_by_target = dict(zip(observed[1].tolist(), observed[0].tolist()))
        self.assertEqual(observed_by_target, {0: 1, 1: 0})
        feature = matcher._feature_cost(outputs, targets["nodes"][0], 0)
        self.assertAlmostEqual(
            sum(float(feature[target, source]) for target, source in observed_by_target.items()),
            0.0,
        )

    def test_structure_can_improve_correspondence_at_suitable_alpha(self):
        target_structure = np.array(
            [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )
        predicted_structure = target_structure[np.ix_([1, 0, 2, 3], [1, 0, 2, 3])]
        feature = np.full((4, 4), 0.15, dtype=np.float64)
        np.fill_diagonal(feature, 0.0)
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0, node_cost=1.0, structure_weight=0.5,
            candidate_count=4,
        )
        initial = matcher.initial_transport(feature)
        transport = matcher._solve_transport(
            feature, target_structure, predicted_structure
        )
        source, target = matcher.harden_transport(transport)
        hard = np.zeros_like(feature)
        hard[target.numpy(), source.numpy()] = 0.25
        initial_terms = fgw_objective_terms(
            feature, target_structure, predicted_structure, initial, 0.5
        )
        hard_terms = fgw_objective_terms(
            feature, target_structure, predicted_structure, hard, 0.5
        )
        self.assertLess(hard_terms["structural"], initial_terms["structural"])
        self.assertLess(hard_terms["weighted_total"], initial_terms["weighted_total"])

    def test_partial_solver_uses_representable_total_mass(self):
        target_count = 15
        query_count = 32
        feature_cost = np.abs(
            np.arange(target_count)[:, None]
            - np.arange(query_count)[None, :]
        ).astype(np.float64)
        structure = np.zeros((target_count, target_count), dtype=np.float64)
        prediction_structure = np.zeros(
            (query_count, query_count), dtype=np.float64
        )
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0,
            node_cost=1.0,
            structure_weight=0.0,
            max_iter=10_000,
        )
        transport = matcher._solve_transport(
            feature_cost, structure, prediction_structure
        )
        self.assertTrue(np.allclose(transport.sum(axis=1), 1.0 / target_count))
        self.assertTrue(
            (transport.sum(axis=0) <= 1.0 / target_count + 1e-8).all()
        )

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

    def test_candidate_pool_does_not_overfill_when_unmatched_is_ranked_first(self):
        outputs, targets = self._candidate_fixture()
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0, node_cost=1.0, structure_weight=0.5,
            candidate_count=3,
        )
        candidates = matcher.matching_candidates(outputs, targets)[0]
        self.assertEqual(candidates.tolist(), [1, 2, 3])

    def test_candidate_pool_expands_to_target_count(self):
        outputs, targets = self._candidate_fixture(target_count=4)
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0, node_cost=1.0, structure_weight=0.5,
            candidate_count=2,
        )
        candidates = matcher.matching_candidates(outputs, targets)[0]
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(candidates.unique()), 4)

    def test_candidate_pool_fills_unused_slots_by_confidence(self):
        outputs, targets = self._candidate_fixture(target_count=2)
        matcher = FusedGromovWassersteinMatcher(
            class_cost=0.0, node_cost=1.0, structure_weight=0.5,
            candidate_count=4,
        )
        candidates = matcher.matching_candidates(outputs, targets)[0]
        self.assertEqual(candidates.tolist(), [1, 2, 0, 3])
        self.assertEqual(len(candidates.unique()), 4)
        self.assertTrue({1, 2}.issubset(set(candidates.tolist())))

    def test_transport_validation_accepts_feasible_fractional_plan(self):
        plan = np.array([[0.25, 0.25, 0.0], [0.25, 0.0, 0.25]])
        observed = self.matcher.validate_transport(plan, 2, 3)
        self.assertTrue(np.array_equal(observed, plan))

    def test_transport_validation_rejects_invalid_and_nonfinite_plans(self):
        invalid = (
            np.array([[0.5, 0.0], [0.5, 0.0]]),
            np.array([[0.5, 0.0], [0.0, np.nan]]),
            np.array([[0.4, 0.0], [0.0, 0.4]]),
            np.array([[0.5, 0.0], [-0.1, 0.6]]),
        )
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises(RuntimeError):
                self.matcher.validate_transport(plan, 2, 2)

    def test_objective_matches_brute_force(self):
        feature = np.array([[0.2, 0.7, 0.4], [0.5, 0.1, 0.8]])
        target = np.array([[0.0, 1.0], [1.0, 0.0]])
        prediction = np.array(
            [[0.0, 0.8, 0.3], [0.8, 0.0, 0.2], [0.3, 0.2, 0.0]]
        )
        plan = np.array([[0.25, 0.25, 0.0], [0.0, 0.25, 0.25]])
        alpha = 0.4
        brute_structure = sum(
            (target[a, b] - prediction[i, j]) ** 2
            * plan[a, i] * plan[b, j]
            for a in range(2) for b in range(2)
            for i in range(3) for j in range(3)
        )
        observed = fgw_objective_terms(feature, target, prediction, plan, alpha)
        self.assertAlmostEqual(observed["feature"], float((feature * plan).sum()))
        self.assertAlmostEqual(observed["structural"], brute_structure)
        self.assertAlmostEqual(
            observed["weighted_total"],
            (1.0 - alpha) * observed["feature"] + alpha * brute_structure,
        )

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
