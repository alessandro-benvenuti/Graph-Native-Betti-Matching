"""Tests for the empirical corruption-model extraction and fits."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.build_empirical_corruption_model import (
    EVENT_TYPES,
    _fit_continuous,
    _fit_counts,
    _fit_metric,
    classify_record,
)


def _logits(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-5, 1 - 1e-5)
    return torch.from_numpy(
        np.stack((np.log1p(-probabilities), np.log(probabilities)), axis=-1)
    )


class EmpiricalCorruptionModelTests(unittest.TestCase):
    def _record(self):
        node_count = 8
        pairs = np.asarray(list(__import__("itertools").combinations(range(node_count), 2)))
        pair_index = {tuple(pair): index for index, pair in enumerate(pairs.tolist())}
        edge_probabilities = np.full(len(pairs), 0.01, dtype=np.float32)

        def edge(left, right, probability):
            edge_probabilities[pair_index[tuple(sorted((left, right)))]] = probability

        # Correct retained edge, one missing edge, strict subdivision attachments,
        # a duplicate attachment, a split attachment, and a retained-node shortcut.
        edge(1, 2, 0.95)
        edge(0, 1, 0.10)
        edge(2, 3, 0.95)
        edge(0, 4, 0.90)
        edge(1, 4, 0.90)
        edge(2, 5, 0.90)
        edge(1, 6, 0.90)
        edge(0, 2, 0.90)
        record = {
            "sample_id": "sample",
            "epoch": 500,
            "nodes": torch.tensor([
                [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.9, 0.0, 0.0],
                [0.15, 0.01, 0.0], [0.61, 0.0, 0.0], [0.59, 0.01, 0.0], [0.5, 0.5, 0.5],
            ]),
            "gt_nodes": torch.tensor([
                [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.9, 0.0, 0.0],
            ]),
            "gt_edges": torch.tensor([[0, 1], [1, 2], [2, 3]]),
            "matched_queries": torch.tensor([0, 1, 2, 3]),
            "matched_gt": torch.tensor([0, 1, 2, 3]),
            "model_input_shape": (64, 64, 64),
            "node_logits": _logits([0.90, 0.80, 0.85, 0.10, 0.75, 0.70, 0.65, 0.60]),
            "edge_logits_forward": _logits(edge_probabilities),
            "edge_logits_reverse": _logits(edge_probabilities),
        }
        return record, pairs

    def test_classification_is_exclusive_and_retains_soft_confidence(self):
        record, pairs = self._record()
        profile, events, baselines, _ = classify_record(
            record, pairs, {"patient_id": "42"}, background_nodes_per_patch=2,
            background_edges_per_category_patch=2,
        )
        node_events = [event for event in events if event["event_type"].endswith("node")]
        by_query = {event["query"]: event["event_type"] for event in node_events}
        self.assertEqual(by_query[3], "missing_node")
        self.assertEqual(by_query[4], "subdivision_node")
        self.assertEqual(by_query[5], "duplicate_node")
        self.assertEqual(by_query[6], "split_node")
        self.assertEqual(by_query[7], "generic_extra_node")
        self.assertEqual(len(by_query), 5)
        self.assertEqual(profile["count_missing_edge"], 1)
        # Edge 2-3 is explained by missing node 3 and is not double-counted.
        self.assertEqual(profile["count_loop_closing_false_edge"], 1)
        # Attachments to queries 4, 5 and 6 are attributes of node insertions.
        false_edges = [event for event in events if "false_edge" in event["event_type"]]
        self.assertEqual({(event["left_query"], event["right_query"]) for event in false_edges}, {(0, 2)})
        correct_nodes = [row for row in baselines if row["category"] == "correct_node"]
        self.assertTrue(any(abs(row["confidence"] - 0.8) < 1e-5 for row in correct_nodes))
        subdivision = next(event for event in events if event["event_type"] == "subdivision_node")
        self.assertEqual(subdivision["incident_edge_count"], 2)

    def test_candidate_fits_return_finite_heldout_metrics(self):
        rng = np.random.default_rng(7)
        train = rng.beta(3, 2, size=1000)
        test = rng.beta(3, 2, size=300)
        fits = _fit_continuous(train, test, support=(0, 1))
        self.assertEqual({fit["family"] for fit in fits}, {"scaled_beta", "scaled_logit_normal"})
        self.assertTrue(all(np.isfinite(fit["heldout_mean_log_likelihood"]) for fit in fits))
        count_fits = _fit_counts(rng.poisson(1.5, 1000), rng.poisson(1.5, 300))
        self.assertTrue(any(fit["family"] == "hurdle_poisson" for fit in count_fits))

    def test_event_taxonomy_is_stable(self):
        self.assertEqual(len(EVENT_TYPES), len(set(EVENT_TYPES)))

    def test_truncated_poisson_recovers_underlying_rate(self):
        rng = np.random.default_rng(42)
        positive = rng.poisson(0.7, 100000)
        positive = positive[positive > 0]
        train = np.concatenate((np.zeros(10000, dtype=int), positive[:25000]))
        test = np.concatenate((np.zeros(2000, dtype=int), positive[25000:30000]))
        fits = _fit_counts(train, test)
        fit = next(item for item in fits if item['family'] == 'hurdle_poisson')
        self.assertAlmostEqual(fit['parameters']['rate'], 0.7, delta=0.03)
        self.assertLess(fit['heldout_ks'], 0.03)

    def test_negative_binomial_recovers_truncated_parameters(self):
        rng = np.random.default_rng(8)
        raw = rng.negative_binomial(1.3, 0.4, 100000)
        positive = raw[raw > 0]
        train = np.concatenate((np.zeros(10000, dtype=int), positive[:20000]))
        test = np.concatenate((np.zeros(2500, dtype=int), positive[20000:25000]))
        fits = _fit_counts(train, test)
        fit = next(item for item in fits if item['family'] == 'hurdle_negative_binomial')
        self.assertAlmostEqual(fit['parameters']['n'], 1.3, delta=0.2)
        self.assertAlmostEqual(fit['parameters']['probability'], 0.4, delta=0.03)
        self.assertLess(fit['heldout_ks'], 0.04)

    def test_attachment_counts_are_discrete_and_constant_cases_work(self):
        rng = np.random.default_rng(9)
        fits = _fit_metric(rng.poisson(2, 500), rng.poisson(2, 100),
                           'duplicate_node', 'incident_edge_count', 0.5)
        self.assertTrue(all(item['family'].startswith('hurdle_') for item in fits))
        self.assertTrue(all(np.isfinite(item['heldout_ks']) for item in fits))
        zeros = _fit_counts(np.zeros(100), np.zeros(50))
        self.assertEqual(zeros[0]['family'], 'point_mass_zero')
        self.assertEqual(zeros[0]['heldout_ks'], 0)
        impossible = _fit_counts(np.zeros(100), np.ones(50))
        self.assertIsNone(impossible[0]['heldout_mean_log_likelihood'])
        with self.assertRaises(ValueError):
            _fit_counts([0.5] * 100, [1] * 50)


if __name__ == "__main__":
    unittest.main()
