import unittest

from scripts.select_focal_visual_examples import select_examples


def row(**updates):
    values = {
        "source_sample_id": "patch",
        "node_fp": "0",
        "node_fn": "0",
        "edge_fp": "0",
        "edge_fn": "0",
        "predicted_nodes": "4",
        "target_nodes": "4",
        "predicted_edges": "3",
        "target_edges": "3",
        "predicted_beta1": "0",
        "target_beta1": "0",
        "beta0_absolute_error": "0",
        "beta1_absolute_error": "0",
    }
    values.update({key: str(value) for key, value in updates.items()})
    return values


class FocalVisualSelectionTests(unittest.TestCase):
    def test_selects_strict_pattern(self):
        baseline = row(node_fp=2, predicted_nodes=6, predicted_beta1=2)
        node = row()
        combined = row(edge_fn=1, predicted_edges=2)
        selected, summary = select_examples(
            {"patch": baseline}, {"patch": node}, {"patch": combined}, limit=5
        )
        self.assertEqual([entry["source_sample_id"] for entry in selected], ["patch"])
        self.assertEqual(summary["selection_mode"], "strict")
        self.assertEqual(summary["strict_candidates"], 1)

    def test_relaxed_fallback_is_labelled(self):
        baseline = row(node_fp=1, predicted_nodes=5, predicted_beta1=1)
        node = row(edge_fn=1)
        combined = row(edge_fn=1, predicted_edges=2)
        selected, summary = select_examples(
            {"patch": baseline}, {"patch": node}, {"patch": combined}, limit=1
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["selection_mode"], "relaxed_node_focal")
        self.assertEqual(summary["strict_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
