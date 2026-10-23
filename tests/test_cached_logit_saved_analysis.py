"""Post-analysis tests use handcrafted saved states, never optimization."""
from pathlib import Path
import tempfile
import unittest

from experiments.cached_logits import analyze_saved as analysis
from tests.fixtures.cached_logit_saved_analysis import make_saved_fixture



class AnalysisTests(unittest.TestCase):
    def test_nonuniform_saved_step_area(self):
        # Correct weighting for 0,1,5: not the mean of the three errors.
        self.assertEqual(analysis.area([0,1,5],[1,0,1]),2.5)
        self.assertEqual(analysis.clipped_area([0,1,5],[1,0,1],3),1.)
        self.assertEqual(analysis.wtl([-1,0,1]),dict(win=1,tie=1,loss=1))

    def test_first_perfection_is_observed_and_can_be_lost(self):
        rows = [dict(step=step,elapsed_seconds=step/2,good=good)
                for step,good in ((0,False),(5,True),(10,False),(20,True),(200,True))]
        result = analysis.first_observed(rows,lambda r:r["good"])
        self.assertEqual(result["first_observed_step"],5)
        self.assertEqual(result["first_onset_step_interval"],[1,5])
        self.assertEqual(result["first_stable_saved_step"],20)
        self.assertFalse(result["maintained_at_later_saved_steps"])
        self.assertTrue(analysis.first_observed(rows,lambda r:False)["never_observed"])

    def test_spatial_mismatch_with_assignment_perfect_tree(self):
        import torch
        record = dict(selected=torch.arange(2),pairs=torch.tensor([[0,1]]),
            assignment=(torch.arange(2),torch.arange(2)),
            node_logits=torch.tensor([[0.,2.],[0.,2.]]),
            relation_logits=torch.tensor([[[0.,2.],[0.,2.]]]),
            coordinates=torch.tensor([[0.,0.,0.],[1.,1.,1.]]),
            gt_nodes=torch.tensor([[0.,0.,0.],[.5,.5,.5]]),
            gt_edges=torch.tensor([[0,1]]))
        config = dict(evaluation=dict(node_threshold=None,edge_threshold=None),
                      diagnostic=dict(spatial_max_node_distance=.1))
        value = analysis.saved_spatial_edges(record,{},"A",config,True)
        self.assertFalse(value["spatial_graph_perfect"])
        self.assertEqual(value["spatial_missing_gt_node_ids"],[1])
        self.assertEqual(value["spatial_missed_edge_count"],1)
        self.assertEqual(value["spatial_false_edge_count"],1)

    def test_full_saved_artifact_analysis_preserves_inputs(self):
        with tempfile.TemporaryDirectory(prefix="cached-logit-analysis-test-") as directory:
            directory = Path(directory)
            root,output = directory/"input",directory/"input"/"postanalysis"
            make_saved_fixture(root)
            before = {str(p):analysis.digest(p) for source in (root/"main-opt",root/"main-cache") for p in source.rglob("*") if p.is_file()}
            result = analysis.analyze(root,output)
            after = {str(p):analysis.digest(p) for source in (root/"main-opt",root/"main-cache") for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before,after)
            self.assertEqual(result["complete"]["samples"],1)
            self.assertEqual(len(result["trajectories"]["first_perfect"]),6)
            self.assertEqual(len(result["support_and_spatial"]["individual_basis_cycles"]),6)
            self.assertTrue(all(r['first_observed_step'] == 5 for r in result['trajectories']['first_perfect']))
            self.assertTrue((output/"convergence-A.png").is_file())
            self.assertTrue((output/"spatial-endpoints.csv").is_file())
            self.assertTrue((output/"gradient-summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                analysis.analyze(root,output)
            with self.assertRaises(ValueError):
                analysis.analyze(root,root/'main-opt'/'reanalysis')
            with self.assertRaises(ValueError):
                analysis.analyze(root,root/'main-cache'/'reanalysis')

    def test_optional_trial_weight_match_is_verified_from_yaml(self):
        import torch
        import yaml
        with tempfile.TemporaryDirectory(prefix="cached-config-test-") as directory:
            cache = make_saved_fixture(Path(directory)/'input')
            path = Path(directory)/"resolved-config.yaml"
            reference = dict(cache["config"])
            reference["topology"] = {**reference["topology"],
                "betti_h0": {**reference["topology"]["betti_h0"],"weight":.03},
                "betti_h1": {**reference["topology"]["betti_h1"],"weight":.003}}
            path.write_text(yaml.safe_dump(reference))
            result = analysis.config_comparison(cache,{"trial_0002":path})
            self.assertFalse(result["trial_0002_verified_exact_h0_h1_weight_match"])
            reference["topology"]["betti_h1"]["weight"] = .01
            path.write_text(yaml.safe_dump(reference))
            result = analysis.config_comparison(cache,{"trial_0002":path})
            self.assertTrue(result["trial_0002_verified_exact_h0_h1_weight_match"])


if __name__ == "__main__":
    unittest.main()
