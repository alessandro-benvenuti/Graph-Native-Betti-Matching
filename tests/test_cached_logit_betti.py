"""Acceptance tests for the disposable cached free-logit experiment."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from configs import load_config
from experiments.cached_logits.core import (Objective, ARMS, canonical, edge_tensor,
    directed_probabilities, fingerprint, mode_view, metrics, optimize, gradient_audit)
from experiments.cached_logits.cache import (make_record, new_directory, torch_write,
    load_torch, validate_source_config)
from experiments.cached_logits.report import percentage, report
from training.losses.criterion import GraphCriterion
from training.losses.betti_h0 import h0_betti_matching_loss
from training.losses.betti_h1 import cycle_space_matching_loss
from tests.test_graph_losses import _config, CountingRelationHead, _batch


from experiments.cached_logits.synthetic import config, example


class CachedLogitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_cached_directed_probabilities_and_focal_match_production(self):
        c=config(); tokens,pred,targets=_batch()
        criterion=GraphCriterion(c,torch.nn.Identity(),CountingRelationHead())
        record=make_record(criterion,tokens[0],{k:v[0] for k,v in pred.items()},
            targets['nodes'][0],targets['edges'][0],(torch.arange(3),torch.arange(3)),
            'synthetic',torch.Generator().manual_seed(1))
        pairs=record['pairs']; objects=tokens[0,:4][record['selected']]
        expected=criterion._symmetric_edge_probabilities(objects,tokens[0,4:5],pairs)
        torch.testing.assert_close(directed_probabilities(record['relation_logits']),expected)
        view=mode_view(record,'B'); rel=view['relation_logits'].requires_grad_()
        losses,_=Objective(c)(view,record['node_logits'],rel,'B')
        mask=view['edge_supervision']
        expected=sum(criterion._classification_loss(rel[mask,i],view['edge_labels'][mask],criterion.edge_config)
            for i in range(2))/2
        torch.testing.assert_close(losses['edge_class'],expected)
        # Orientation averaging is intentionally not focal(mean logits).
        asymmetric=torch.tensor([[[0.,5.],[0.,-1.]]])
        self.assertGreater(float((directed_probabilities(asymmetric)-asymmetric.mean(1).softmax(-1)[:,1]).abs()),.1)

    def test_h0_h1_values_and_logit_gradients_match_helpers_in_both_modes(self):
        c=config(); record=example('unmatched_node'); objective=Objective(c)
        for mode in ('A','B'):
            view=mode_view(record,mode)
            nodes=record['node_logits'].clone().requires_grad_()
            relations=view['relation_logits'].clone().requires_grad_()
            losses,_=objective(view,nodes,relations,mode)
            p,q,presence=objective.filtration(view,nodes,relations,mode)
            for name,fn in (('h0',h0_betti_matching_loss),('h1',cycle_space_matching_loss)):
                spec=c['topology']['betti_'+name]
                kwargs=dict(num_vertices=len(q),diagonal_factor=spec['diagonal_factor'],
                    normalize=spec['normalize'],normalization=spec['normalization'])
                if name=='h0': kwargs.update(node_probabilities=q,target_node_presence=presence,unmatched_weight=spec['unmatched_weight'])
                else: kwargs.update(false_positive_weight=.25,false_negative_weight=1.)
                expected,_=fn(p,view['pairs'],view['truth'],**kwargs)
                torch.testing.assert_close(losses[name],expected)
                observed=torch.autograd.grad(losses[name],(nodes,relations),retain_graph=True,allow_unused=True)
                wanted=torch.autograd.grad(expected,(nodes,relations),retain_graph=True,allow_unused=True)
                for left,right in zip(observed,wanted):
                    if left is None: self.assertIsNone(right)
                    else: torch.testing.assert_close(left,right)
            if mode=='A': self.assertTrue(torch.equal(q,torch.ones_like(q)))

    def test_clamp_and_detach_preserve_node_but_remove_unmatched_edge_gradients(self):
        c=config(); r=example('unmatched_node'); view=mode_view(r,'B')
        nodes=r['node_logits'].clone().requires_grad_(); rel=view['relation_logits'].clone().requires_grad_()
        p,q,_=Objective(c).filtration(view,nodes,rel,'B')
        p.sum().backward()
        mm=(view['pairs']<3).all(1)
        self.assertTrue(torch.equal(nodes.grad[:3],torch.zeros_like(nodes.grad[:3])))
        self.assertGreater(float(nodes.grad[3].abs().sum()),0)
        self.assertEqual(float(rel.grad[~mm].abs().sum()),0)
        self.assertGreater(float(rel.grad[mm].abs().sum()),0)
        self.assertTrue(torch.equal(q[:3],torch.ones(3)))
        c['topology']['complex']['detach_unmatched_edge_probabilities']=False
        rel2=rel.detach().clone().requires_grad_()
        Objective(c).filtration(view,nodes.detach(),rel2,'B')[0].sum().backward()
        self.assertGreater(float(rel2.grad[~mm].abs().sum()),0)

    def test_classification_solves_all_simple_graphs(self):
        for name in ('missing_connection','false_triangle','broken_loop','unmatched_node','correct'):
            r=example(name); original=fingerprint(r)
            for mode in ('A','B'):
                state=optimize(r,config(),mode,'classification',steps=200,spatial=False)
                initial,final=state['history'][0],state['history'][-1]
                self.assertLess(final['losses']['classification'],initial['losses']['classification'])
                self.assertAlmostEqual(final['metrics']['edge_f1'],1.)
                self.assertEqual(final['metrics']['beta1_error'],0)
                self.assertEqual(final['metrics']['false_cycle_rank'],0)
                self.assertEqual(final['metrics']['missed_cycle_rank'],0)
                if mode=='B': self.assertEqual(final['metrics']['node_f1'],1.)
            self.assertEqual(fingerprint(r),original)

    def test_false_cycle_critical_gt_edge_is_suppressed_honestly(self):
        r=example('false_triangle'); v=mode_view(r,'A'); c=config()
        rel=v['relation_logits'].clone().requires_grad_(); nodes=r['node_logits']
        losses,matching=Objective(c)(v,nodes,rel,'A')
        audit=gradient_audit(v,nodes,rel,'A',losses,matching,dict(classification=1.,h0=.03,h1=.01))
        critical=[e for e in audit['critical_edges']['h1'] if e.get('false_class')]
        self.assertTrue(any(e['gt_correct'] for e in critical))
        self.assertGreater(audit['h1']['edges']['matched_matched/gt_correct/suppression']['absolute_logit_gradient_mass'],0)
        self.assertGreater(audit['conflicts']['h1']['directed_count'],0)

    def test_duplicates_and_source_immutability(self):
        self.assertEqual(canonical([[1,0],[0,1]]),[(0,1)])
        with self.assertRaisesRegex(ValueError,'duplicate'): canonical([[1,0],[0,1]],reject_duplicates=True)
        r=example(); r['pairs']=torch.cat((r['pairs'],r['pairs'][:1].flip(1)))
        with self.assertRaisesRegex(ValueError,'duplicate'): mode_view(r,'B')

    def test_empty_and_isolated_nodes_finite(self):
        for n in (0,1,3):
            r=example()
            r.update(coordinates=torch.zeros(n,3),node_logits=torch.zeros(n,2),gt_nodes=torch.zeros(n,3),
                gt_edges=edge_tensor([]),assignment=(torch.arange(n),torch.arange(n)),selected=torch.arange(n),
                unmatched_queries=torch.empty(0,dtype=torch.long),pairs=torch.combinations(torch.arange(n),r=2),truth=edge_tensor([]))
            r.update(relation_logits=torch.zeros(len(r['pairs']),2,2),edge_labels=torch.zeros(len(r['pairs']),dtype=torch.long),
                edge_supervision=torch.ones(len(r['pairs']),dtype=torch.bool),node_labels=torch.ones(n,dtype=torch.long))
            for mode in ('A','B'):
                state=optimize(r,config(),mode,'h0_h1',steps=1,spatial=False)
                self.assertTrue(state['complete'])

    def test_every_arm_initialization_and_frozen_parts(self):
        r=example('unmatched_node'); original=fingerprint(r)
        for mode in ('A','B'):
            starts=[]
            initial_tensors=[]
            for arm in ARMS:
                zero_state=optimize(r,config(),mode,arm,steps=1,max_seconds=0,spatial=False)
                initial_tensors.append((zero_state['nodes'],zero_state['relations']))
                state=optimize(r,config(),mode,arm,steps=1,spatial=False)
                starts.append(state['history'][0])
                if mode=='A': torch.testing.assert_close(state['nodes'],r['node_logits'],rtol=0,atol=0)
            for tensors in initial_tensors[1:]:
                for observed,expected in zip(tensors,initial_tensors[0]):
                    torch.testing.assert_close(observed,expected,rtol=0,atol=0)
            for key in ('metrics','losses'):
                self.assertEqual(starts[0][key],starts[1][key]); self.assertEqual(starts[1][key],starts[2][key])
        self.assertEqual(fingerprint(r),original)

    def test_save_resume_matches_uninterrupted(self):
        r=example('unmatched_node'); c=config()
        full=optimize(r,c,'B','h0_h1',steps=20,spatial=False)
        saved=[]
        optimize(r,c,'B','h0_h1',steps=20,save=saved.append,spatial=False)
        partial=next(s for s in saved if s['step']==5)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'; torch_write(path,partial)
            resumed=optimize(r,c,'B','h0_h1',steps=20,state=load_torch(path),spatial=False)
        for key in ('nodes','relations'):
            torch.testing.assert_close(resumed[key],full[key],rtol=0,atol=1e-7)
        self.assertEqual(resumed['history'][-1]['metrics'],full['history'][-1]['metrics'])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            optimize(r,c,'B','h0_h1',steps=20,lr=.02,state=partial,spatial=False)
        stopped=optimize(r,c,'B','h0_h1',steps=20,max_seconds=0,spatial=False)
        self.assertFalse(stopped['complete'])

    def test_missing_support_and_zero_denominators(self):
        r=example('broken_loop'); r['gt_nodes']=torch.cat((r['gt_nodes'],torch.ones(1,3)))
        r['gt_edges']=edge_tensor([(0,1),(1,3),(0,3)])
        r['truth']=edge_tensor([(0,1)])
        r['edge_labels']=torch.tensor([1,0,0])
        m=metrics(r,r['node_logits'],r['relation_logits'],'B',config())
        self.assertEqual(m['excluded_target_cycle_rank'],1)
        self.assertEqual(m['support_target_beta1'],0)
        self.assertEqual(m['missed_cycle_rank'],1)
        self.assertIsNone(percentage(1.,0.)); self.assertEqual(percentage(1.,2.),50.)

    def test_output_refuses_existing_and_report_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): new_directory(tmp)
            output=new_directory(Path(tmp)/'report')
            entries=[]; r=example('false_triangle')
            for mode in ('A','B'):
                for arm in ARMS:
                    state=optimize(r,config(),mode,arm,steps=1,spatial=True)
                    entries.append(dict(record=r,state=state))
            report(output,dict(provenance={'split':'train'},selection={'synthetic':True}),entries)
            for name in ('summary.json','per-patch.csv','report.md','gradients.json','trajectories-A.png','trajectories-B.png'):
                self.assertTrue((output/name).is_file(),name)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['endpoints']['A/initial']['samples'],1)
            self.assertIn('GT_loop',summary['cohorts'])


    def test_cli_runtime_bound_resume_and_step_zero_verification(self):
        from argparse import Namespace
        from experiments.cached_logits.synthetic import create_cache
        from experiments.cached_logits.__main__ import run, load_cache
        from experiments.cached_logits.verify import verify
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=create_cache(root/'cache')/'predictions.pt'
            digest=fingerprint(load_cache(cache))
            args=Namespace(cache=cache,output=root/'run',steps=20,lr=.03,threads=1,max_seconds=.00001,resume=False)
            self.assertEqual(run(args),2)
            self.assertFalse((args.output/'COMPLETE.json').exists())
            self.assertTrue((args.output/'INCOMPLETE.json').is_file())
            args.resume=True; args.max_seconds=60
            self.assertEqual(run(args),0)
            self.assertFalse((args.output/'INCOMPLETE.json').exists())
            verified=verify(load_cache(cache),args.output)
            self.assertEqual(len(verified['results']),12)
            self.assertTrue((args.output/'SYNTHETIC_VERIFIED.json').is_file())
            self.assertFalse((args.output/'SMOKE_VERIFIED.json').exists())
            self.assertEqual(fingerprint(load_cache(cache)),digest)
            with self.assertRaises(FileExistsError): run(args)

    def test_network_cache_stage_eval_buffers_provenance_and_training_only(self):
        from argparse import Namespace
        from types import SimpleNamespace
        import yaml
        from experiments.cached_logits.cache import cache_predictions
        from experiments.cached_logits.__main__ import load_cache
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relation_embed=CountingRelationHead()
                self.bn=torch.nn.BatchNorm3d(1)
            def forward(self,volume):
                assert not self.training and not torch.is_grad_enabled()
                self.bn(volume)
                tokens,predictions,_=_batch()
                return tokens.detach(),{k:v.detach() for k,v in predictions.items()},None
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dataset=root/'dataset'
            c=config()
            model=TinyModel()
            checkpoint=root/'source.pt'
            torch_write(checkpoint,{'epoch':500,'net':model.state_dict()})
            config_path=root/'source.yaml'; config_path.write_text(yaml.safe_dump(c))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(split='train',dataset_root=str(dataset),
                sample_ids=['one','two'],categories={'one':'pending','two':'pending'})))
            _,_,targets=_batch()
            batch=[torch.ones(1,1,2,2,2),None,targets['nodes'],targets['edges']]
            class Loader:
                dataset=SimpleNamespace(records=[SimpleNamespace(sample_id=s,image=dataset/'train/raw'/s) for s in ('one','two')])
                def __iter__(self): return iter([batch,batch])
            args=Namespace(manifest=manifest,dataset=dataset,output=root/'cache',config=config_path,
                checkpoint=checkpoint,workers=0,spatial_max_node_distance=.1,device='cpu',
                dataset_name='synthetic_mri',select_count=None)
            with patch('configs.load_config',return_value=c), patch('models.build_model',return_value=model), \
                 patch('data.loaders.build_evaluation_loader',return_value=Loader()) as loader, \
                 patch('experiments.cached_logits.cache.code_provenance',return_value=dict(commit='test',branch='disposable',dirty='')):
                cache_predictions(args)
                loader.assert_called_once()
                self.assertEqual(loader.call_args.kwargs['split'],'train')
            cache=load_cache(args.output/'predictions.pt')
            self.assertEqual(cache['provenance']['checkpoint_epoch'],500)
            self.assertTrue(cache['provenance']['batch_norm_buffers_unchanged'])
            self.assertEqual(len(cache['provenance']['checkpoint_sha256']),64)
            self.assertEqual(len(cache['records']),2)
            self.assertTrue((args.output/'step-zero.json').is_file())
            for record in cache['records']:
                self.assertIn('target_cycle_basis_support_audit',record)
                self.assertTrue(record['node_supervision'].all())

    def test_next_commands_require_real_current_smoke_and_benchmark(self):
        from experiments.cached_logits.verify import next_commands
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'not verified'): next_commands(root)
            (root/'smoke-opt').mkdir()
            marker=root/'smoke-opt/SMOKE_VERIFIED.json'
            marker.write_text(json.dumps(dict(synthetic=True)))
            with self.assertRaisesRegex(ValueError,'Synthetic'): next_commands(root)
            marker.write_text(json.dumps(dict(synthetic=False,code={'commit':'abc'})))
            with patch('experiments.cached_logits.cache.code_provenance',return_value={'commit':'abc','dirty':''}):
                self.assertIn('submit.sh cache-main',next_commands(root))
                (root/'main-cache').mkdir(); (root/'main-cache/CACHE_COMPLETE.json').touch()
                self.assertIn('submit.sh benchmark',next_commands(root))
                (root/'cpu-benchmark').mkdir(); benchmark=root/'cpu-benchmark/benchmark.json'
                benchmark.write_text(json.dumps(dict(complete=True,projected_32_patch_200_step_seconds=4000)))
                self.assertNotIn('sbatch',next_commands(root))
                benchmark.write_text(json.dumps(dict(complete=True,projected_32_patch_200_step_seconds=100)))
                self.assertIn('submit.sh main',next_commands(root))

    def test_config_validation(self):
        c=config(); validate_source_config(c)
        c['loss']['edge']['balancing']['mode']='ratio_upsample'
        with self.assertRaises(ValueError): validate_source_config(c)

if __name__=='__main__': unittest.main()
