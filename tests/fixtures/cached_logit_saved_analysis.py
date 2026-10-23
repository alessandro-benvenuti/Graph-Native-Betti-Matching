"""Handcrafted saved states for post-analysis tests; no optimizer is invoked."""
import copy
import hashlib
import json
from pathlib import Path
import torch


def make_saved_fixture(root):
    root = Path(root)
    cache_dir, run_dir = root/'main-cache',root/'main-opt'
    cache_dir.mkdir(parents=True)
    run_dir.mkdir()
    coords = torch.tensor([[.1,.1,.1],[.4,.4,.4],[.8,.8,.8]])
    pairs = torch.tensor([[0,1],[0,2],[1,2]])
    initial = torch.tensor([[[0.,2.],[0.,2.]],[[2.,0.],[2.,0.]],[[0.,2.],[0.,2.]]])
    final = torch.tensor([[[0.,2.],[0.,2.]]]*3)
    record = dict(source_sample_id='handcrafted_loop',selection_category='genuine_loop_edge_break',
        selected=torch.arange(3),pairs=pairs,truth=pairs,assignment=(torch.arange(3),torch.arange(3)),
        node_logits=torch.tensor([[0.,2.]]*3),relation_logits=initial,coordinates=coords,
        gt_nodes=coords,gt_edges=pairs,target_cycle_basis_support_audit=[dict(
            basis_index=0,target_edges=pairs.tolist(),target_vertices=[0,1,2],
            represented_in_local_target=True,missing_matched_target_nodes=[])])
    config = dict(evaluation=dict(node_threshold=None,edge_threshold=None),
        diagnostic=dict(spatial_max_node_distance=.1),
        loss=dict(weights=dict(**{'class':4.},edges=6.),node=dict(classification=dict(name='focal',focal_gamma=2.,class_weights=[1.,1.])),
                  edge=dict(classification=dict(name='focal',focal_gamma=.5,class_weights=[1.,1.]))),
        topology=dict(complex=dict(mode='node_aware',aggregation='hybrid',alpha=.5,
                                   detach_unmatched_edge_probabilities=False),
                      betti_h0=dict(weight=0.,normalization='feature_count'),
                      betti_h1=dict(weight=0.,normalization='feature_count')))
    cache = dict(records=[record],config=config,
        provenance=dict(split='train',synthetic=True),
        selection=dict(synthetic=True,actual=dict(genuine_loop_edge_break=1),rows=[dict(target_beta1=1)]))
    torch.save(cache,cache_dir/'predictions.pt')
    cache_hash = hashlib.sha256((cache_dir/'predictions.pt').read_bytes()).hexdigest()
    (cache_dir/'CACHE_COMPLETE.json').write_text(json.dumps(dict(sha256=cache_hash,samples=1)))
    arms = dict(classification=[0.,0.],h0=[.03,0.],h0_h1=[.03,.01])
    run = dict(cache_sha256=cache_hash,steps=200,lr=.03,arms=arms)
    (run_dir/'run.json').write_text(json.dumps(run))
    (run_dir/'COMPLETE.json').write_text(json.dumps(dict(cache_sha256=cache_hash,steps=200,samples=1,seconds=2.)))

    def metrics(closed,spatial):
        m = dict(predicted_nodes=3,predicted_edges=3 if closed else 2,false_edges=0,missed_edges=0 if closed else 1,
            node_precision=1.,node_recall=1.,node_f1=1.,edge_precision=1.,edge_recall=1. if closed else 2/3,
            edge_f1=1. if closed else .8,beta0=1,beta1=int(closed),beta0_error=0,beta1_error=int(not closed),
            target_beta0=1,target_beta1=1,support_target_beta1=1,shared_cycle_rank=int(closed),
            false_cycle_rank=0,missed_cycle_rank=int(not closed),excluded_target_cycle_rank=0,
            correct_edges=pairs.tolist() if closed else [[0,1],[1,2]])
        if spatial:
            m['spatial_cycle_audit'] = dict(target_nodes=3,predicted_nodes=3,matched_nodes=3,
                unmatched_predicted_nodes=0,unmatched_target_nodes=0,target_edges=3,
                predicted_edges=3 if closed else 2,target_beta0=1,predicted_beta0=1,
                target_beta1=1,predicted_beta1=int(closed),beta1_absolute_error=int(not closed),
                shared_cycle_rank=int(closed),false_cycle_rank=0,missed_cycle_rank=int(not closed),
                spatial_cycle_exact=closed,target_cycle_diagnostics=[dict(basis_index=0,
                    gt_cycle_edges=pairs.tolist(),hard_closed=closed,missing_gt_nodes=[],
                    missing_hard_gt_edges=[] if closed else [[0,2]])])
        return m

    for mode in ('A','B'):
        for arm,(h0,h1) in arms.items():
            history=[]
            for step in (0,1,5,10,20,50,100,200):
                losses=dict(classification=1/(step+1),h0=.2/(step+1),h1=.1/(step+1))
                weighted=dict(classification=losses['classification'],h0=h0*losses['h0'],h1=h1*losses['h1'])
                history.append(dict(step=step,elapsed_seconds=.001+step*.001,metrics=metrics(step>=5,step in (0,200)),
                    losses=losses,weighted_losses=weighted,total_loss=sum(weighted.values())))
            audits={}
            for step in (0,20,200):
                audit={}
                for name,weight in (('classification',1.),('h0',h0),('h1',h1)):
                    groups={f'{support}/{label}/{action}':dict(directed_count=0,pair_count=0,absolute_logit_gradient_mass=0.)
                        for support in ('matched_matched','unmatched_incident')
                        for label in ('gt_correct','gt_incorrect') for action in ('promotion','suppression')}
                    audit[name]=dict(weight=weight,edges=groups,
                        edge_gradient=torch.zeros(3,2,2).tolist(),node_gradient=torch.zeros(3,2).tolist(),
                        nodes={f'{label}/{action}':dict(count=0,mass=0.)
                               for label in ('matched','unmatched') for action in ('promotion','suppression')})
                audit['conflicts']={name:dict(directed_count=0,pair_count=0,topology_contrast_mass=0.,
                    node_count=0,node_topology_mass=0.) for name in ('h0','h1')}
                audits[str(step)]=audit
            state=dict(complete=True,step=200,settings=dict(mode=mode,arm=arm,steps=200,lr=.03),
                history=history,audits=audits,nodes=record['node_logits'],relations=final,elapsed=.3)
            torch.save(state,run_dir/f'handcrafted_loop-{mode}-{arm}.pt')
    return cache
