"""Tiny synthetic fixtures; never evidence about real training patches."""
from pathlib import Path
import torch
from configs import load_config
from .core import canonical, edge_tensor, fingerprint, metrics, mode_view, sha256
from .cache import new_directory, torch_write, json_write, code_provenance


def config():
    c=load_config(Path(__file__).resolve().parents[2]/'configs/pretrain_mixed.yaml',
        environment={'GNBM_OUTPUT_DIR':'/unused','PLANTS_DATASET':'/unused','SYNTHETIC_MRI_DATASET':'/unused'})
    c['model']['decoder'].update(hidden_dim=8,object_queries=4,relation_tokens=1)
    c['loss']['node']['classification'].update(name='focal',focal_gamma=2.,class_weights=[1.,1.])
    c['loss']['edge']['classification'].update(name='focal',focal_gamma=.5,class_weights=[1.,1.])
    c['loss']['edge']['balancing']['mode']='none'
    c['loss']['edge']['candidates']['include_unmatched']=False
    c['topology']['complex'].update(mode='node_aware',detach_unmatched_edge_probabilities=True,
        aggregation='hybrid',alpha=.5,unmatched_object_threshold=0.,max_active_unmatched=8)
    c['diagnostic']={'spatial_max_node_distance':.1}
    return c


def example(name='missing_connection'):
    # Three matched queries, optionally one unmatched; all coordinates fixed.
    n=4 if name=='unmatched_node' else 3
    truth={'missing_connection':[(0,1),(1,2)],'false_triangle':[(0,1),(1,2)],
        'broken_loop':[(0,1),(1,2),(0,2)],'unmatched_node':[(0,1),(1,2)],
        'correct':[(0,1),(1,2)]}[name]
    pairs=torch.combinations(torch.arange(n),r=2)
    labels=torch.tensor([int(tuple(e) in truth) if max(e)<3 else -1 for e in pairs.tolist()])
    p=torch.tensor([.75 if value==1 else .25 for value in labels])
    if name=='missing_connection': p[2]=.25
    if name=='false_triangle': p=torch.tensor([.6,.95,.9])
    if name=='broken_loop': p[1]=.25
    if name=='unmatched_node': p[labels==-1]=.85
    relations=torch.stack([torch.zeros_like(p),torch.logit(p)],dim=-1)[:,None,:].repeat(1,2,1)
    coordinates=torch.tensor([[.1,.1,.1],[.4,.4,.4],[.8,.8,.8],[.1,.8,.1]])[:n]
    return dict(source_sample_id=name,selection_category=name,coordinates=coordinates,
        node_logits=torch.tensor([[0.,1.]]*n),gt_nodes=coordinates[:3].clone(),
        gt_edges=edge_tensor(truth),assignment=(torch.arange(3),torch.arange(3)),selected=torch.arange(n),
        unmatched_queries=torch.arange(3,n),pairs=pairs,relation_logits=relations,
        truth=edge_tensor(canonical(truth)),edge_labels=labels,edge_supervision=labels>=0,
        node_labels=torch.tensor([1,1,1]+([0] if n==4 else [])))



def create_cache(output):
    output=new_directory(output)
    c=config()
    records=[example('false_triangle'),example('unmatched_node')]
    payload=dict(schema=1,config=c,records=records,
        selection={'synthetic':True,'requested':2,'shortfall':0},
        provenance={'split':'train','synthetic':True,'code':code_provenance()})
    payload['step_zero']={r['source_sample_id']:{m:metrics(mode_view(r,m),r['node_logits'],
        mode_view(r,m)['relation_logits'],m,c) for m in ('A','B')} for r in records}
    payload['content_sha256']=fingerprint(payload)
    torch_write(output/'predictions.pt',payload)
    json_write(output/'CACHE_COMPLETE.json',dict(sha256=sha256(output/'predictions.pt'),samples=2,synthetic=True))
    return output
