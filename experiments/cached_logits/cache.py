"""Network-only cache stage and bounded, training-only selection."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import yaml

from .core import canonical, edge_tensor, fingerprint, finite, sha256, validate_record, metrics, mode_view

from .provenance import code_provenance

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = Path('/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g05_seed364505')
DATASET = Path('/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary')
OUTPUT_ROOT = Path('/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100')


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def torch_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def new_directory(path):
    path = Path(path).resolve()
    if str(path).startswith('/lustre/') and not path.is_relative_to(OUTPUT_ROOT):
        raise ValueError('cluster outputs must stay inside the isolated output root')
    # Even an empty existing directory is refused. No force-overwrite option.
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_torch(path):
    # Only trusted, locally generated cache/state files should be loaded.
    return torch.load(str(path), map_location='cpu', weights_only=False)




def validate_source_config(config):
    loss = config['loss']
    for name, gamma in (('node',2.),('edge',.5)):
        spec = loss[name]['classification']
        if spec['name'] != 'focal' or float(spec['focal_gamma']) != gamma:
            raise ValueError(f'Unexpected source {name} focal settings: {spec}')
    if loss['edge']['candidates']['include_unmatched']:
        raise ValueError('source must use matched-only edge classification')
    if loss['edge']['balancing']['mode'] != 'none':
        raise ValueError('source must disable ratio upsampling')
    if config['model']['matcher'].get('type', 'hungarian') != 'hungarian':
        raise ValueError('this diagnostic requires Hungarian assignments')
    if not {'class','edges'} <= set(loss['enabled']): raise ValueError('missing classification objective')


def make_record(criterion, tokens, predictions, gt_nodes, gt_edges, assignment, sample_id, generator):
    source, target = [t.to(tokens.device) for t in assignment]
    selected, _, _ = criterion._topology_queries(predictions['pred_logits'], source)
    pairs = torch.combinations(torch.arange(len(selected),device=tokens.device),r=2)
    objects = tokens[:criterion.object_queries][selected]
    relation = tokens[criterion.object_queries:criterion.object_queries+criterion.relation_tokens]
    directed = torch.stack([criterion.relation_embed(criterion._relation_features(
        objects, relation, pairs if direction == 0 else pairs[:,[1,0]])) for direction in range(2)],dim=1)
    production_p = criterion._symmetric_edge_probabilities(objects,relation,pairs)
    torch.testing.assert_close(directed.softmax(-1)[...,1].mean(1), production_p, rtol=1e-6, atol=1e-7)
    truth = edge_tensor(canonical(criterion._local_true_edges(edge_tensor(canonical(gt_edges)).to(tokens.device),target,tokens.device).cpu()))
    truth_set = set(canonical(truth))
    pairs_cpu = pairs.cpu()
    mm = (pairs_cpu < len(source)).all(1)
    labels = torch.tensor([int(tuple(pair) in truth_set) if bool(matched) else -1
                           for pair,matched in zip(pairs_cpu.tolist(),mm)])
    supervision = mm.clone()
    spec = criterion.candidates
    positives = torch.where(mm & (labels==1))[0]
    negatives = torch.where(mm & (labels==0))[0]
    cap = spec.get('positive_cap')
    if cap is not None: supervision[positives[int(cap):]] = False
    maximum = spec.get('max_per_graph')
    if maximum is not None:
        limit = max(0,int(maximum)-int((supervision & (labels==1)).sum()))
        order = torch.randperm(len(negatives),generator=generator)
        supervision[negatives[order[limit:]]] = False
    record = dict(source_sample_id=sample_id, selection_category='pending',
        coordinates=predictions['pred_nodes'].detach().cpu().clone(),
        node_logits=predictions['pred_logits'].detach().cpu().clone(),
        gt_nodes=gt_nodes.detach().cpu().clone(), gt_edges=edge_tensor(canonical(gt_edges)),
        assignment=(source.cpu(),target.cpu()), selected=selected.cpu(),
        unmatched_queries=selected[len(source):].cpu(), pairs=pairs_cpu,
        relation_logits=directed.detach().cpu(), truth=truth, edge_labels=labels,
        edge_supervision=supervision, node_labels=torch.zeros(len(predictions['pred_logits']),dtype=torch.long))
    record['node_labels'][source.cpu()] = 1
    record['node_supervision'] = torch.ones(len(record['node_logits']),dtype=torch.bool)
    validate_record(record)
    return record


def select_manifest(args):
    from data.loaders.discovery import discover_synthetic_mri
    from scripts.select_topology_overfit_patches import category_candidates, select_rows
    output = Path(args.output)
    if str(output.resolve()).startswith('/lustre/') and not output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('selection output escaped isolated output root')
    if output.exists(): raise FileExistsError(output)
    # Strict root/split discovery disallows accidentally pointing at a test leaf.
    records = discover_synthetic_mri(Path(args.dataset), 'train', allow_direct=False)
    available = {r.sample_id:r for r in records}
    if args.pool:
        pool = json.loads(Path(args.pool).read_text())
        provenance = pool.get('provenance',{})
        if provenance.get('split') != 'train' or Path(provenance.get('dataset_root','')).resolve() != Path(args.dataset).resolve():
            raise ValueError('pool needs explicit train split and exact dataset_root provenance')
        rows = pool['rows']
        if len({r['source_sample_id'] for r in rows}) != len(rows): raise ValueError('duplicate pool IDs')
        if any(r['source_sample_id'] not in available for r in rows): raise ValueError('non-training pool sample')
        selected, quotas = select_rows(rows,total=args.count)
        pools = category_candidates(rows)
        ids = [r['source_sample_id'] for r in selected]
        categories = {r['source_sample_id']:r['selection_category'] for r in selected}
        availability = {k:len(v) for k,v in pools.items()}
        source = dict(pool=str(Path(args.pool).resolve()),sha256=sha256(args.pool),quotas=quotas)
    else:
        # Filename listing only; image/model/GT processing remains bounded.
        count = min(args.count,len(records))
        ids = sorted(np.random.default_rng(args.seed).choice(sorted(available), count, replace=False).tolist())
        categories = {s:'bounded_pool_pending' for s in ids}
        availability, source = {}, dict(seed=args.seed, bounded_pool=True)
    result = dict(schema=1,split='train',dataset_root=str(Path(args.dataset).resolve()),sample_ids=ids,
        categories=categories,requested=args.count,shortfall=args.count-len(ids),
        category_availability=availability,selection_provenance=source)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects a frozen manifest.
    with output.open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


def cache_predictions(args):
    from configs import load_config
    from data.loaders import build_evaluation_loader
    from models import build_model
    from models.checkpoint import load_checkpoint_file, extract_model_state, load_legacy_model_checkpoint
    from models.matcher import build_matcher
    from training.losses.criterion import GraphCriterion
    from scripts.select_topology_overfit_patches import category_candidates, select_rows, CATEGORY_ORDER
    from scripts.diagnose_node_edge_betti import _original_target_cycle_diagnostics

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest['split'] != 'train' or Path(manifest['dataset_root']).resolve() != Path(args.dataset).resolve():
        raise ValueError('manifest is not from requested training root')
    if len(manifest['sample_ids']) > 128: raise ValueError('bounded pool limit is 128 patches')
    if len(set(manifest['sample_ids'])) != len(manifest['sample_ids']): raise ValueError('duplicate manifest IDs')
    output = new_directory(args.output)
    started = time.perf_counter()
    if args.workers < 0 or args.workers > 8 or args.spatial_max_node_distance <= 0:
        raise ValueError('invalid worker count or spatial distance')
    provenance_code = code_provenance()
    if provenance_code['dirty']:
        raise ValueError('commit experimental code before producing a network cache')
    source_config = load_config(args.config)
    validate_source_config(source_config)
    config = copy.deepcopy(source_config)
    # Deliberate mode B support override only. Read all other topology settings.
    config['topology']['complex']['mode'] = 'node_aware'
    config['runtime'].update(device=args.device,workers=args.workers)
    config['diagnostic'] = dict(spatial_max_node_distance=args.spatial_max_node_distance)
    config['topology']['betti_h1'].update(false_positive_weight=.25,false_negative_weight=1.)
    config['data'].update(batch_size=1,validation_batch_size=1)
    config['data']['datasets'][args.dataset_name]['root'] = str(args.dataset)
    checkpoint_digest = sha256(args.checkpoint)
    checkpoint = load_checkpoint_file(args.checkpoint)
    if int(checkpoint.get('epoch',-1)) != 500: raise ValueError('checkpoint must be completed epoch 500')
    finite(extract_model_state(checkpoint))
    seed = int(config['experiment']['seed'])
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = build_evaluation_loader(config,dataset_name=args.dataset_name,split='train',sample_ids=manifest['sample_ids'])
    # Verify actual loader records resolve to the requested train leaf.
    train_leaf = (Path(args.dataset)/'train').resolve()
    for r in loader.dataset.records:
        if train_leaf not in r.image.resolve().parents: raise ValueError('loader escaped training split')
    model = build_model(config).to(args.device).eval()
    loading = load_legacy_model_checkpoint(model,args.checkpoint,map_location='cpu')
    matcher = build_matcher(config)
    criterion = GraphCriterion(config,matcher,model.relation_embed).to(args.device)
    # All BN buffers are checked after caching, not merely assumed unchanged.
    buffers = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    records = []
    with torch.no_grad():
        for i,batch in enumerate(loader):
            volume = batch[0] if config['training']['input']=='image' else batch[1]
            finite(volume)
            targets = dict(nodes=[batch[2][0].to(args.device)],edges=[batch[3][0].to(args.device)])
            finite(targets)
            tokens,predictions,_ = model(volume.to(device=args.device,dtype=torch.float32))
            finite(tokens); finite(predictions)
            assignments = matcher(predictions,targets)
            record = make_record(criterion,tokens[0],{k:v[0] for k,v in predictions.items()},
                targets['nodes'][0],targets['edges'][0],assignments[0],loader.dataset.records[i].sample_id,generator)
            record['selection_category'] = manifest['categories'][record['source_sample_id']]
            records.append(record)
    for name,value in model.named_buffers():
        if not torch.equal(value.cpu(),buffers[name]): raise RuntimeError('evaluation changed model buffers: '+name)
    rows = []
    for record in records:
        m = metrics(record,record['node_logits'],record['relation_logits'],'B',config)
        rows.append(dict(source_sample_id=record['source_sample_id'], target_beta1=m['target_beta1'],
            predicted_beta1=m['beta1'],target_beta0=m['target_beta0'],predicted_beta0=m['beta0'],
            matched_gt_fraction=(len(record['assignment'][1])/len(record['gt_nodes']) if len(record['gt_nodes']) else 1.),
            unmatched_target_nodes=m['excluded_target_nodes'],
            unmatched_predicted_nodes=m['predicted_nodes']-sum(
                int(record['node_logits'][q].argmax())==1 for q in record['assignment'][0]),
            missing_gt_edges=m['missed_edges'],false_positive_edges=m['false_edges'],
            severity=3*m['beta1_error']+2*m['beta0_error']+m['missed_edges']+m['false_edges']))
    pools = category_candidates(rows)
    selected_records = records
    if args.select_count:
        if args.select_count < 5: raise ValueError('main enriched selection needs at least five patches')
        selected, quotas = select_rows(rows,total=args.select_count)
        lookup = {r['source_sample_id']:r for r in records}
        selected_records = []
        for row in selected:
            record = lookup[row['source_sample_id']]
            record['selection_category'] = row['selection_category']; selected_records.append(record)
    else:
        quotas = {}
        for record in selected_records:
            memberships = [c for c in CATEGORY_ORDER if any(r['source_sample_id']==record['source_sample_id'] for r in pools[c])]
            record['selection_category'] = memberships[0] if memberships else 'other'
    selection = dict(sample_ids=[r['source_sample_id'] for r in selected_records],
        category_availability={k:len(v) for k,v in pools.items()},quotas=quotas,
        actual={k:sum(r['selection_category']==k for r in selected_records) for k in (*CATEGORY_ORDER,'other','fallback_high_error')},
        requested=args.select_count or len(records),shortfall=max(0,(args.select_count or len(records))-len(selected_records)),
        pool_size=len(records), rows=rows, fixed_assignment_selection=True)
    base, remainder = divmod(selection['requested'],len(CATEGORY_ORDER))
    selection['category_quota_shortfall'] = {c:max(0,base+int(i<remainder)-selection['actual'][c])
        for i,c in enumerate(CATEGORY_ORDER)}
    selection['note'] = 'Selection uses cached fixed-assignment errors on frozen B support; enriched, not prevalence representative.'
    if sha256(args.checkpoint) != checkpoint_digest:
        raise ValueError('source checkpoint changed during caching')
    provenance = dict(source_config_sha256=sha256(args.config),checkpoint_sha256=checkpoint_digest,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_epoch=500,loading=asdict(loading),
        code=provenance_code,manifest_sha256=sha256(args.manifest),manifest=manifest,
        dataset_root=str(Path(args.dataset).resolve()),split='train',batch_norm_buffers_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,torch_version=str(torch.__version__),
        overrides={'topology.complex.mode':'node_aware','betti_h1.false_positive_weight':.25,
                   'betti_h1.false_negative_weight':1.},source_config=source_config,
        omitted_loss_terms=['boxes','nodes','cardinality'],
        omission_reason='coordinates fixed; cardinality uses nondifferentiable argmax',
        classification_orientation='equal mean of the two directed losses (expected random orientation)',
        inference_edge_probability='softmax of mean directed logits',
        topology_edge_probability='mean of directed positive softmax probabilities')
    for record in selected_records:
        record['target_cycle_basis_support_audit'] = _original_target_cycle_diagnostics(
            record['gt_edges'],target_node_count=len(record['gt_nodes']),
            matched_target_nodes=record['assignment'][1].tolist())
    payload = dict(schema=1,records=selected_records,config=config,provenance=provenance,selection=selection)
    payload['step_zero'] = {r['source_sample_id']:{mode:metrics(
        mode_view(r,mode),
        r['node_logits'],mode_view(r,mode)['relation_logits'],mode,config)
        for mode in ('A','B')} for r in selected_records}
    json_write(output/'step-zero.json',payload['step_zero'])
    payload['content_sha256'] = fingerprint(payload)
    torch_write(output/'predictions.pt',payload)
    json_write(output/'selection.json',selection)
    json_write(output/'provenance.json',provenance)
    (output/'resolved-config.yaml').write_text(yaml.safe_dump(config))
    json_write(output/'CACHE_COMPLETE.json',dict(sha256=sha256(output/'predictions.pt'),samples=len(selected_records)))
    print(json.dumps(dict(output=str(output),selection=selection,seconds=time.perf_counter()-started),indent=2))
