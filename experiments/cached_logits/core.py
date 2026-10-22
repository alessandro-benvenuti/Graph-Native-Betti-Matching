"""Disposable free-logit objective. Production training files are never modified."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from training.losses.criterion import GraphCriterion
from training.losses.betti_filtration import node_edge_confidences
from training.losses.betti_h0 import h0_betti_matching_loss
from training.losses.betti_h1 import cycle_space_matching_loss, compute_cycle_space_matching
from training.evaluation.metrics import graph_betti_numbers

ARMS = {"classification": (0., 0.), "h0": (.03, 0.), "h0_h1": (.03, .01)}
STEPS = (0, 1, 5, 10, 20, 50, 100, 200)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    """Content hash independent of torch serialization storage IDs."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(str((str(array.dtype), tuple(array.shape))).encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item: visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def canonical(edges, *, reject_duplicates=False):
    result = []
    for u, v in torch.as_tensor(edges).reshape(-1, 2).tolist():
        u, v = int(u), int(v)
        if u == v: raise ValueError('self-loop')
        result.append(tuple(sorted((u, v))))
    if reject_duplicates and len(result) != len(set(result)):
        raise ValueError('duplicate undirected candidate')
    return sorted(set(result))


def edge_tensor(edges):
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2)


def directed_probabilities(logits):
    return logits.softmax(-1)[..., 1].mean(1)


def finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all(): raise FloatingPointError('non-finite tensor')
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)


def validate_record(record):
    finite(record)
    pairs = record['pairs']
    canonical(pairs, reject_duplicates=True)
    if not torch.equal(pairs, edge_tensor(canonical(pairs))):
        raise ValueError('candidate pairs must be canonical and ordered')
    selected = record['selected']
    count = len(selected)
    expected = torch.combinations(torch.arange(count), r=2)
    if not torch.equal(pairs, expected): raise ValueError('incomplete fixed support')
    if len(set(selected.tolist())) != count: raise ValueError('duplicate query')
    source, target = record['assignment']
    if not torch.equal(selected[:len(source)], source): raise ValueError('matched prefix changed')
    if len(set(target.tolist())) != len(target): raise ValueError('non-injective assignment')
    if record['relation_logits'].shape != (len(pairs), 2, 2): raise ValueError('directed shape')
    if pairs.numel() and int(pairs.max()) >= count: raise ValueError('candidate bounds')
    truth = set(canonical(record['truth']))
    if not truth <= set(canonical(pairs)): raise ValueError('missing true candidate')
    mm = (pairs < len(source)).all(1)
    if (record['edge_supervision'] & ~mm).any(): raise ValueError('unmatched classification label')
    expected_labels = torch.tensor([int(tuple(p) in truth) for p in pairs.tolist()])
    if not torch.equal(record['edge_labels'][mm], expected_labels[mm]): raise ValueError('wrong labels')
    if (record['edge_labels'][~mm] != -1).any(): raise ValueError('unmatched edges need sentinel labels')


def mode_view(record, mode):
    validate_record(record)
    if mode not in ('A', 'B'): raise ValueError(mode)
    result = dict(record)
    matched = len(record['assignment'][0])
    if mode == 'A':
        mask = (record['pairs'] < matched).all(1)
        result['selected'] = record['selected'][:matched]
        for key in ('pairs', 'relation_logits', 'edge_labels', 'edge_supervision'):
            result[key] = record[key][mask].clone()
    return result


class Objective:
    def __init__(self, config):
        self.config = config
        self.criterion = GraphCriterion(config, torch.nn.Identity(), torch.nn.Identity())
        # The source is completed epoch 500; curriculum has reached final gamma.
        self.criterion.set_training_progress(500, 100.)

    def filtration(self, view, nodes, relations, mode):
        pairs = view['pairs']
        m = len(view['assignment'][0])
        presence = (torch.arange(len(view['selected'])) < m).to(nodes.dtype)
        q = nodes.softmax(-1)[view['selected'], 1]
        q = torch.where(presence.bool(), torch.ones_like(q), q)
        raw = directed_probabilities(relations)
        if mode == 'A': return raw, torch.ones_like(q), presence
        spec = self.config['topology']['complex']
        mm = (pairs < m).all(1)
        if spec['detach_unmatched_edge_probabilities']:
            raw = torch.where(mm, raw, raw.detach())
        return node_edge_confidences(q, raw, pairs, aggregation=spec['aggregation'],
                                     alpha=spec['alpha']), q, presence

    def __call__(self, view, nodes, relations, mode):
        zero = (nodes.sum() + relations.sum()) * 0.
        labels = torch.zeros(len(nodes), dtype=torch.long)
        labels[view['assignment'][0]] = 1
        node_loss = self.criterion._classification_loss(nodes, labels, self.criterion.node_config) if len(nodes) else zero
        mask = view['edge_supervision']
        edge_loss = zero
        if mask.any():
            # Expected training orientation loss; never softmax(mean(logits)).
            edge_loss = sum(self.criterion._classification_loss(
                relations[mask, direction], view['edge_labels'][mask], self.criterion.edge_config
            ) for direction in range(2)) * .5
        weights = self.config['loss']['weights']
        classification = weights['class'] * node_loss + weights['edges'] * edge_loss
        p, q, presence = self.filtration(view, nodes, relations, mode)
        topology = self.config['topology']
        losses = dict(node_class=node_loss, edge_class=edge_loss, classification=classification)
        matchings = {}
        for key, function in (('h0', h0_betti_matching_loss), ('h1', cycle_space_matching_loss)):
            spec = topology['betti_' + key]
            kw = dict(num_vertices=len(q), diagonal_factor=spec['diagonal_factor'],
                      normalize=spec['normalize'], normalization=spec['normalization'])
            if key == 'h0':
                # Explicit births at zero for A; production clamping/absence for B.
                kw.update(unmatched_weight=spec['unmatched_weight'], node_probabilities=q,
                          target_node_presence=presence)
            else:
                kw.update(false_positive_weight=.25, false_negative_weight=1.)
            # Production excludes empty-target H0 (no fixed oldest real class).
            if key == 'h0' and len(view['assignment'][0]) == 0:
                losses[key], matchings[key] = zero, None
            else:
                losses[key], matchings[key] = function(p, view['pairs'], view['truth'], **kw)
        finite(losses)
        return losses, matchings


def prf(tp, fp, fn):
    # Empty-set exactness convention; percentages still use null for zero baseline.
    precision = tp / (tp + fp) if tp + fp else float(fn == 0)
    recall = tp / (tp + fn) if tp + fn else 1.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.
    return precision, recall, f1


def hard_graph(view, nodes, relations, mode, config):
    selected = view['selected']
    threshold = config['evaluation']['node_threshold']
    live = torch.ones(len(selected), dtype=torch.bool) if mode == 'A' else (
        nodes[selected].argmax(-1) == 1 if threshold is None else
        nodes[selected].softmax(-1)[:, 1] > float(threshold))
    threshold = config['evaluation']['edge_threshold']
    # Keep existing spatial inference convention, which differs from topology.
    logits = relations.mean(1)
    keep = logits.argmax(-1) == 1 if threshold is None else logits.softmax(-1)[:, 1] > float(threshold)
    keep &= live[view['pairs'][:, 0]] & live[view['pairs'][:, 1]]
    mapping = dict(zip(*[x.tolist() for x in view['assignment']]))
    n_gt = len(view['gt_nodes'])
    ids = [mapping.get(int(q), n_gt + int(q)) for q in selected]
    edges = {tuple(sorted((ids[u], ids[v]))) for u, v in view['pairs'][keep].tolist()}
    return live, ids, edges


def metrics(view, nodes, relations, mode, config, *, spatial=False):
    live, ids, pred = hard_graph(view, nodes, relations, mode, config)
    gt = set(canonical(view['gt_edges']))
    n_gt = len(view['gt_nodes'])
    represented_nodes = set(view['assignment'][1].tolist())
    represented = {e for e in gt if set(e) <= represented_nodes}
    pred_nodes = {ids[i] for i in torch.where(live)[0].tolist()}
    # Remap surviving vertices to count components without padded isolated slots.
    remap = {v: i for i, v in enumerate(sorted(pred_nodes))}
    b0, b1 = graph_betti_numbers(len(pred_nodes), [(remap[u], remap[v]) for u, v in pred])
    t0, t1 = graph_betti_numbers(n_gt, list(gt))
    represented_map = {v: i for i, v in enumerate(sorted(represented_nodes))}
    r0, r1 = graph_betti_numbers(len(represented_nodes), [
        (represented_map[u], represented_map[v]) for u, v in represented])
    candidates = sorted(pred | gt)
    matching = compute_cycle_space_matching([float(e in pred) for e in candidates],
        candidates, gt, num_vertices=max([n_gt] + [v + 1 for e in candidates for v in e]))
    result = dict(predicted_nodes=len(pred_nodes), predicted_edges=len(pred),
                  false_edges=len(pred - gt), missed_edges=len(gt - pred),
                  beta0=b0, beta1=b1, beta0_error=abs(b0-t0), beta1_error=abs(b1-t1),
                  support_beta0_error=abs(b0-r0), support_beta1_error=abs(b1-r1),
                  target_beta0=t0, target_beta1=t1, support_target_beta1=r1,
                  excluded_target_nodes=n_gt-len(represented_nodes),
                  excluded_target_node_ids=sorted(set(range(n_gt))-represented_nodes),
                  excluded_target_cycle_rank=t1-r1,
                  shared_cycle_rank=matching.shared_rank, false_cycle_rank=matching.false_prediction_rank,
                  missed_cycle_rank=matching.missed_target_rank,
                  support_missed_cycle_rank=r1-matching.shared_rank,
                  correct_edges=sorted(pred & gt))
    for prefix, truth_set, prediction_set in (
        ('edge', gt, pred), ('support_edge', represented, pred),
        ('node', set(range(n_gt)), pred_nodes)):
        tp, fp, fn = len(truth_set & prediction_set), len(prediction_set-truth_set), len(truth_set-prediction_set)
        for name, value in zip(('precision','recall','f1'), prf(tp, fp, fn)):
            result[prefix+'_'+name] = value
    result['node_metrics_applicable'] = mode == 'B'
    threshold = config['evaluation']['node_threshold']
    all_live = nodes.argmax(-1) == 1 if threshold is None else nodes.softmax(-1)[:,1] > float(threshold)
    assigned = torch.zeros(len(nodes),dtype=torch.bool); assigned[view['assignment'][0]] = True
    all_node = prf(int((all_live & assigned).sum()), int((all_live & ~assigned).sum()),
                   len(view['gt_nodes'])-int((all_live & assigned).sum()))
    result.update(zip(('all_query_node_precision','all_query_node_recall','all_query_node_f1'),all_node))
    in_support = torch.zeros(len(nodes),dtype=torch.bool); in_support[view['selected']] = True
    result['active_queries_outside_graph_support'] = int((all_live & ~in_support).sum())
    if spatial:
        from scripts.compare_loop_patch_predictions import spatial_cycle_result
        active = torch.where(live)[0].tolist()
        local = {ids[v]: i for i, v in enumerate(active)}
        prediction = dict(nodes_dhw=view['coordinates'][view['selected'][active], :3].tolist(),
                          edges=[(local[u], local[v]) for u, v in sorted(pred)])
        result['spatial_cycle_audit'] = spatial_cycle_result(prediction, view['gt_nodes'],
            view['gt_edges'], max_node_distance=config['diagnostic']['spatial_max_node_distance'])
        audit = result['spatial_cycle_audit']
        audit['beta0_absolute_error'] = abs(audit['predicted_beta0']-audit['target_beta0'])
    return result


def gradient_audit(view, nodes, relations, mode, losses, matchings, weights):
    inputs = [relations, nodes] if nodes.requires_grad else [relations]
    grads, result = {}, {}
    mm = (view['pairs'] < len(view['assignment'][0])).all(1)
    truth = set(canonical(view['truth']))
    correct = torch.tensor([tuple(p) in truth for p in view['pairs'].tolist()], dtype=torch.bool)
    node_correct = torch.zeros(len(nodes), dtype=torch.bool); node_correct[view['assignment'][0]] = True
    for name in ('classification', 'h0', 'h1'):
        values = torch.autograd.grad(losses[name], inputs, retain_graph=True, allow_unused=True)
        eg = torch.zeros_like(relations) if values[0] is None else values[0]
        ng = torch.zeros_like(nodes) if len(values) < 2 or values[1] is None else values[1]
        # Derivative along binary logit contrast (z1-z0); descent <0 promotes.
        contrast = eg[..., 1] - eg[..., 0]
        grads[name] = contrast
        groups = {}
        for support, mask in (('matched_matched', mm), ('unmatched_incident', ~mm)):
            for label, truth_mask in (('gt_correct', correct), ('gt_incorrect', ~correct)):
                for action, sign_mask in (('promotion', contrast < -1e-12), ('suppression', contrast > 1e-12)):
                    selection = (mask & truth_mask)[:, None] & sign_mask
                    groups[f'{support}/{label}/{action}'] = dict(
                        directed_count=int(selection.sum()), pair_count=int(selection.any(1).sum()),
                        absolute_logit_gradient_mass=float(eg.abs().sum(-1)[selection].sum()))
        node_groups = {}
        nc = ng[:, 1]-ng[:, 0]
        for label, mask in (('matched', node_correct), ('unmatched', ~node_correct)):
            for action, sign_mask in (('promotion', nc < -1e-12), ('suppression', nc > 1e-12)):
                selection = mask & sign_mask
                node_groups[label+'/'+action] = dict(count=int(selection.sum()), mass=float(ng[selection].abs().sum()))
        result[name] = dict(edges=groups, nodes=node_groups, weight=weights[name],
                            edge_gradient=eg.tolist(), node_gradient=ng.tolist())
    result['conflicts'] = {}
    for name in ('h0', 'h1'):
        conflict = grads['classification'] * grads[name] < -1e-24
        cg = torch.tensor(result['classification']['node_gradient'],dtype=nodes.dtype).reshape(-1,2)
        tg = torch.tensor(result[name]['node_gradient'],dtype=nodes.dtype).reshape(-1,2)
        node_conflict = (cg[:,1]-cg[:,0]) * (tg[:,1]-tg[:,0]) < -1e-24
        result['conflicts'][name] = dict(directed_count=int(conflict.sum()),
            pair_count=int(conflict.any(1).sum()), topology_contrast_mass=float(grads[name][conflict].abs().sum()),
            node_count=int(node_conflict.sum()),node_topology_mass=float(tg[node_conflict].abs().sum()))
    critical = {}
    for name, matching in matchings.items():
        records = []
        if matching is not None:
            classes = matching.prediction_pairs if name == 'h0' else matching.prediction_classes
            for index, item in enumerate(classes):
                edge = item.death_edge if name == 'h0' else item.birth_edge
                records.append(dict(edge=list(edge), gt_correct=tuple(sorted(edge)) in truth,
                    false_class=index in matching.unmatched_prediction_indices))
            targets = matching.target_pairs if name == 'h0' else matching.target_classes
            for index in matching.unmatched_target_indices:
                item = targets[index]
                edge = item.death_edge if name == 'h0' else item.birth_edge
                records.append(dict(edge=list(edge), gt_correct=tuple(sorted(edge)) in truth, missed_target=True))
        critical[name] = records
    result['critical_edges'] = critical
    result['interpretation'] = 'Local logit gradients, not causal explanations of every final change.'
    return result


def optimize(record, config, mode, arm, *, steps=200, lr=.03, state=None,
             max_seconds=3600., save=None, spatial=True):
    """Fresh Adam per arm; resumable at every completed update."""
    source_hash = fingerprint(record)
    view = mode_view(record, mode)
    objective = Objective(config)
    nodes = record['node_logits'].clone().requires_grad_(mode == 'B')
    relations = view['relation_logits'].clone().requires_grad_(True)
    inputs = [relations] + ([nodes] if mode == 'B' else [])
    optimizer = torch.optim.Adam(inputs, lr=lr)
    settings = dict(mode=mode, arm=arm, steps=steps, lr=lr, cache=source_hash,
                    config=fingerprint(config), schema=1)
    history, audits, start, elapsed = [], {}, 0, 0.
    if state is not None:
        if state['settings'] != settings: raise ValueError('resume settings/cache mismatch')
        with torch.no_grad(): nodes.copy_(state['nodes']); relations.copy_(state['relations'])
        optimizer.load_state_dict(state['optimizer'])
        history, audits, start, elapsed = copy.deepcopy(state['history']), copy.deepcopy(state['audits']), state['step'], state['elapsed']
    begun = time.perf_counter()
    h0w, h1w = ARMS[arm]
    weights = dict(classification=1., h0=h0w, h1=h1w)
    def snapshot(step, complete=False):
        return copy.deepcopy(dict(settings=settings, step=step, nodes=nodes.detach(), relations=relations.detach(),
            optimizer=optimizer.state_dict(), history=history, audits=audits,
            elapsed=elapsed+time.perf_counter()-begun, complete=complete))
    for step in range(start, steps+1):
        losses, matching = objective(view, nodes, relations, mode)
        total = sum(weights[k]*losses[k] for k in weights)
        if step in set(STEPS) | {steps} and not any(row['step'] == step for row in history):
            row = dict(step=step, losses={k:float(v.detach()) for k,v in losses.items()},
                weighted_losses={k:float(losses[k].detach())*w for k,w in weights.items()},
                total_loss=float(total.detach()), elapsed_seconds=elapsed+time.perf_counter()-begun,
                metrics=metrics(view,nodes.detach(),relations.detach(),mode,config,spatial=spatial and step in (0,steps)))
            history.append(row)
        if step in (0,20,200) and str(step) not in audits:
            audits[str(step)] = gradient_audit(view,nodes,relations,mode,losses,matching,weights)
        if step == steps:
            final = snapshot(step, True)
            if save: save(final)
            break
        if time.perf_counter()-begun >= max_seconds:
            final = snapshot(step)
            if save: save(final)
            return final
        optimizer.zero_grad(); total.backward()
        finite([p.grad for p in inputs]); optimizer.step(); finite(inputs)
        if save and (step+1) % 5 == 0: save(snapshot(step+1))
    if fingerprint(record) != source_hash: raise RuntimeError('source cache mutated')
    # A optimizer has only M-M pairs; all other cached relations remain untouched.
    return final
