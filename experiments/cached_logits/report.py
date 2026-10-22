"""Fixed-endpoint paired summaries, cycle turnover, cohorts and trajectories."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics

from training.evaluation.metrics import graph_betti_numbers
from .core import ARMS
from .cache import json_write

NUMERIC = ('node_precision','node_recall','node_f1','edge_precision','edge_recall','edge_f1',
    'support_edge_f1','beta0_error','beta1_error','support_beta0_error','support_beta1_error',
    'false_edges','missed_edges','predicted_nodes','predicted_edges','shared_cycle_rank',
    'false_cycle_rank','missed_cycle_rank','support_missed_cycle_rank','excluded_target_nodes',
    'excluded_target_cycle_rank','active_queries_outside_graph_support',
    'all_query_node_precision','all_query_node_recall','all_query_node_f1','runtime_seconds')


def percentage(delta, baseline):
    return None if baseline == 0 else 100.*delta/abs(baseline)


def flatten(states):
    rows = []
    for entry in states:
        state, record = entry['state'], entry['record']
        mode, arm = state['settings']['mode'], state['settings']['arm']
        initial = state['history'][0]['metrics']
        final = state['history'][-1]['metrics']
        before, after = set(map(tuple,initial['correct_edges'])),set(map(tuple,final['correct_edges']))
        common_rank = graph_betti_numbers(len(record['gt_nodes']),list(before & after))[1]
        rows.append(dict(sample_id=record['source_sample_id'],category=record['selection_category'],
            mode=mode,arm=arm,step=state['step'],**final,
            runtime_seconds=state['elapsed'],
            genuine_cycle_gains=final['shared_cycle_rank']-common_rank,
            genuine_cycle_losses=initial['shared_cycle_rank']-common_rank))
        if arm == 'classification':
            rows.append(dict(sample_id=record['source_sample_id'],category=record['selection_category'],
                mode=mode,arm='initial',step=0,**initial,runtime_seconds=0.,
                genuine_cycle_gains=0,genuine_cycle_losses=0))
    return rows


def aggregate(rows):
    result = {'samples':len(rows)}
    for metric in (*NUMERIC,'genuine_cycle_gains','genuine_cycle_losses'):
        result[metric] = statistics.mean(r[metric] for r in rows) if rows else None
    # Most primary values are patch means, counts are additionally summed.
    for metric in ('false_edges','missed_edges','shared_cycle_rank','false_cycle_rank','missed_cycle_rank',
                   'genuine_cycle_gains','genuine_cycle_losses','runtime_seconds'):
        result['total_'+metric] = sum(r[metric] for r in rows)
    if rows:
        for baseline in ('initial','classification'):
            result['delta_vs_'+baseline] = {m:statistics.mean(r['delta_vs_'+baseline][m] for r in rows)
                for m in NUMERIC if 'delta_vs_'+baseline in rows[0]}
    return result


def paired(rows):
    index = {(r['sample_id'],r['mode'],r['arm']):r for r in rows}
    result = {}
    for mode in ('A','B'):
        for baseline in ('initial','classification'):
            for arm in ARMS:
                key = f'{mode}/{arm}/vs_{baseline}'
                result[key] = {}
                for metric in ('beta0_error','beta1_error','support_beta0_error','support_beta1_error','edge_f1'):
                    counts = dict(win=0,tie=0,loss=0)
                    for r in rows:
                        if r['mode'] != mode or r['arm'] != arm: continue
                        delta = r[metric]-index[r['sample_id'],mode,baseline][metric]
                        if metric != 'edge_f1': delta = -delta
                        counts['tie' if abs(delta)<1e-10 else ('win' if delta>0 else 'loss')] += 1
                    result[key][metric] = counts
    return result


def write_csv(path, rows):
    if not rows: return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})



def evidence_tables(rows, cohorts, gradients):
    lines=['## Paired outcomes versus classification', '',
        '|Mode/arm|β0 win/tie/loss|β1 win/tie/loss|edge F1 win/tie/loss|', '|---|---|---|---|']
    pairs=paired(rows)
    for mode in ('A','B'):
        for arm in ('h0','h0_h1'):
            values=pairs[f'{mode}/{arm}/vs_classification']
            lines.append('|'+mode+'/'+arm+'|'+'|'.join('/'.join(str(values[m][k]) for k in ('win','tie','loss')) for m in ('beta0_error','beta1_error','edge_f1'))+'|')
    lines += ['', '## Cohorts: H0+H1 minus classification', '',
        '|Cohort/mode|patches|Δ edge F1|Δ abs β0|Δ abs β1|', '|---|---|---|---|---|']
    for cohort,values in cohorts.items():
        for mode in ('A','B'):
            endpoint=values['endpoints'][mode+'/h0_h1']
            if endpoint['samples']:
                delta=endpoint['delta_vs_classification']
                lines.append('|'+cohort+'/'+mode+'|'+str(endpoint['samples'])+'|'+'|'.join(f'{delta[m]:+.4f}' for m in ('edge_f1','beta0_error','beta1_error'))+'|')
    lines += ['', '## Gradient evidence for the H0+H1 arm', '',
        '|Mode/step/component|GT-edge suppression count / mass|non-GT promotion count / mass|false-class critical edges that are GT|', '|---|---|---|---|']
    for mode in ('A','B'):
        for step in (0,20,200):
            selected=[r for r in gradients if r['mode']==mode and r['arm']=='h0_h1' and r['step']==step]
            if not selected: continue
            for component in ('classification','h0','h1'):
                totals=[]
                for label,action in (('gt_correct','suppression'),('gt_incorrect','promotion')):
                    groups=[r[component]['edges'][f'{support}/{label}/{action}'] for r in selected for support in ('matched_matched','unmatched_incident')]
                    totals.append(str(sum(g['directed_count'] for g in groups))+' / '+f"{sum(g['absolute_logit_gradient_mass'] for g in groups):.6g}")
                critical=sum(e.get('false_class',False) and e['gt_correct'] for r in selected for e in r['critical_edges'].get(component,[]))
                lines.append('|'+f'{mode}/{step}/{component}'+'|'+'|'.join(totals)+'|'+str(critical)+'|')
    return lines+['']


def report(output, cache, entries):
    output = Path(output)
    rows = flatten(entries)
    index = {(r['sample_id'],r['mode'],r['arm']):r for r in rows}
    for row in rows:
        for baseline in ('initial','classification'):
            base = index[row['sample_id'],row['mode'],baseline]
            deltas = {m:row[m]-base[m] for m in NUMERIC}
            row['delta_vs_'+baseline] = deltas
            row['percent_vs_'+baseline] = {m:percentage(deltas[m],base[m]) for m in NUMERIC}
    summary = {f'{mode}/{arm}':aggregate([r for r in rows if r['mode']==mode and r['arm']==arm])
        for mode in ('A','B') for arm in ('initial',*ARMS)}
    cohorts = {}
    for name,predicate in (
        ('GT_loop',lambda r:r['target_beta1']>0),('GT_nonloop',lambda r:r['target_beta1']==0),
        ('all_cycles_representable',lambda r:r['excluded_target_cycle_rank']==0),
        ('some_cycles_unrepresentable',lambda r:r['excluded_target_cycle_rank']>0),
        *((f'category:{c}',lambda r,c=c:r['category']==c) for c in sorted({r['category'] for r in rows}))):
        subset = [r for r in rows if predicate(r)]
        cohorts[name] = dict(endpoints={f'{mode}/{arm}':aggregate([r for r in subset if r['mode']==mode and r['arm']==arm])
            for mode in ('A','B') for arm in ('initial',*ARMS)}, paired=paired(subset))
    json_write(output/'per-patch.json',rows); write_csv(output/'per-patch.csv',rows)
    json_write(output/'summary.json',dict(endpoints=summary,paired=paired(rows),cohorts=cohorts,
        provenance=cache['provenance'],selection=cache['selection']))
    trajectory = []
    gradients = []
    for entry in entries:
        state = entry['state']; sid = entry['record']['source_sample_id']
        for observation in state['history']:
            trajectory.append(dict(sample_id=sid,mode=state['settings']['mode'],arm=state['settings']['arm'],
                step=observation['step'],**observation['metrics'],**observation['losses'],
                **{'weighted_'+k:v for k,v in observation['weighted_losses'].items()},
                elapsed_seconds=observation['elapsed_seconds']))
        for step,audit in state['audits'].items():
            gradients.append(dict(sample_id=sid,mode=state['settings']['mode'],arm=state['settings']['arm'],step=int(step),**audit))
    json_write(output/'trajectories.json',trajectory); write_csv(output/'trajectories.csv',trajectory)
    json_write(output/'gradients.json',gradients)
    # Readable aggregate attribution without losing per-logit evidence in JSON.
    gradient_rows = []
    for record in gradients:
        for component in ('classification','h0','h1'):
            for group,values in record[component]['edges'].items():
                gradient_rows.append(dict(sample_id=record['sample_id'],mode=record['mode'],arm=record['arm'],
                    step=record['step'],component=component,group=group,**values))
    write_csv(output/'gradient-edges.csv',gradient_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode in ('A','B'):
        fig,axes = plt.subplots(2,3,figsize=(12,6))
        for arm in ARMS:
            values = [r for r in trajectory if r['mode']==mode and r['arm']==arm]
            steps = sorted({r['step'] for r in values})
            for ax,keys in zip(axes.flat,(
                ('classification','h0','h1'),('weighted_classification','weighted_h0','weighted_h1'),
                ('beta0_error','beta1_error'),('edge_f1',),('shared_cycle_rank',),('false_cycle_rank',))):
                for key in keys:
                    ax.plot(steps,[statistics.mean(r[key] for r in values if r['step']==s) for s in steps],
                            label=arm+'/'+key,marker='.',markersize=3)
                ax.set_xlabel('optimization step'); ax.legend(fontsize=6)
        fig.suptitle(f'Mode {mode}: patch means (fixed endpoint, no best-step selection)')
        fig.tight_layout(); fig.savefig(output/f'trajectories-{mode}.png',dpi=150); plt.close(fig)
    lines = ['# Cached-logit supervised diagnostic','',
        '**SYNTHETIC FIXTURES ONLY. No network checkpoint or real patches evaluated.**' if cache['provenance'].get('synthetic') else 'Real training-patch diagnostic; no validation/test evaluation.', '',
        'Patch-specific free variables on deliberately enriched training patches. No generalization claim; correctness is relative to supplied GT.', '',
        'All primary numbers below are patch means under frozen Hungarian assignment. These are not detection edge mAP. A uses all matched vertices at birth zero; B uses a frozen node-aware support, so their supports differ. Node metrics in A describe coverage and are not node-detection outcomes. B graph metrics cover the fixed selected query set; all-query node classification metrics and active queries outside graph support are recorded separately. No edges outside graph support are evaluated.', '',
        '|Mode/arm|Node P/R/F1|Edge P/R/F1|abs β0 / β1|support abs β0 / β1|False / missed edges|Nodes / edges|Genuine / false / missed cycle ranks|seconds total|',
        '|---|---|---|---|---|---|---|---|---|']
    fmt = lambda v:f'{v:.3f}'
    for key,s in summary.items():
        groups = [('node_precision','node_recall','node_f1'),('edge_precision','edge_recall','edge_f1'),
            ('beta0_error','beta1_error'),('support_beta0_error','support_beta1_error'),('false_edges','missed_edges'),
            ('predicted_nodes','predicted_edges'),('shared_cycle_rank','false_cycle_rank','missed_cycle_rank')]
        lines.append('|'+key+'|'+'|'.join(' / '.join(fmt(s[m]) for m in group) for group in groups)+'|'+fmt(s['total_runtime_seconds'])+'|')
    lines += ['', '## Paired absolute endpoint deltas', '',
        '|Mode/arm vs baseline|Δ edge F1|Δ abs β0|Δ abs β1|Δ genuine rank|Δ false rank|', '|---|---|---|---|---|---|']
    for key,s in summary.items():
        if key.endswith('/initial'): continue
        for baseline in ('initial','classification'):
            delta=s['delta_vs_'+baseline]
            lines.append('|'+key+' vs '+baseline+'|'+'|'.join(fmt(delta[m]) for m in ('edge_f1','beta0_error','beta1_error','shared_cycle_rank','false_cycle_rank'))+'|')
    lines += ['', 'Percent deltas and all metrics are in per-patch.json/CSV; null means N/A for a zero baseline. Cohort endpoint tables and paired win/tie/loss counts are in summary.json.', '',
        '## Selection and support', '', '```json',json.dumps({k:v for k,v in cache['selection'].items() if k!='rows'},indent=2),'```','',
        'Representable target rank is the cycle rank of the GT subgraph induced by assigned target nodes. Full-GT and support errors are both reported; missing nodes are never supplied to predictions. Genuine gains/losses use intersection ranks of initial/final shared cycle spaces and are recorded separately in per-patch outputs.', '',
        '## Local gradient attribution', '',
        'gradients.json records classification/H0/H1 gradients at steps 0, 20, 200 when reached, including node attributions, critical-edge GT membership and gradient conflicts. gradient-edges.csv summarizes promotion/suppression counts and absolute logit-gradient mass by GT membership and candidate type. Both directed orientations are counted; pair counts deduplicate them. These are local diagnostics, not causal explanations of all final changes.', '',
        ]
    lines += evidence_tables(rows,cohorts,gradients)
    lines += ['## Interpretation', '']
    conclusions = {}
    for mode in ('A','B'):
        base=summary[mode+'/classification']; top=summary[mode+'/h0_h1']
        signed = [top['edge_f1']-base['edge_f1'], top['shared_cycle_rank']-base['shared_cycle_rank'],
            *[base[m]-top[m] for m in ('beta0_error','beta1_error','false_cycle_rank','missed_cycle_rank','false_edges','missed_edges')]]
        positive = any(v > 1e-10 for v in signed); negative = any(v < -1e-10 for v in signed)
        verdict = ('mixed gains and regressions' if negative else 'improvement without aggregate regression') if positive else ('regression without aggregate gain' if negative else 'no endpoint benefit beyond classification')
        conclusions[mode] = verdict
        lines.append(f'On this selected set, Mode {mode} shows **{verdict}** for H0+H1 at the fixed endpoint. This is a descriptive comparison of patch means; consult paired outcomes and cycle turnover.')
        lines.append('')
        lines.append(f"Mode {mode}, H0+H1 versus classification at the fixed endpoint: Δ edge F1 {top['edge_f1']-base['edge_f1']:+.4f}, Δ abs β0 {top['beta0_error']-base['beta0_error']:+.4f}, Δ abs β1 {top['beta1_error']-base['beta1_error']:+.4f}, Δ genuine cycle rank {top['shared_cycle_rank']-base['shared_cycle_rank']:+.4f}. Genuine cycle gains/losses versus initialization: {top['total_genuine_cycle_gains']}/{top['total_genuine_cycle_losses']}.")
        lines.append('')
    lines.append('The descriptive verdict '+('differs between modes' if conclusions['A'] != conclusions['B'] else 'is the same in both modes')+'. Their graph supports differ, so this does not isolate the mathematical effect of adding node supervision alone.')
    lines.append('')
    lines += ['Assess edge correctness, genuine cycles and turnover together; matching β1 counts alone does not establish loop recovery. This diagnostic can test whether the objective corrects free logits. It cannot identify the cause of a network-training limitation or demonstrate benefit during network training.', '',
        'Node-aware filtration uses source aggregation, normalization, clamping and detachment, with mode forced to node_aware. The topology-query set remains fixed. Classification is the average of the two orientation losses, the expected random-orientation objective. Topology weights are constant 0.03/0.01, with H1 FP/FN 0.25/1.0; this differs from the campaign ramp. Coordinate/cardinality losses are omitted because coordinates are frozen and cardinality uses argmax.', '',
        'Spatial cycle audits at initialization and endpoint reuse compare_loop_patch_predictions.spatial_cycle_result with a separately recorded distance threshold. Primary assignment metrics use all matched nodes even when their coordinates are poor; interpret the spatial audit separately.', '',
        '![Mode A](trajectories-A.png)', '', '![Mode B](trajectories-B.png)', '']
    (output/'report.md').write_text('\n'.join(lines))
