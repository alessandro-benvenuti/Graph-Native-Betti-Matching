"""Read saved cached-logit artifacts; never import an optimizer or training loop.

Inputs are trusted experiment checkpoints and YAML/JSON files supplied by their
owner. Outputs must be new and outside all source-artifact directories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

ARMS = ("classification", "h0", "h0_h1")
WEIGHTS = {"classification": (0., 0.), "h0": (.03, 0.), "h0_h1": (.03, .01)}
METRICS = ("edge_f1", "false_edges", "missed_edges", "beta0_error", "beta1_error",
           "shared_cycle_rank", "false_cycle_rank", "missed_cycle_rank")
ERRORS = ("one_minus_edge_f1", "false_edges", "missed_edges", "beta0_error",
          "beta1_error", "false_cycle_rank", "missed_cycle_rank")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                             for k, v in row.items()})


def mean(values):
    values = list(values)
    return stats.mean(values) if values else None


def pairs(edges):
    return {tuple(sorted(map(int, e))) for e in edges}


def perfect(m):
    """Exact original-GT graph under the fixed assignment, including vertices."""
    return (m["false_edges"] == m["missed_edges"] == 0
            and m["node_precision"] == m["node_recall"] == 1
            and m["beta0_error"] == m["beta1_error"] == 0)


def first_observed(rows, predicate):
    flags = [bool(predicate(row)) for row in rows]
    indices = [i for i, flag in enumerate(flags) if flag]
    if not indices:
        return dict(first_observed_step=None, first_observed_seconds=None,
                    first_onset_step_interval=None, maintained_at_later_saved_steps=False,
                    first_stable_saved_step=None, never_observed=True)
    index = indices[0]
    stable = next((i for i in indices if all(flags[i:])), None)
    return dict(first_observed_step=rows[index]["step"],
                first_observed_seconds=rows[index]["elapsed_seconds"],
                first_onset_step_interval=[0 if index == 0 else rows[index-1]["step"]+1,
                                           rows[index]["step"]],
                maintained_at_later_saved_steps=all(flags[index:]),
                first_stable_saved_step=None if stable is None else rows[stable]["step"],
                never_observed=False)


def area(xs, ys):
    """Trapezoids between observations, NOT exact errors at unsaved updates."""
    if len(xs) != len(ys) or any(b < a for a, b in zip(xs, xs[1:])):
        raise ValueError("Invalid or nonmonotonic observations")
    return sum((b-a)*(u+v)/2 for a, b, u, v in zip(xs, xs[1:], ys, ys[1:]))


def interpolate(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        raise ValueError("Interpolation would extrapolate")
    for i, upper in enumerate(xs):
        if x <= upper:
            if i == 0 or upper == xs[i-1]:
                return ys[i]
            return ys[i-1] + (ys[i]-ys[i-1])*(x-xs[i-1])/(upper-xs[i-1])
    return ys[-1]


def clipped_area(xs, ys, horizon):
    points = [(x, y) for x, y in zip(xs, ys) if x < horizon]
    points.append((horizon, interpolate(xs, ys, horizon)))
    return area([x for x, _ in points], [y for _, y in points])


def error_value(row, name):
    return 1-row["edge_f1"] if name == "one_minus_edge_f1" else row[name]


def wtl(deltas, higher_is_better=False, tolerance=1e-10):
    result = dict(win=0, tie=0, loss=0)
    for delta in deltas:
        advantage = delta if higher_is_better else -delta
        result["tie" if abs(advantage) <= tolerance else "win" if advantage > 0 else "loss"] += 1
    return result


def trajectory_analysis(histories):
    grouped = defaultdict(list)
    first, accumulated = [], []
    for (sid, mode, arm), history in histories.items():
        grouped[mode, arm].append((sid, history))
        first.append(dict(sample_id=sid, mode=mode, arm=arm,
                          **first_observed(history, perfect)))
        for metric in ERRORS:
            accumulated.append(dict(sample_id=sid, mode=mode, arm=arm, metric=metric,
                sampled_step_auc=area([r["step"] for r in history],
                                      [error_value(r, metric) for r in history]),
                sampled_step_mean_error=area([r["step"] for r in history],
                                      [error_value(r, metric) for r in history])/history[-1]["step"]))
    steps = sorted({r["step"] for history in histories.values() for r in history})
    observations = []
    for (mode, arm), entries in grouped.items():
        for step in steps:
            rows = [next(r for r in history if r["step"] == step) for _, history in entries]
            observations.append(dict(mode=mode, arm=arm, step=step, samples=len(rows),
                perfect_patches=sum(perfect(r) for r in rows),
                mean_elapsed_seconds=mean(r["elapsed_seconds"] for r in rows),
                **{m: mean(r[m] for r in rows) for m in METRICS}))
    comparisons = []
    for mode in ("A", "B"):
        ids = [sid for sid, _ in grouped[mode, "classification"]]
        for arm in ARMS[1:]:
            for step in steps:
                for metric in (*METRICS, "perfect"):
                    deltas = []
                    for sid in ids:
                        current = next(r for r in histories[sid, mode, arm] if r["step"] == step)
                        baseline = next(r for r in histories[sid, mode, "classification"] if r["step"] == step)
                        deltas.append(int(perfect(current))-int(perfect(baseline)) if metric == "perfect"
                                      else current[metric]-baseline[metric])
                    comparisons.append(dict(mode=mode, arm=arm, step=step, metric=metric,
                        mean_delta=mean(deltas), **wtl(deltas, metric in ("edge_f1", "shared_cycle_rank", "perfect"))))
    auc_comparisons = []
    lookup = {(r["sample_id"], r["mode"], r["arm"], r["metric"]): r["sampled_step_auc"]
              for r in accumulated}
    for mode in ("A", "B"):
        ids = [sid for sid, _ in grouped[mode, "classification"]]
        for arm in ARMS[1:]:
            for metric in ERRORS:
                deltas = [lookup[sid, mode, arm, metric]-lookup[sid, mode, "classification", metric]
                          for sid in ids]
                auc_comparisons.append(dict(mode=mode, arm=arm, metric=metric,
                    mean_delta_sampled_step_auc=mean(deltas), **wtl(deltas)))
    # Predeclared common wall-time horizon, shared by every patch and arm.
    horizon = min(history[-1]["elapsed_seconds"] for history in histories.values())
    wall = []
    for (sid, mode, arm), history in histories.items():
        xs = [0.] + [r["elapsed_seconds"] for r in history]
        for metric in ERRORS:
            ys = [error_value(history[0], metric)] + [error_value(r, metric) for r in history]
            wall.append(dict(sample_id=sid, mode=mode, arm=arm, metric=metric,
                common_seconds=horizon, sampled_time_auc=clipped_area(xs, ys, horizon)))
    return dict(observations=observations, first_perfect=first, accumulated=accumulated,
                paired= comparisons, auc_paired=auc_comparisons, wall_time_auc=wall,
                common_wall_time_seconds=horizon)


def audit_gradients(states, cache):
    import numpy as np
    rows, activity = [], []
    persistent = defaultdict(list)
    records = {r["source_sample_id"]:r for r in cache["records"]}
    for (sid, mode, arm), state in states.items():
        h0w, h1w = WEIGHTS[arm]
        record = records[sid]
        candidate_pairs = record["pairs"].tolist()
        if mode == "A":
            candidate_pairs = [e for e in candidate_pairs if max(e) < len(record["assignment"][0])]
        truth = pairs(record["truth"].tolist())
        is_true = np.asarray([tuple(e) in truth for e in candidate_pairs], dtype=bool)[:,None]
        for observation in state["history"]:
            step = observation["step"]
            losses, weighted = observation["losses"], observation["weighted_losses"]
            for key, weight in (("classification", 1.), ("h0", h0w), ("h1", h1w)):
                if not math.isclose(weighted[key], weight*losses[key], rel_tol=1e-6, abs_tol=1e-9):
                    raise ValueError("Saved weighted loss disagrees with arm weight")
            if not math.isclose(observation["total_loss"], sum(weighted.values()), rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Saved objective disagrees with component sum")
            activity.append(dict(sample_id=sid, mode=mode, arm=arm, step=step,
                total_loss=observation["total_loss"], **losses,
                **{"weighted_"+key: value for key, value in weighted.items()}))
        signs = defaultdict(list)
        for step_string, audit in state["audits"].items():
            step = int(step_string)
            cg = np.asarray(audit["classification"]["edge_gradient"], dtype=float).reshape(-1, 2, 2)
            cn = np.asarray(audit["classification"]["node_gradient"], dtype=float).reshape(-1, 2)
            totalg, totaln = cg.copy(), cn.copy()
            for component, weight in (("h0", h0w), ("h1", h1w)):
                if audit[component]["weight"] != weight:
                    raise ValueError("Saved gradient audit has unexpected weight")
                g = np.asarray(audit[component]["edge_gradient"], dtype=float).reshape(-1, 2, 2)
                ng = np.asarray(audit[component]["node_gradient"], dtype=float).reshape(-1, 2)
                totalg += weight*g
                totaln += weight*ng
                adverse = {name: [v for key, v in audit[component]["edges"].items()
                                 if key.endswith(suffix)]
                    for name, suffix in (("gt_suppression", "gt_correct/suppression"),
                                         ("non_gt_promotion", "gt_incorrect/promotion"))}
                conflict = audit["conflicts"][component]
                contrast = g[..., 1]-g[..., 0]
                class_contrast = cg[..., 1]-cg[..., 0]
                conflict_mask = contrast*class_contrast < -1e-24
                adverse_mask = (is_true & (contrast > 1e-12)) | (~is_true & (contrast < -1e-12))
                signs[component].append((step, conflict_mask, adverse_mask))
                row = dict(sample_id=sid, mode=mode, arm=arm, step=step, component=component,
                    weight=weight, active_in_objective=weight > 0,
                    applied_after_this_observation=step < state["settings"]["steps"],
                    directed_candidates=contrast.size,
                    directed_gt_candidates=2*int(is_true.sum()),
                    directed_non_gt_candidates=2*int((~is_true).sum()),
                    nonzero_directed_gradients=int((abs(contrast)>1e-12).sum()),
                    raw_edge_gradient_l1=float(abs(g).sum()),
                    weighted_edge_gradient_l1=float(weight*abs(g).sum()),
                    raw_node_gradient_l1=float(abs(ng).sum()),
                    weighted_node_gradient_l1=float(weight*abs(ng).sum()),
                    classification_edge_gradient_l1=float(abs(cg).sum()),
                    classification_node_gradient_l1=float(abs(cn).sum()),
                    classification_conflicting_directed_count=conflict["directed_count"],
                    raw_conflicting_contrast_mass=conflict["topology_contrast_mass"],
                    weighted_conflicting_contrast_mass=weight*conflict["topology_contrast_mass"],
                    conflicting_node_count=conflict["node_count"],
                    raw_conflicting_node_mass=conflict["node_topology_mass"],
                    weighted_conflicting_node_mass=weight*conflict["node_topology_mass"],
                    gt_node_suppression_count=audit[component]["nodes"]["matched/suppression"]["count"],
                    gt_node_suppression_mass=audit[component]["nodes"]["matched/suppression"]["mass"],
                    non_gt_node_promotion_count=audit[component]["nodes"]["unmatched/promotion"]["count"],
                    non_gt_node_promotion_mass=audit[component]["nodes"]["unmatched/promotion"]["mass"])
                row["weighted_gt_node_suppression_mass"] = weight*row["gt_node_suppression_mass"]
                row["weighted_non_gt_node_promotion_mass"] = weight*row["non_gt_node_promotion_mass"]
                for name, groups in adverse.items():
                    row[name+"_directed_count"] = sum(v["directed_count"] for v in groups)
                    row[name+"_raw_mass"] = sum(v["absolute_logit_gradient_mass"] for v in groups)
                    row[name+"_weighted_mass"] = weight*row[name+"_raw_mass"]
                rows.append(row)
            # Compare combined objective with classification; no optimizer replay.
            cc = cg[..., 1]-cg[..., 0]
            tc = totalg[..., 1]-totalg[..., 0]
            activity.append(dict(sample_id=sid, mode=mode, arm=arm, step=step,
                combined_gradient_audit=True,
                combined_edge_gradient_l1=float(abs(totalg).sum()),
                combined_node_gradient_l1=float(abs(totaln).sum()),
                combined_gt_edge_suppression_directed_count=int((is_true & (tc > 1e-12)).sum()),
                combined_non_gt_edge_promotion_directed_count=int((~is_true & (tc < -1e-12)).sum()),
                combined_opposes_classification_directed_count=int((cc*tc < -1e-24).sum()),
                combined_opposing_contrast_mass=float(abs(tc)[cc*tc < -1e-24].sum())))
        for component, observations in signs.items():
            observations.sort(key=lambda r: r[0])
            first, last = observations[0], observations[-1]
            common = np.logical_and.reduce([mask for _, mask, _ in observations])
            adverse_common = np.logical_and.reduce([mask for _, _, mask in observations])
            persistent[mode, arm, component].append(dict(sample_id=sid,
                active_in_objective=WEIGHTS[arm][0 if component == "h0" else 1] > 0,
                audited_steps=[step for step, _, _ in observations],
                conflicting_at_all_audits_directed_count=int(common.sum()),
                conflicting_at_first_and_last_directed_count=int((first[1]&last[1]).sum()),
                adverse_at_all_audits_directed_count=int(adverse_common.sum()),
                adverse_at_first_and_last_directed_count=int((first[2]&last[2]).sum())))
    summary = []
    keys = sorted({(r["mode"],r["arm"],r["component"],r["step"]) for r in rows})
    for mode, arm, component, step in keys:
        group = [r for r in rows if (r["mode"],r["arm"],r["component"],r["step"]) == (mode,arm,component,step)]
        nonnumeric = {"sample_id", "mode", "arm", "step", "component", "weight", "active_in_objective", "applied_after_this_observation"}
        result = dict(mode=mode,arm=arm,component=component,step=step,patches=len(group),
            weight=group[0]["weight"],
            patches_with_adverse_edges=sum(r["gt_suppression_directed_count"]+r["non_gt_promotion_directed_count"] > 0 for r in group),
            patches_with_classification_conflict=sum(r["classification_conflicting_directed_count"] > 0 for r in group))
        result.update({k:sum(r[k] for r in group) for k in group[0] if k not in nonnumeric})
        denom = result["directed_candidates"]
        result["conflict_fraction_of_all_candidates"] = result["classification_conflicting_directed_count"]/denom if denom else None
        active = result["nonzero_directed_gradients"]
        result["adverse_fraction_of_nonzero_gradients"] = (result["gt_suppression_directed_count"]+result["non_gt_promotion_directed_count"])/active if active else None
        summary.append(result)
    activity_summary = []
    for mode in ("A","B"):
        for arm in ARMS:
            for component in ("h0","h1"):
                scalar = [r for r in activity if (r["mode"],r["arm"]) == (mode,arm)
                          and not r.get("combined_gradient_audit")]
                gradients = [r for r in rows if (r["mode"],r["arm"],r["component"]) == (mode,arm,component)]
                activity_summary.append(dict(mode=mode,arm=arm,component=component,
                    expected_weight=WEIGHTS[arm][0 if component == "h0" else 1],
                    saved_scalar_observations=len(scalar),
                    nonzero_weighted_scalar_observations=sum(abs(r["weighted_"+component])>1e-12 for r in scalar),
                    gradient_audits=len(gradients),
                    nonzero_weighted_gradient_audits=sum(r["weighted_edge_gradient_l1"]+r["weighted_node_gradient_l1"]>1e-12 for r in gradients),
                    total_sampled_weighted_edge_gradient_l1=sum(r["weighted_edge_gradient_l1"] for r in gradients),
                    total_sampled_weighted_node_gradient_l1=sum(r["weighted_node_gradient_l1"] for r in gradients)))
    return dict(per_patch=rows, summary=summary, objective_activity=activity,
                objective_activity_summary=activity_summary,
                persistence=[dict(mode=k[0],arm=k[1],component=k[2],**r)
                             for k, values in persistent.items() for r in values])


def saved_spatial_edges(record, state, mode, config, initial):
    """Re-evaluate saved hard graph with the original spatial match convention.

    The same ungated Hungarian distance assignment is gated AFTER matching.
    No model, loss, optimizer, or new data are used. Tree edges are included.
    """
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    selected = record["selected"]
    candidate_pairs = record["pairs"]
    if mode == "A":
        m = len(record["assignment"][0])
        selected = selected[:m]
        mask = (candidate_pairs < m).all(1)
        candidate_pairs = candidate_pairs[mask]
    nodes = record["node_logits"] if initial else state["nodes"]
    relations = record["relation_logits"] if initial else state["relations"]
    if initial and mode == "A":
        relations = relations[mask]
    nt = config["evaluation"]["node_threshold"]
    et = config["evaluation"]["edge_threshold"]
    live = torch.ones(len(selected),dtype=torch.bool) if mode == "A" else (
        nodes[selected].argmax(-1) == 1 if nt is None else nodes[selected].softmax(-1)[:,1] > nt)
    logits = relations.mean(1)
    keep = logits.argmax(-1) == 1 if et is None else logits.softmax(-1)[:,1] > et
    keep &= live[candidate_pairs[:,0]] & live[candidate_pairs[:,1]]
    active = torch.where(live)[0].tolist()
    local = {old:i for i,old in enumerate(active)}
    edges = [(local[u],local[v]) for u,v in candidate_pairs[keep].tolist()]
    coordinates = record["coordinates"][selected[active],:3].numpy().astype(np.float32)
    target = record["gt_nodes"][:,:3].numpy().astype(np.float32)
    threshold = config["diagnostic"]["spatial_max_node_distance"]
    mapping = {}
    if len(coordinates) and len(target):
        distances = np.linalg.norm(coordinates[:,None,:]-target[None,:,:],axis=2)
        a,b = linear_sum_assignment(distances)
        mapping = {int(u):int(v) for u,v in zip(a,b) if distances[u,v] <= threshold}
    gt = pairs(record["gt_edges"].tolist())
    tp, fp, incident = set(), [], []
    for u,v in edges:
        if u not in mapping or v not in mapping:
            incident.append((u,v))
        elif tuple(sorted((mapping[u],mapping[v]))) in gt:
            tp.add(tuple(sorted((mapping[u],mapping[v]))))
        else:
            fp.append((u,v))
    missed = sorted(gt-tp)
    unmapped_gt = sorted(set(range(len(target)))-set(mapping.values()))
    unmapped_pred = sorted(set(range(len(active)))-set(mapping))
    def prf(tp,fp,fn):
        p = tp/(tp+fp) if tp+fp else float(fn == 0)
        r = tp/(tp+fn) if tp+fn else 1.
        f = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        return dict(precision=p,recall=r,f1=f)
    sources,targets = [t.tolist() for t in record["assignment"]]
    assigned = dict(zip(sources,targets))
    changed = [dict(query=int(selected[active[u]]),fixed_gt=assigned.get(int(selected[active[u]])),spatial_gt=v)
               for u,v in mapping.items() if assigned.get(int(selected[active[u]])) != v]
    return dict(spatial_graph_perfect=not (unmapped_gt or unmapped_pred or fp or incident or missed),
        spatial_node=prf(len(mapping),len(unmapped_pred),len(unmapped_gt)),
        spatial_edge=prf(len(tp),len(fp)+len(incident),len(missed)),
        spatial_missing_gt_node_ids=unmapped_gt,
        spatial_unmatched_predicted_query_ids=[int(selected[active[u]]) for u in unmapped_pred],
        spatial_false_edges_non_gt=fp,spatial_edges_incident_to_unmatched_nodes=incident,
        spatial_false_edge_count=len(fp)+len(incident),spatial_missed_gt_edges=missed,
        spatial_missed_edge_count=len(missed),spatial_matched_node_count=len(mapping),
        correspondence_differs_from_fixed_assignment=changed,
        original_gt_coordinates=target.tolist(),saved_prediction_coordinates=coordinates.tolist())


def support_and_spatial(cache, histories, states):
    import math
    rows, cycles, spatial = [], [], []
    for record in cache["records"]:
        sid = record["source_sample_id"]
        gt_edges = pairs(record["gt_edges"].tolist())
        sources, targets = [r.tolist() for r in record["assignment"]]
        represented = set(targets)
        excluded_edges = sorted(e for e in gt_edges if not set(e) <= represented)
        distances = []
        threshold = cache["config"]["diagnostic"]["spatial_max_node_distance"]
        for source, target in zip(sources, targets):
            distance = math.dist(record["coordinates"][source, :3].tolist(), record["gt_nodes"][target, :3].tolist())
            distances.append(dict(query=source, gt_node=target, distance=distance,
                                  exceeds_spatial_threshold=distance > threshold))
        base = histories[sid, "A", "classification"][0]
        rows.append(dict(sample_id=sid, target_nodes=len(record["gt_nodes"]),
            unmatched_target_nodes=sorted(set(range(len(record["gt_nodes"]))) - represented),
            excluded_gt_edges=excluded_edges, excluded_gt_edge_count=len(excluded_edges),
            original_gt_cycle_rank=base["target_beta1"],
            represented_gt_cycle_rank=base["support_target_beta1"],
            excluded_gt_cycle_rank=base["excluded_target_cycle_rank"],
            assigned_pairs_exceeding_spatial_threshold=sum(r["exceeds_spatial_threshold"] for r in distances),
            spatial_threshold=threshold, assigned_coordinate_distances=distances))
        for mode in ("A", "B"):
            for arm in ARMS:
                history = histories[sid, mode, arm]
                for row in (history[0], history[-1]):
                    audit = row.get("spatial_cycle_audit")
                    if audit is None:
                        raise ValueError("Expected stored endpoint spatial audit")
                    full = saved_spatial_edges(record,states[sid,mode,arm],mode,cache["config"],row["step"] == 0)
                    if not (audit["matched_nodes"] == full["spatial_matched_node_count"]
                            and audit["unmatched_target_nodes"] == len(full["spatial_missing_gt_node_ids"])
                            and audit["unmatched_predicted_nodes"] == len(full["spatial_unmatched_predicted_query_ids"])):
                        raise ValueError("Reconstructed saved graph disagrees with stored spatial audit")
                    spatial.append(dict(sample_id=sid, mode=mode, arm=arm, step=row["step"],
                        fixed_assignment_perfect=perfect(row), **audit, **full))
                for cycle in record.get("target_cycle_basis_support_audit",[]):
                    edges = pairs(cycle["target_edges"])
                    def closed(r):
                        return edges <= pairs(r["correct_edges"])
                    first = first_observed(history, closed)
                    cycles.append(dict(sample_id=sid, mode=mode, arm=arm,
                        basis_index=cycle["basis_index"], target_edges=sorted(edges),
                        target_vertices=cycle["target_vertices"],
                        represented_in_local_target=cycle["represented_in_local_target"],
                        missing_matched_target_nodes=cycle["missing_matched_target_nodes"],
                        **first,
                        observations=[dict(step=r["step"], fixed_assignment_closed=closed(r),
                            missing_gt_edges=sorted(edges-pairs(r["correct_edges"])),
                            spatial_cycle_audit=next((c for c in r.get("spatial_cycle_audit", {}).get("target_cycle_diagnostics", [])
                                if pairs(c["gt_cycle_edges"]) == edges), None)) for r in history]))
    return dict(representability=rows, spatial=spatial, individual_basis_cycles=cycles)


def flat(value, prefix=""):
    result = {}
    for key, child in value.items():
        path = prefix+"."+key if prefix else key
        if isinstance(child, dict):
            result.update(flat(child, path))
        else:
            result[path] = child
    return result


def config_comparison(cache, campaigns):
    import yaml
    config = cache["config"]
    effective = flat(config)
    additions = {r["path"]:r for r in cache["provenance"].get("configuration_compatibility",{}).get("additions",[])}
    relevant = [key for key in effective if key.startswith(("loss.","topology.","evaluation.node_threshold","evaluation.edge_threshold"))]
    output, reference_settings = [], {}
    for name, path in campaigns.items():
        reference = yaml.safe_load(Path(path).read_text())
        if reference.get("defaults"):
            raise ValueError("Campaign comparison requires the actual resolved YAML, not a template")
        reference_settings[name] = reference
        other = flat(reference)
        for key in sorted(set(relevant) | {k for k in other if k.startswith(("loss.", "topology."))}):
            value, ref = effective.get(key), other.get(key)
            output.append(dict(reference=name,path=key,diagnostic_cached_config=value,campaign=ref,
                equal=value == ref, substituted_historical_default=additions.get(key),
                diagnostic_effective_override=(
                    "Per-arm weights: classification 0/0, H0 .03/0, H0+H1 .03/.01; constant from step 0. Cached enabled/log_only/schedules/weight do not control this objective."
                    if key in ("topology.betti_h0.enabled","topology.betti_h1.enabled","topology.betti_h0.log_only","topology.betti_h1.log_only","topology.betti_h0.weight","topology.betti_h1.weight","topology.betti_h0.activation_epoch","topology.betti_h1.activation_epoch","topology.betti_h0.warmup_epochs","topology.betti_h1.warmup_epochs","topology.betti_h0.ramp_epochs","topology.betti_h1.ramp_epochs") else None)))
    compact_keys = ("loss.weights.class", "loss.weights.edges",
        "loss.node.classification.name", "loss.node.classification.class_weights",
        "loss.node.classification.focal_gamma", "loss.node.classification.curriculum.enabled",
        "loss.edge.classification.name", "loss.edge.classification.class_weights",
        "loss.edge.classification.focal_gamma", "loss.edge.classification.curriculum.enabled",
        "loss.edge.balancing.mode", "loss.edge.candidates.include_unmatched",
        "topology.betti_h0.weight", "topology.betti_h1.weight",
        "topology.betti_h0.normalization", "topology.betti_h1.normalization",
        "topology.betti_h0.normalize", "topology.betti_h1.normalize",
        "topology.betti_h0.diagonal_factor", "topology.betti_h1.diagonal_factor",
        "topology.betti_h0.unmatched_weight",
        "topology.betti_h1.false_positive_weight", "topology.betti_h1.false_negative_weight",
        "topology.complex.mode", "topology.complex.aggregation", "topology.complex.alpha",
        "topology.complex.detach_unmatched_edge_probabilities",
        "topology.complex.unmatched_object_threshold", "topology.complex.max_active_unmatched",
        "topology.sampling.enabled", "topology.sampling.betti_h0_max_graphs_per_rank",
        "topology.sampling.betti_h1_max_graphs_per_rank",
        "topology.betti_h0.ramp_epochs", "topology.betti_h1.ramp_epochs")
    compact = [r for r in output if r["path"] in compact_keys]
    verified_weight_match = None
    if "trial_0002" in reference_settings:
        c = reference_settings["trial_0002"]["topology"]
        verified_weight_match = (c["betti_h0"]["weight"] == .03 and c["betti_h1"]["weight"] == .01)
    return dict(comparisons=output, compact=compact,
        trial_0002_verified_exact_h0_h1_weight_match=verified_weight_match,
        diagnostic_config=config, reference_configs=reference_settings,
        historical_defaults=list(additions.values()),
        comparison_status="actual resolved configurations" if campaigns else "pending actual campaign resolved YAML",
        conventions=dict(classification_reduction="Weighted focal/CE sum divided by sum of target class weights, separately per graph and direction; then mean of the two directed edge losses.",
            node_classification="All object queries, B only optimized; matched queries positive, all others negative.",
            edge_classification="All matched-matched candidates, fixed assignment, no unmatched-incident labels. A and B classify this same M-M subset.",
            optimized_node_coefficient=config["loss"]["weights"]["class"],
            optimized_edge_coefficient=config["loss"]["weights"]["edges"],
            focal_progress="Criterion pinned to epoch 500, progress 100%, final gamma; campaign follows its configured training schedule.",
            mode_A="Matched nodes only; node probabilities 1, vertex births zero; only directed M-M relation logits optimized.",
            mode_B="Fixed cached matched prefix plus selected unmatched queries; all node logits and cached relations optimized. Matched topology node probabilities clamped to 1; hard graph node activity uses optimized probabilities.",
            query_selection="Cached once using production selector; not reselected during optimization. Campaign reselects from its changing predictions and assignments.",
            topology_probabilities="Mean of the two directed positive softmax probabilities.",
            inference_probabilities="Softmax of mean of the two directed logits.",
            graph_reduction="Independent Adam for each graph/mode/arm; no shared-network or multi-graph batch gradient.",
            campaign_reduction="Node focal averages all queries in a batch. Matched-edge focal pools candidates across the batch, so graphs with more candidates have greater weight; orientation is randomly selected. Each topology term averages selected eligible graph losses. DDP subsequently averages shared-parameter gradients across ranks.",
            diagnostic_reduction="Each graph has a separate objective and Adam state. Both orientation focal means are averaged. Per-graph topology helpers are reused, with no cross-graph loss averaging in the optimization.",
            included_terms="Classification plus the explicit constant arm-weighted H0/H1. Original enabled/log_only flags are bypassed intentionally.",
            excluded_terms=cache["provenance"].get("omitted_loss_terms",["boxes","nodes","cardinality"])))


def plots(output, analysis, histories, gradient):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = dict(classification="#444444", h0="#0072B2", h0_h1="#D55E00")
    for mode in ("A", "B"):
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), constrained_layout=True)
        for ax, metric in zip(axes.flat, METRICS):
            for arm in ARMS:
                rows = [r for r in analysis["observations"] if r["mode"] == mode and r["arm"] == arm]
                ax.plot([r["step"] for r in rows], [r[metric] for r in rows], marker=".", color=colors[arm], label=arm)
            ax.set(title=metric, xlabel="update", xscale="symlog")
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f"Mode {mode}: all arms at the same saved updates (patch means)")
        fig.savefig(output/f"convergence-{mode}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for mode, ax in zip(("A", "B"), axes):
        for arm in ARMS:
            rows = [r for r in analysis["observations"] if r["mode"] == mode and r["arm"] == arm]
            ax.step([r["step"] for r in rows],[r["perfect_patches"] for r in rows], where="post",color=colors[arm],label=arm)
        ax.set(title="Mode "+mode, xlabel="saved update",ylabel="perfect patches (observed)")
    axes[0].legend(fontsize=8)
    fig.savefig(output/"perfect-graphs.png",dpi=180)
    plt.close(fig)
    horizon = analysis["common_wall_time_seconds"]
    times = np.linspace(0, horizon, 101)
    fig, axes = plt.subplots(1, 2, figsize=(9,3.5),constrained_layout=True)
    for mode, ax in zip(("A", "B"), axes):
        for arm in ARMS:
            curves = []
            for (sid, m, a), history in histories.items():
                if (m,a) != (mode,arm): continue
                xs = [0.] + [r["elapsed_seconds"] for r in history]
                ys = [history[0]["edge_f1"]]+[r["edge_f1"] for r in history]
                curves.append([interpolate(xs,ys,t) for t in times])
            ax.plot(times,np.mean(curves,axis=0),color=colors[arm],label=arm)
        ax.set(title="Mode "+mode,xlabel="seconds per patch (common horizon)",ylabel="mean edge F1")
    axes[0].legend(fontsize=8)
    fig.savefig(output/"wall-time.png",dpi=180)
    plt.close(fig)


def markdown(output, cache, result):
    t, g, s = result["trajectories"],result["gradients"],result["support_and_spatial"]
    lines = ["# Existing-artifact cached-logit analysis", "",
        "No optimization, model forward pass, training, or data collection was performed. Input file hashes were checked before and after analysis.", "",
        "First perfection means first **observed saved step** with the full GT graph exact under the fixed assignment, including nodes. Unsaved updates are unobserved. Later maintenance means maintenance at subsequent saved steps only.", "",
        "Error areas use trapezoidal interpolation over common saved steps 0–200. They estimate sampled trajectory burden; they are not exact sums over all 200 updates. All arms use the same endpoint and observation grid.", "",
        "## Trajectories", "",
        "|Mode/arm|step|edge F1|false/missed edges|abs β0/β1|genuine/false/missed rank|perfect patches|", "|---|---:|---:|---|---|---|---:|"]
    for r in t["observations"]:
        lines.append(f'|{r["mode"]}/{r["arm"]}|{r["step"]}|{r["edge_f1"]:.5f}|{r["false_edges"]:.3f}/{r["missed_edges"]:.3f}|{r["beta0_error"]:.3f}/{r["beta1_error"]:.3f}|{r["shared_cycle_rank"]:.3f}/{r["false_cycle_rank"]:.3f}/{r["missed_cycle_rank"]:.3f}|{r["perfect_patches"]}|')
    lines += ["", "## First observed perfect graph", "", "Per-patch onset intervals, never-perfect flags, and later saved-step maintenance are in first-perfect.csv.", "",
        "|Mode/arm|never observed|median first saved step|lost perfection at later saved step|", "|---|---:|---:|---:|"]
    for mode in ("A","B"):
        for arm in ARMS:
            rows = [r for r in t["first_perfect"] if (r["mode"],r["arm"]) == (mode,arm)]
            values = [r["first_observed_step"] for r in rows if not r["never_observed"]]
            lines.append(f'|{mode}/{arm}|{sum(r["never_observed"] for r in rows)}|{stats.median(values) if values else "N/A"}|{sum(not r["never_observed"] and not r["maintained_at_later_saved_steps"] for r in rows)}|')
    lines += ["", "## Paired sampled error areas versus classification", "", "Negative delta favors the topology arm. Units are error × updates.", "",
        "|Mode/arm|metric|mean Δ area|win/tie/loss|", "|---|---|---:|---|"]
    for r in t["auc_paired"]:
        lines.append(f'|{r["mode"]}/{r["arm"]}|{r["metric"]}|{r["mean_delta_sampled_step_auc"]:+.5f}|{r["win"]}/{r["tie"]}/{r["loss"]}|')
    lines += ["", "Intermediate-step paired outcomes for every metric are in paired-steps.csv. Wall-time areas use one common horizon across every patch/mode/arm: "+f'{t["common_wall_time_seconds"]:.4f} seconds.', "",
        "Timing includes topology forward computations and audit/metric/checkpoint overhead, even for classification-only. It cannot estimate the training speed advantage of disabling topology entirely.", "",
        "|Mode/arm|total saved execution seconds|median per patch seconds|", "|---|---:|---:|"]
    for mode in ("A","B"):
        for arm in ARMS:
            seconds = [r["seconds"] for r in result["runtime"] if (r["mode"],r["arm"]) == (mode,arm)]
            lines.append(f'|{mode}/{arm}|{sum(seconds):.4f}|{stats.median(seconds):.4f}|')
    lines += ["",
        "## Representability and spatial correctness", "",
        f'Target nodes excluded by assignment: {sum(len(r["unmatched_target_nodes"]) for r in s["representability"])}. GT edges excluded: {sum(r["excluded_gt_edge_count"] for r in s["representability"])}. GT cycle rank excluded: {sum(r["excluded_gt_cycle_rank"] for r in s["representability"])}.', "",
        f'Assigned coordinate pairs beyond the recorded spatial threshold: {sum(r["assigned_pairs_exceeding_spatial_threshold"] for r in s["representability"])}. Coordinates were frozen; label changes cannot correct their localization.', "",
        "Stored spatial cycle exactness concerns cycle spaces, not tree-edge correctness. Spatial node/edge precision, recall and full graph exactness below are re-evaluated from saved logits and cached GT using the same distance-gated correspondence convention. No coordinates or logits are updated.", "",
        "|Mode/arm|step|unmatched GT nodes|unmatched predicted nodes|spatial edge F1|false/missed spatial cycle rank|spatial graph perfect patches|", "|---|---:|---:|---:|---:|---|---:|"]
    for mode in ("A","B"):
        for arm in ARMS:
            for step in (0,result["complete"]["steps"]):
                rows = [r for r in s["spatial"] if (r["mode"],r["arm"],r["step"]) == (mode,arm,step)]
                lines.append(f'|{mode}/{arm}|{step}|{sum(r["unmatched_target_nodes"] for r in rows)}|{sum(r["unmatched_predicted_nodes"] for r in rows)}|{mean(r["spatial_edge"]["f1"] for r in rows):.5f}|{sum(r["false_cycle_rank"] for r in rows)}/{sum(r["missed_cycle_rank"] for r in rows)}|{sum(r["spatial_graph_perfect"] for r in rows)}|')
    lines += ["", "## Loop coverage", "", "Selection categories (exclusive assigned labels):", "", "```json", json.dumps(cache["selection"].get("actual",cache["selection"]),indent=2), "```", "",
        "The bounded pool is not replenished when a category lacks candidates. Candidate pools overlap; assigned labels are exclusive and selection is greedy in category order. The genuine-loop-edge-break predicate uses fixed-assignment missing edges, not a spatial node match, and does not require the missing edge itself to lie on a GT cycle.", "",
        "Cycle-by-cycle saved-step closure, missing edges, support exclusions and original stored endpoint spatial diagnostics are in individual-cycles.json. Deterministic basis cycles can share edges and are not independent statistical samples.", "",
        "```json",json.dumps(result["loop_coverage"],indent=2),"```", "",
        "|Sample/basis|Mode/arm|represented|closed initially|first observed closure|maintained at later saved steps|spatially closed at endpoint|", "|---|---|---|---|---|---|---|"]
    for r in s["individual_basis_cycles"]:
        last = r["observations"][-1]["spatial_cycle_audit"]
        lines.append(f'|{r["sample_id"]}/{r["basis_index"]}|{r["mode"]}/{r["arm"]}|{r["represented_in_local_target"]}|{r["observations"][0]["fixed_assignment_closed"]}|{r["first_observed_step"]}|{r["maintained_at_later_saved_steps"]}|{None if last is None else last["hard_closed"]}|')
    lines += ["",
        "## Gradient evidence", "",
        "Raw and objective-weighted magnitudes are separate. Counts refer to directed candidate orientations, not unique undirected edges. Step 200 gradients are diagnostics at the final state and are not followed by an optimizer update. Audits exist only at steps 0,20,200, so persistence between audits is unknown.", "",
        "|Mode/component (H0+H1 arm)|step|patches with adverse edges|GT suppression count / raw mass|non-GT promotion count / raw mass|weighted conflict contrast mass|", "|---|---:|---:|---|---|---:|"]
    for r in g["summary"]:
        if r["arm"] != "h0_h1": continue
        lines.append(f'|{r["mode"]}/{r["component"]}|{r["step"]}|{r["patches_with_adverse_edges"]}/{r["patches"]}|{r["gt_suppression_directed_count"]} / {r["gt_suppression_raw_mass"]:.6g}|{r["non_gt_promotion_directed_count"]} / {r["non_gt_promotion_raw_mass"]:.6g}|{r["weighted_conflicting_contrast_mass"]:.6g}|')
    lines += ["", "Objective activity checks verified the saved weighted component losses and total objective at every stored observation. Full active/inactive-arm gradients, conflict frequencies, node gradients, and repeated-coordinate persistence are in gradient-*.csv.", "",
        "|Mode/arm/component|weight|nonzero weighted scalar observations|nonzero weighted gradient audits|", "|---|---:|---|---|"]
    for r in g["objective_activity_summary"]:
        lines.append(f'|{r["mode"]}/{r["arm"]}/{r["component"]}|{r["expected_weight"]}|{r["nonzero_weighted_scalar_observations"]}/{r["saved_scalar_observations"]}|{r["nonzero_weighted_gradient_audits"]}/{r["gradient_audits"]}|')
    lines += ["",
        "## Configuration fidelity", "", result["configuration"]["comparison_status"], "",
        "The effective objective overrides the source topology enabled/log-only/weight/schedule fields. The saved weights and raw gradients determine whether a term participated; source flags alone do not. Intentional differences: free logits, independent per-patch Adam, frozen assignments/coordinates/query set, H1=.01, constant weights, averaged directed classification, and omitted coordinate/cardinality terms.", "",
        "|Reference|setting|diagnostic cached value|campaign|historical substitution|", "|---|---|---|---|---|"]
    for r in result["configuration"]["compact"]:
        value = r["diagnostic_cached_config"]
        if r["path"] == "topology.betti_h0.weight": value = "0/.03/.03 (constant arm weights)"
        if r["path"] == "topology.betti_h1.weight": value = "0/0/.01 (constant arm weights)"
        if r["path"].endswith(".ramp_epochs"): value = "No ramp (constant)"
        lines.append(f'|{r["reference"]}|{r["path"]}|{json.dumps(value)}|{json.dumps(r["campaign"])}|{bool(r["substituted_historical_default"])}|')
    material = [r for r in result["configuration"]["compact"] if r["reference"] == "trial_0000"
        and not r["equal"] and r["path"] in (
            "topology.betti_h0.normalization","topology.betti_h1.normalization",
            "topology.complex.detach_unmatched_edge_probabilities",
            "topology.complex.aggregation","topology.complex.alpha",
            "loss.weights.class","loss.weights.edges",
            "loss.node.classification.focal_gamma","loss.edge.classification.focal_gamma")]
    lines += ["", "Material setting discrepancies beyond the specified H1 weight and constant schedule:"]
    for r in material:
        origin = "substituted historical default" if r["substituted_historical_default"] else "effective diagnostic setting"
        lines.append(f'- {r["path"]}: diagnostic {json.dumps(r["diagnostic_cached_config"])}; trial 0000 {json.dumps(r["campaign"])} ({origin}).')
    if not material:
        lines.append("None in the checked coefficient/gamma/normalization/aggregation/detachment fields, or the reference YAML was not supplied; see comparison status above.")
    lines += ["", "Trial 0002 exact H0=.03/H1=.01 weight match verified from stored YAML: "+str(result["configuration"]["trial_0002_verified_exact_h0_h1_weight_match"])+". A null value means its YAML was unavailable, not that the trial has matching settings.", "",
        "## Scope", "", "Final correctness, sampled convergence, and adverse local gradients are separate questions. This supervised training-patch diagnostic does not test shared-parameter interference, changing representations/assignments, network optimizer dynamics, validation/test performance, or generalization.", "",
        "![Mode A](convergence-A.png)", "![Mode B](convergence-B.png)", "![Observed perfection](perfect-graphs.png)", "![Wall time](wall-time.png)"]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def analyze(root, output, campaigns=None, h1_plan=None):
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = Path(h1_plan).resolve() if h1_plan else root/'h1-plan.json'
    if h1_plan and not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    protected = [root/"main-opt",root/"main-cache"]
    protected += [Path(p).resolve().parent for p in (campaigns or {}).values()]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("Output cannot contain or overlap source-artifact directories")
    if str(output).startswith('/lustre/') and not output.is_relative_to(root):
        raise ValueError("Cluster analysis outputs must stay under the disposable diagnostic root")
    if output.exists():
        raise FileExistsError("Analysis output already exists")
    sources = sorted(p for directory in (root/"main-opt", root/"main-cache")
                     for p in directory.rglob("*") if p.is_file())
    sources = sorted(set(sources) | {Path(p).resolve() for p in (campaigns or {}).values()}
                     | ({plan_path} if plan_path.is_file() else set()))
    def label(path):
        return str(path.relative_to(root)) if path.is_relative_to(root) else "external:"+str(path)
    if (root/"main-opt/RUNNING.lock").exists():
        raise ValueError("Run has a live/stale lock; analysis expects a completed, inactive run")
    before = {label(p):digest(p) for p in sources}
    complete = read_json(root/"main-opt/COMPLETE.json")
    run = read_json(root/"main-opt/run.json")
    cache_path = root/"main-cache/predictions.pt"
    cache_hash = before[str(cache_path.relative_to(root))]
    if not cache_hash == complete["cache_sha256"] == run["cache_sha256"] == read_json(root/"main-cache/CACHE_COMPLETE.json")["sha256"]:
        raise ValueError("Run and cache hashes disagree")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if complete["samples"] != len(cache["records"]) or complete["steps"] != run["steps"]:
        raise ValueError("Completion metadata disagree")
    if run["steps"] != 200:
        raise ValueError("Main analysis expects the prespecified 200-step endpoint")
    if run["arms"] != {k:list(v) for k,v in WEIGHTS.items()}:
        raise ValueError("Run weights differ from prespecified analysis arms")
    histories, states = {}, {}
    grid = None
    for record in cache["records"]:
        sid = record["source_sample_id"]
        for mode in ("A","B"):
            for arm in ARMS:
                path = root/"main-opt"/f"{sid}-{mode}-{arm}.pt"
                state = torch.load(path,map_location="cpu",weights_only=False)
                if not state["complete"] or state["step"] != run["steps"]:
                    raise ValueError("Incomplete saved arm")
                if any(state["settings"][k] != v for k,v in dict(mode=mode,arm=arm,steps=run["steps"],lr=run["lr"]).items()):
                    raise ValueError("Saved arm settings disagree")
                history = [dict(step=r["step"],elapsed_seconds=r["elapsed_seconds"],**r["metrics"])
                           for r in state["history"]]
                steps = [r["step"] for r in history]
                if grid is None: grid = steps
                if grid != steps or steps[0] != 0 or steps[-1] != run["steps"] or steps != sorted(set(steps)):
                    raise ValueError("Arms do not share one fixed observation grid")
                key = (sid,mode,arm)
                histories[key], states[key] = history, state
                if arm != "classification" and history[0] != histories[sid,mode,"classification"][0]:
                    # Elapsed initialization overhead varies, graph metrics must not.
                    baseline = histories[sid,mode,"classification"][0]
                    if {k:v for k,v in history[0].items() if k != "elapsed_seconds"} != {k:v for k,v in baseline.items() if k != "elapsed_seconds"}:
                        raise ValueError("Different step-zero graphs across arms")
    result = dict(complete=complete,run=run,selection=cache["selection"],
        trajectories=trajectory_analysis(histories),gradients=audit_gradients(states,cache),
        support_and_spatial=support_and_spatial(cache,histories,states),
        configuration=config_comparison(cache,campaigns or {}),
        runtime=[dict(sample_id=k[0],mode=k[1],arm=k[2],seconds=state["elapsed"])
                 for k,state in states.items()])
    result["frozen_h1_plan"] = read_json(plan_path) if plan_path.is_file() else None
    result["analyzer_source_sha256"] = digest(Path(__file__))
    pool = cache["selection"].get("rows",[])
    support = result["support_and_spatial"]["representability"]
    result["loop_coverage"] = dict(
        bounded_pool_patches=len(pool),
        pool_original_gt_loop_patches=sum(r["target_beta1"]>0 for r in pool),
        pool_original_gt_cycle_rank=sum(r["target_beta1"] for r in pool),
        selected_original_gt_loop_patches=sum(r["original_gt_cycle_rank"]>0 for r in support),
        selected_original_gt_cycle_rank=sum(r["original_gt_cycle_rank"] for r in support),
        selected_represented_gt_cycle_rank=sum(r["represented_gt_cycle_rank"] for r in support),
        selected_excluded_gt_cycle_rank=sum(r["excluded_gt_cycle_rank"] for r in support),
        assigned_category_counts=cache["selection"].get("actual"),
        category_availability=cache["selection"].get("category_availability"),
        category_quota_shortfall=cache["selection"].get("category_quota_shortfall"))
    after = {label(p):digest(p) for p in sources}
    if before != after:
        raise ValueError("An input changed during read-only analysis")
    output.mkdir(parents=True)
    write_json(output/"input-sha256.json",before)
    write_json(output/"analysis.json",result)
    write_csv(output/"observations.csv",result["trajectories"]["observations"])
    write_csv(output/"first-perfect.csv",result["trajectories"]["first_perfect"])
    write_csv(output/"paired-steps.csv",result["trajectories"]["paired"])
    write_csv(output/"sampled-error-areas.csv",result["trajectories"]["accumulated"])
    write_csv(output/"paired-error-areas.csv",result["trajectories"]["auc_paired"])
    write_csv(output/"wall-time-areas.csv",result["trajectories"]["wall_time_auc"])
    write_csv(output/"gradient-per-patch.csv",result["gradients"]["per_patch"])
    write_csv(output/"gradient-summary.csv",result["gradients"]["summary"])
    write_csv(output/"gradient-persistence.csv",result["gradients"]["persistence"])
    write_csv(output/"objective-activity.csv",result["gradients"]["objective_activity"])
    write_csv(output/"objective-activity-summary.csv",result["gradients"]["objective_activity_summary"])
    write_csv(output/"spatial-endpoints.csv",result["support_and_spatial"]["spatial"])
    write_json(output/"representability.json",result["support_and_spatial"]["representability"])
    write_json(output/"individual-cycles.json",result["support_and_spatial"]["individual_basis_cycles"])
    write_csv(output/"configuration-comparison.csv",result["configuration"]["comparisons"])
    write_csv(output/"runtime.csv",result["runtime"])
    plots(output,result["trajectories"],histories,result["gradients"])
    markdown(output,cache,result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--campaign-config",type=Path)
    parser.add_argument("--weight-match-config",type=Path)
    parser.add_argument("--h1-plan",type=Path)
    args = parser.parse_args()
    campaigns = {name:path for name,path in (("trial_0000",args.campaign_config),("trial_0002",args.weight_match_config)) if path}
    result = analyze(args.root,args.output,campaigns,args.h1_plan)
    print(json.dumps(dict(output=str(args.output),samples=result["complete"]["samples"],read_only=True)))
