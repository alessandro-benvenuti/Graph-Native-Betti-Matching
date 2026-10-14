#!/usr/bin/env python3
"""Summarize full validation corruption diagnostics across checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


CORE_METRICS = (
    "node_cost_per_gt_node",
    "coordinate_cost_per_gt_node",
    "edge_raw_cost_per_candidate",
    "edge_effective_cost_per_candidate",
    "edge_raw_cost_per_gt_edge",
    "edge_effective_cost_per_gt_edge",
    "matched_node_probability_mean",
    "unmatched_node_probability_mean",
    "gt_edge_probability_mean",
    "matched_nonedge_probability_mean",
    "unmatched_endpoint_edge_probability_mean",
)

TOPOLOGY_METRICS = (
    "nodes",
    "edges",
    "components",
    "cycle_rank",
    "endpoints",
    "bifurcations",
    "false_edges",
    "false_loop_edges",
    "missing_edges",
    "extra_nodes",
    "missing_nodes",
    "duplicate_candidates",
    "subdivision_node_candidates",
    "split_node_candidates",
)

PAIRED_METRICS = (
    "node_cost_per_gt_node",
    "coordinate_cost_per_gt_node",
    "edge_raw_cost_per_candidate",
    "edge_effective_cost_per_candidate",
)


def _number(value):
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantile(sorted_values, probability):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _describe(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q10": _quantile(values, 0.10),
        "q90": _quantile(values, 0.90),
    }


def _format(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _thresholds(fieldnames):
    result = set()
    for field in fieldnames:
        if not field.startswith("t") or "_" not in field:
            continue
        raw = field[1:].split("_", 1)[0]
        try:
            result.add(float(raw))
        except ValueError:
            pass
    return sorted(result)


def _read_comparison(path):
    by_epoch = defaultdict(list)
    paired = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        thresholds = _thresholds(fieldnames)
        for row in reader:
            epoch = int(row["epoch"])
            sample_id = row["sample_id"]
            by_epoch[epoch].append(row)
            paired[sample_id][epoch] = row
    return fieldnames, thresholds, by_epoch, paired


def _metric_summary(by_epoch, thresholds, fieldnames):
    fields = [field for field in CORE_METRICS if field in fieldnames]
    for threshold in thresholds:
        prefix = f"t{threshold:g}_"
        fields.extend(
            prefix + metric
            for metric in TOPOLOGY_METRICS
            if prefix + metric in fieldnames
        )
        for metric in ("nodes", "edges", "components", "cycle_rank", "endpoints", "bifurcations"):
            delta = prefix + metric + "_minus_gt"
            if prefix + metric in fieldnames and "gt_" + metric in fieldnames:
                fields.append(delta)

    output = {}
    for epoch, rows in sorted(by_epoch.items()):
        output[epoch] = {}
        for field in fields:
            if field.endswith("_minus_gt"):
                base = field[: -len("_minus_gt")]
                metric = base.split("_", 1)[1]
                values = [
                    _number(row.get(base)) - _number(row.get("gt_" + metric))
                    for row in rows
                    if _number(row.get(base)) is not None
                    and _number(row.get("gt_" + metric)) is not None
                ]
            else:
                values = [_number(row.get(field)) for row in rows]
            output[epoch][field] = _describe(values)
    return output


def _candidate_occurrence(by_epoch, thresholds):
    candidates = (
        "false_edges",
        "false_loop_edges",
        "missing_edges",
        "extra_nodes",
        "missing_nodes",
        "duplicate_candidates",
        "subdivision_node_candidates",
        "split_node_candidates",
    )
    output = {}
    for epoch, rows in sorted(by_epoch.items()):
        output[epoch] = {}
        for threshold in thresholds:
            for candidate in candidates:
                field = f"t{threshold:g}_{candidate}"
                values = [_number(row.get(field)) for row in rows]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                output[epoch][field] = {
                    "mean_count": statistics.fmean(values),
                    "patches_with_any": sum(value > 0 for value in values),
                    "fraction_with_any": sum(value > 0 for value in values) / len(values),
                }
    return output


def _paired_changes(paired):
    epochs = sorted({epoch for values in paired.values() for epoch in values})
    comparisons = list(zip(epochs, epochs[1:]))
    if len(epochs) > 2:
        comparisons.append((epochs[0], epochs[-1]))
    output = []
    for start, end in comparisons:
        for metric in PAIRED_METRICS:
            changes = []
            for rows in paired.values():
                if start not in rows or end not in rows:
                    continue
                initial = _number(rows[start].get(metric))
                final = _number(rows[end].get(metric))
                if initial is not None and final is not None:
                    changes.append(final - initial)
            if not changes:
                continue
            tolerance = 1e-12
            output.append(
                {
                    "start_epoch": start,
                    "end_epoch": end,
                    "metric": metric,
                    "samples": len(changes),
                    "mean_change": statistics.fmean(changes),
                    "median_change": statistics.median(changes),
                    "fraction_improved": sum(value < -tolerance for value in changes) / len(changes),
                    "fraction_unchanged": sum(abs(value) <= tolerance for value in changes) / len(changes),
                    "fraction_worsened": sum(value > tolerance for value in changes) / len(changes),
                }
            )
    return output


def _calibration(root, epochs):
    output = {}
    for epoch in epochs:
        path = Path(root) / f"epoch_{epoch:04d}" / "calibration.csv"
        if not path.is_file():
            continue
        rows = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["prediction_type"]].append(row)
        output[epoch] = {}
        for prediction_type, values in rows.items():
            total = sum(int(row["count"]) for row in values)
            weighted_error = 0.0
            for row in values:
                count = int(row["count"])
                confidence = _number(row["mean_probability"])
                frequency = _number(row["positive_fraction"])
                if count and confidence is not None and frequency is not None:
                    weighted_error += count * abs(confidence - frequency)
            output[epoch][prediction_type] = {
                "count": total,
                "ece": weighted_error / total if total else None,
            }
    return output


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(summary):
    epochs = summary["epochs"]
    metrics = summary["metric_summary"]
    occurrence = summary["candidate_occurrence"]
    calibration = summary["calibration"]
    paired = summary["paired_changes"]
    lines = [
        "# Validation corruption summary",
        "",
        f"- Unique validation patches: **{summary['unique_samples']:,}**",
        f"- Checkpoints: **{', '.join(str(epoch) for epoch in epochs)}**",
        f"- Rows analysed: **{summary['rows']:,}**",
        "",
        "## Candidate soft-distance components",
        "",
        "| Epoch | Node / GT node | Coordinate / GT node | Raw edge / candidate | Effective edge / candidate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for epoch in epochs:
        values = metrics[str(epoch)]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                epoch,
                _format(values["node_cost_per_gt_node"]["mean"]),
                _format(values["coordinate_cost_per_gt_node"]["mean"]),
                _format(values["edge_raw_cost_per_candidate"]["mean"], 6),
                _format(values["edge_effective_cost_per_candidate"]["mean"], 6),
            )
        )

    lines.extend(["", "## Mean confidence", "", "| Epoch | Matched node | Unmatched node | GT edge | Matched non-edge | Unmatched-endpoint edge |", "|---:|---:|---:|---:|---:|---:|"])
    for epoch in epochs:
        values = metrics[str(epoch)]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                epoch,
                _format(values["matched_node_probability_mean"]["mean"]),
                _format(values["unmatched_node_probability_mean"]["mean"]),
                _format(values["gt_edge_probability_mean"]["mean"]),
                _format(values["matched_nonedge_probability_mean"]["mean"], 6),
                _format(values["unmatched_endpoint_edge_probability_mean"]["mean"], 6),
            )
        )

    chosen_threshold = 0.5 if 0.5 in summary["thresholds"] else summary["thresholds"][0]
    lines.extend([
        "",
        f"## Error candidates at threshold {chosen_threshold:g}",
        "",
        "Each cell is `mean count per patch / percentage of patches with at least one`.",
        "",
        "| Epoch | Extra nodes | Missing nodes | False edges | Missing edges | False-loop edges | Duplicates | Subdivisions | Split nodes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    names = ("extra_nodes", "missing_nodes", "false_edges", "missing_edges", "false_loop_edges", "duplicate_candidates", "subdivision_node_candidates", "split_node_candidates")
    for epoch in epochs:
        values = occurrence[str(epoch)]
        cells = []
        for name in names:
            item = values.get(f"t{chosen_threshold:g}_{name}")
            cells.append(
                "n/a" if item is None else f"{item['mean_count']:.2f} / {100 * item['fraction_with_any']:.1f}%"
            )
        lines.append("| {} | {} |".format(epoch, " | ".join(cells)))

    lines.extend(["", "## Confidence calibration", "", "| Epoch | Node ECE | Raw-edge ECE | Effective-edge ECE |", "|---:|---:|---:|---:|"])
    for epoch in epochs:
        values = calibration.get(str(epoch), {})
        lines.append(
            "| {} | {} | {} | {} |".format(
                epoch,
                _format(values.get("node", {}).get("ece")),
                _format(values.get("edge_raw", {}).get("ece")),
                _format(values.get("edge_effective", {}).get("ece")),
            )
        )

    first, last = epochs[0], epochs[-1]
    lines.extend(["", f"## Paired change from epoch {first} to {last}", "", "Negative cost change means improvement.", "", "| Metric | Mean change | Median change | Improved patches | Worsened patches |", "|---|---:|---:|---:|---:|"])
    for item in paired:
        if item["start_epoch"] == first and item["end_epoch"] == last:
            lines.append(
                "| {} | {} | {} | {:.1f}% | {:.1f}% |".format(
                    item["metric"],
                    _format(item["mean_change"], 6),
                    _format(item["median_change"], 6),
                    100 * item["fraction_improved"],
                    100 * item["fraction_worsened"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.analysis_root
    comparison = root / "checkpoint_comparison.csv"
    if not comparison.is_file():
        raise FileNotFoundError(comparison)
    output = args.output_dir or (root / "summary")
    output.mkdir(parents=True, exist_ok=True)

    fieldnames, thresholds, by_epoch, paired = _read_comparison(comparison)
    metric_summary = _metric_summary(by_epoch, thresholds, fieldnames)
    occurrence = _candidate_occurrence(by_epoch, thresholds)
    paired_changes = _paired_changes(paired)
    calibration = _calibration(root, sorted(by_epoch))
    result = {
        "unique_samples": len(paired),
        "rows": sum(len(rows) for rows in by_epoch.values()),
        "epochs": sorted(by_epoch),
        "thresholds": thresholds,
        "metric_summary": metric_summary,
        "candidate_occurrence": occurrence,
        "paired_changes": paired_changes,
        "calibration": calibration,
    }
    # JSON stringifies integer dictionary keys; make that transformation before
    # rendering Markdown so both representations use the same structure.
    serializable = json.loads(json.dumps(result))
    (output / "summary.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_markdown(serializable), encoding="utf-8")

    metric_rows = []
    for epoch, values in sorted(metric_summary.items()):
        for metric, statistics_row in values.items():
            metric_rows.append({"epoch": epoch, "metric": metric, **statistics_row})
    _write_csv(output / "metric_summary.csv", metric_rows)
    _write_csv(output / "paired_changes.csv", paired_changes)
    print((output / "summary.md").read_text(encoding="utf-8"))
    print(f"Summary written to {output}")


if __name__ == "__main__":
    main()
