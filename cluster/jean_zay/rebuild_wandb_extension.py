#!/usr/bin/env python3
"""Build and optionally upload gap-free epoch-1-to-tail W&B histories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from rebuild_wandb_campaign import (
    ENTITY,
    HISTORY_NAME,
    MANIFEST_NAME,
    PROJECT,
    atomic_text,
    checkpoint_header,
    load_json,
    load_jsonl_last,
    require_exact,
    scan_segment,
    selection_summary,
    upload_run,
)


SOURCE_EPOCH = 500
MAX_EPOCH = 750
GROUP = "boundary-gamma-extension-750-seed364505-canonical"
RUNS = (
    (
        "finetune_boundary_mri750_baseline_tail250_seed364505",
        "finetune_boundary_mri500_baseline_seed364505",
    ),
    (
        "finetune_boundary_mri750_node_focal_tail250_seed364505",
        "finetune_boundary_mri500_node_focal_seed364505",
    ),
    (
        "finetune_boundary_mri750_node_edge_focal_g05_tail250_seed364505",
        "finetune_boundary_mri500_node_edge_focal_g05_seed364505",
    ),
)


def read_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                epoch = int(row["epoch"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                raise RuntimeError(f"malformed {path}:{line_number}: {error}") from error
            if epoch in rows:
                raise RuntimeError(f"duplicate epoch {epoch} in {path}")
            rows[epoch] = row
    return rows


def prepare_run(run_dir: Path, source_dir: Path) -> dict:
    status = load_json(run_dir / "training-status.json")
    if status.get("reason") not in {"max_epochs", "early_stopping"}:
        raise RuntimeError(f"run is not terminal: {run_dir.name}: {status}")
    final_epoch = int(status["epoch"])
    if not SOURCE_EPOCH < final_epoch <= MAX_EPOCH:
        raise RuntimeError(
            f"invalid terminal epoch for {run_dir.name}: {final_epoch}"
        )
    if status.get("reason") == "max_epochs" and final_epoch != MAX_EPOCH:
        raise RuntimeError(
            f"max_epochs run ends at {final_epoch}, expected {MAX_EPOCH}"
        )
    if not (run_dir / "training-complete").is_file():
        raise RuntimeError(f"missing training-complete marker: {run_dir.name}")

    latest_epoch, latest_iteration = checkpoint_header(
        run_dir / "models" / "latest_checkpoint.pt"
    )
    if latest_epoch != final_epoch or int(status["iteration"]) != latest_iteration:
        raise RuntimeError(
            f"checkpoint/status mismatch in {run_dir.name}: "
            f"checkpoint=({latest_epoch},{latest_iteration}) status="
            f"({status.get('epoch')},{status.get('iteration')})"
        )
    if latest_iteration % final_epoch:
        raise RuntimeError(
            f"cannot infer fixed iterations per epoch: {latest_iteration}/{final_epoch}"
        )
    iterations_per_epoch = latest_iteration // final_epoch

    source_rows = read_rows(source_dir / HISTORY_NAME)
    source_manifest = load_json(source_dir / MANIFEST_NAME)
    if int(source_manifest.get("final_epoch", -1)) != SOURCE_EPOCH:
        raise RuntimeError(
            f"invalid source canonical manifest: {source_dir / MANIFEST_NAME}"
        )
    require_exact(
        f"{source_dir.name} canonical prefix",
        source_rows,
        range(1, SOURCE_EPOCH + 1),
    )
    for epoch, row in source_rows.items():
        for key in ("train/epoch", "performance/epoch"):
            if int(row.get(key, -1)) != epoch:
                raise RuntimeError(f"invalid {key} at source epoch {epoch}")
        if int(row.get("train/iteration", -1)) != epoch * iterations_per_epoch:
            raise RuntimeError(f"invalid source iteration at epoch {epoch}")

    performance = load_jsonl_last(run_dir / "performance.jsonl")
    metrics = load_jsonl_last(run_dir / "validation-metrics.jsonl")
    require_exact(
        f"{run_dir.name} performance",
        performance,
        range(1, final_epoch + 1),
    )
    metric_epochs = set(range(5, final_epoch + 1, 5))
    if final_epoch not in metric_epochs:
        metric_epochs.add(final_epoch)
    require_exact(f"{run_dir.name} metrics", metrics, metric_epochs)

    train_candidates: dict[int, list[dict]] = defaultdict(list)
    validation = {}
    segment_report = []
    transaction_files = sorted(
        path
        for path in (run_dir / "wandb").glob("offline-run-*/run-*.wandb*")
        if path.is_file()
    )
    if not transaction_files:
        raise RuntimeError(f"no W&B transactions: {run_dir.name}")
    for number, transaction in enumerate(transaction_files, 1):
        train, segment_validation, records, tail_error = scan_segment(transaction)
        for epoch, candidate in train.items():
            if epoch > SOURCE_EPOCH:
                train_candidates[epoch].append(candidate)
        validation.update({
            epoch: row
            for epoch, row in segment_validation.items()
            if epoch > SOURCE_EPOCH
        })
        segment_report.append({
            "number": number,
            "path": str(transaction.relative_to(run_dir)),
            "bytes": transaction.stat().st_size,
            "records": records,
            "tail_error": tail_error,
            "completed_extension_train_epochs": sorted(
                epoch for epoch in train if epoch > SOURCE_EPOCH
            ),
            "extension_validation_epochs": sorted(
                epoch for epoch in segment_validation if epoch > SOURCE_EPOCH
            ),
        })

    extension_epochs = set(range(SOURCE_EPOCH + 1, final_epoch + 1))
    train = {}
    incomplete = {}
    for epoch in extension_epochs:
        candidates = train_candidates.get(epoch, [])
        complete = [
            candidate
            for candidate in candidates
            if candidate["rows"] == iterations_per_epoch
        ]
        if complete:
            train[epoch] = complete[-1]["values"]
        else:
            incomplete[epoch] = [candidate["rows"] for candidate in candidates]
    if incomplete:
        raise RuntimeError(
            f"{run_dir.name}: no complete W&B replay for epochs {incomplete}"
        )
    extension_metric_epochs = {
        epoch for epoch in metric_epochs if epoch > SOURCE_EPOCH
    }
    require_exact(
        f"{run_dir.name} extension validation loss",
        validation,
        extension_metric_epochs,
    )

    rows = [source_rows[epoch] for epoch in range(1, SOURCE_EPOCH + 1)]
    for epoch in range(SOURCE_EPOCH + 1, final_epoch + 1):
        row = {
            "epoch": epoch,
            "train/epoch": epoch,
            "train/iteration": epoch * iterations_per_epoch,
            **train[epoch],
            "performance/epoch": epoch,
        }
        row.update({
            f"performance/{key}": value
            for key, value in performance[epoch].items()
            if key != "epoch" and isinstance(value, (int, float))
        })
        if epoch in validation:
            row.update({
                key: value
                for key, value in validation[epoch].items()
                if key.startswith("validation/")
            })
        if epoch in metrics:
            row.update({
                f"metrics/{key}": value
                for key, value in metrics[epoch].items()
                if key not in {"epoch", "iteration"}
                and isinstance(value, (int, float))
            })
            row["metrics/epoch"] = epoch
            row["metrics/iteration"] = int(metrics[epoch]["iteration"])
        rows.append(row)

    atomic_text(
        run_dir / HISTORY_NAME,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
    )
    metadata = load_json(run_dir / "wandb-run.json")
    manifest = {
        "schema_version": 1,
        "run_name": run_dir.name,
        "source_run_id": metadata["id"],
        "source_prefix_run": source_dir.name,
        "source_prefix_canonical_id": source_manifest["canonical_run_id"],
        "canonical_run_id": "c" + str(metadata["id"]),
        "final_epoch": final_epoch,
        "latest_iteration": latest_iteration,
        "iterations_per_epoch": iterations_per_epoch,
        "history": HISTORY_NAME,
        "history_rows": len(rows),
        "metric_epochs": sorted(metric_epochs),
        "validation_epochs": sorted(metric_epochs),
        "selection_summary": selection_summary(run_dir),
        "transactions": segment_report,
    }
    atomic_text(
        run_dir / MANIFEST_NAME,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="epoch-750 continuation root")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--group", default=GROUP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    prepared = []
    for run_name, source_name in RUNS:
        print(f"\n===== {run_name} =====")
        run_dir = root / run_name
        manifest = prepare_run(run_dir, source_root / source_name)
        prepared.append((run_dir, manifest))
        print(
            f"  verified gap-free local history: epochs 1..{manifest['final_epoch']}"
        )

    if not args.upload:
        print("\nNo network writes were made. Re-run with --upload after review.")
        return 0

    print(f"\nUploading to {args.entity}/{args.project}, group={args.group}")
    for run_dir, manifest in prepared:
        print(f"\n===== upload {run_dir.name} =====")
        upload_run(run_dir, manifest, args.entity, args.project, args.group)
    print("\nAll extension runs uploaded and verified with gap-free histories.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
