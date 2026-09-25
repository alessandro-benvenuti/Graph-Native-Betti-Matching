#!/usr/bin/env python3
"""Audit checkpoint and W&B history integrity for the boundary gamma sweep."""

from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping


ENTITY = "alessandrobenvenuti2002-politecnico-di-torino"
PROJECT = "focal-loss"

RUNS = (
    ("pretrain_boundary_mixed_baseline_seed364505", 100),
    ("pretrain_boundary_mixed_node_focal_seed364505", 100),
    ("pretrain_boundary_mixed_node_edge_focal_g05_seed364505", 100),
    ("pretrain_boundary_mixed_node_edge_focal_g10_seed364505", 100),
    ("pretrain_boundary_mixed_node_edge_focal_g20_seed364505", 100),
    ("finetune_boundary_mri500_baseline_seed364505", 500),
    ("finetune_boundary_mri500_node_focal_seed364505", 500),
    ("finetune_boundary_mri500_node_edge_focal_g05_seed364505", 500),
    ("finetune_boundary_mri500_node_edge_focal_g10_seed364505", 500),
    ("finetune_boundary_mri500_node_edge_focal_g20_seed364505", 500),
)

RECORDS = {
    "best-metric.json": "best_metric_checkpoint.pt",
    "best-node-f1.json": "best_node_f1_checkpoint.pt",
    "best-edge-f1.json": "best_edge_f1_checkpoint.pt",
}

# Observed Slurm segmentation for this campaign.  These counts let the audit
# detect an accidentally deleted offline directory even when the remaining
# transactions themselves are healthy.
EXPECTED_TRANSACTIONS = {
    "pretrain_boundary_mixed_baseline_seed364505": 1,
    "pretrain_boundary_mixed_node_focal_seed364505": 1,
    "pretrain_boundary_mixed_node_edge_focal_g05_seed364505": 1,
    "pretrain_boundary_mixed_node_edge_focal_g10_seed364505": 1,
    "pretrain_boundary_mixed_node_edge_focal_g20_seed364505": 1,
    # Ten original segments plus the final epoch-493..500 recovery segment.
    "finetune_boundary_mri500_baseline_seed364505": 11,
    "finetune_boundary_mri500_node_focal_seed364505": 9,
    "finetune_boundary_mri500_node_edge_focal_g05_seed364505": 10,
    "finetune_boundary_mri500_node_edge_focal_g10_seed364505": 10,
    "finetune_boundary_mri500_node_edge_focal_g20_seed364505": 10,
}


class Reporter:
    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def error(self, message: str) -> None:
        self.errors += 1
        print(f"  ERROR: {message}")

    def warning(self, message: str) -> None:
        self.warnings += 1
        print(f"  WARNING: {message}")

    def ok(self, message: str) -> None:
        print(f"  OK: {message}")


def compact(values: Iterable[int], limit: int = 16) -> str:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return "none"
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    if len(ranges) > limit:
        return ",".join(ranges[:limit]) + f",... ({len(ordered)} values)"
    return ",".join(ranges)


def read_json(path: Path, reporter: Reporter) -> Mapping | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        reporter.error(f"cannot parse {path.name}: {error}")
        return None
    if not isinstance(value, Mapping):
        reporter.error(f"{path.name} is not a JSON object")
        return None
    return value


def read_jsonl_epochs(path: Path, reporter: Reporter) -> list[int]:
    if not path.is_file():
        reporter.error(f"missing {path.name}")
        return []
    epochs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                epoch = int(record["epoch"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                reporter.error(f"malformed {path.name}:{line_number}: {error}")
                continue
            epochs.append(epoch)
    return epochs


def report_epoch_series(
    label: str,
    epochs: list[int],
    expected: set[int],
    reporter: Reporter,
) -> None:
    counts = Counter(epochs)
    observed = set(counts)
    missing = expected - observed
    extra = observed - expected
    duplicates = {epoch: count for epoch, count in counts.items() if count > 1}
    regressions = sum(right < left for left, right in zip(epochs, epochs[1:]))
    if missing:
        reporter.error(f"{label} missing epochs: {compact(missing)}")
    if extra:
        reporter.warning(f"{label} has unexpected epochs: {compact(extra)}")
    if duplicates:
        reporter.warning(
            f"{label} replayed {len(duplicates)} epoch(s): "
            + ", ".join(f"{epoch}x{count}" for epoch, count in sorted(duplicates.items()))
        )
    if regressions:
        reporter.warning(f"{label} contains {regressions} resume-order regression(s)")
    if not missing and not extra:
        reporter.ok(
            f"{label} covers {len(expected)} expected epochs "
            f"({min(expected) if expected else '-'}..{max(expected) if expected else '-'})"
        )


def torch_load(path: Path):
    import torch

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(str(path), **options)


def validate_checkpoint(
    path: Path,
    reporter: Reporter,
    cache: dict[tuple[int, int], tuple[int, int]],
    deep: bool,
) -> tuple[int, int] | None:
    if not path.is_file():
        reporter.error(f"missing checkpoint: models/{path.name}")
        return None
    stat = path.stat()
    if stat.st_size == 0:
        reporter.error(f"empty checkpoint: models/{path.name}")
        return None
    identity = (stat.st_dev, stat.st_ino)
    if identity in cache:
        return cache[identity]
    try:
        payload = torch_load(path)
    except Exception as error:
        reporter.error(f"cannot deserialize models/{path.name}: {error}")
        return None
    required = {"net", "optimizer", "scheduler", "epoch", "iteration", "trainer_state"}
    if not isinstance(payload, Mapping):
        reporter.error(f"models/{path.name} payload is not a mapping")
        return None
    missing = required - set(payload)
    if missing:
        reporter.error(f"models/{path.name} is missing keys: {sorted(missing)}")
        return None
    try:
        result = (int(payload["epoch"]), int(payload["iteration"]))
    except (TypeError, ValueError) as error:
        reporter.error(f"models/{path.name} has invalid epoch/iteration: {error}")
        return None
    if deep:
        import torch

        nonfinite = []
        for name, value in payload["net"].items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not bool(torch.isfinite(value).all()):
                    nonfinite.append(name)
        if nonfinite:
            reporter.error(
                f"models/{path.name} contains non-finite tensors: {nonfinite[:8]}"
            )
    cache[identity] = result
    return result


def history_value(item):
    raw = getattr(item, "value_json", "")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def scan_transaction(path: Path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    epochs = {key: [] for key in ("performance/epoch", "validation/epoch", "metrics/epoch")}
    records = 0
    exit_codes = []
    error = None
    store = DataStore()
    try:
        store.open_for_scan(str(path))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            records += 1
            kind = record.WhichOneof("record_type")
            if kind == "history":
                for item in record.history.item:
                    key = getattr(item, "key", "")
                    if not key and getattr(item, "nested_key", None):
                        key = "/".join(item.nested_key)
                    if key not in epochs:
                        continue
                    value = history_value(item)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        epochs[key].append(int(value))
            elif kind == "exit":
                exit_codes.append(int(record.exit.exit_code))
    except Exception as caught:  # Truncated Slurm tails are reported, not hidden.
        error = f"{type(caught).__name__}: {caught}"
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return epochs, records, exit_codes, error


def cloud_epochs(run, key: str) -> list[int]:
    values = []
    for row in run.scan_history(keys=[key], page_size=1000):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(int(value))
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="campaign output/checkpoint root")
    parser.add_argument("--cloud", action="store_true", help="compare local coverage with wandb.ai")
    parser.add_argument(
        "--scan-transactions",
        action="store_true",
        help="fully scan every offline .wandb transaction (reads about 9 GB)",
    )
    parser.add_argument(
        "--deep-checkpoints",
        action="store_true",
        help="also reject non-finite tensors in every distinct checkpoint",
    )
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    reporter = Reporter()
    checkpoint_cache: dict[tuple[int, int], tuple[int, int]] = {}
    seen_run_ids: dict[str, str] = {}
    cloud_api = None
    if args.cloud:
        import wandb

        cloud_api = wandb.Api(timeout=120)

    print(f"Campaign root: {root}")
    print(f"Cloud comparison: {args.cloud}")
    print(f"Transaction scan: {args.scan_transactions}")
    print(f"Deep checkpoint scan: {args.deep_checkpoints}")

    for run_name, configured_epochs in RUNS:
        print("\n" + "=" * 100)
        print(run_name)
        run_dir = root / run_name
        if not run_dir.is_dir():
            reporter.error(f"missing run directory: {run_dir}")
            continue

        metadata = read_json(run_dir / "wandb-run.json", reporter)
        status = read_json(run_dir / "training-status.json", reporter)
        completion = (run_dir / "training-complete").is_file()
        run_id = metadata.get("id") if metadata else None
        print(f"  identity={run_id or 'missing'} completion_marker={completion}")
        if run_id:
            if run_id in seen_run_ids:
                reporter.error(
                    f"W&B run ID {run_id} is also used by {seen_run_ids[run_id]}"
                )
            else:
                seen_run_ids[run_id] = run_name

        models = run_dir / "models"
        latest = validate_checkpoint(
            models / "latest_checkpoint.pt", reporter, checkpoint_cache, args.deep_checkpoints
        )
        final_epoch = int(status["epoch"]) if status and "epoch" in status else (
            latest[0] if latest else configured_epochs
        )
        if final_epoch > configured_epochs:
            reporter.error(
                f"final epoch {final_epoch} exceeds configured maximum {configured_epochs}"
            )
        if latest and latest[0] != final_epoch:
            reporter.error(
                f"latest checkpoint epoch {latest[0]} != training status epoch {final_epoch}"
            )
        if completion and not status:
            reporter.error("completion marker exists but training-status.json is missing")
        if not completion:
            reporter.warning("training-complete marker is missing")
            if latest and latest[0] < configured_epochs:
                reporter.error(
                    f"training is incomplete: latest checkpoint epoch {latest[0]} "
                    f"< configured maximum {configured_epochs}"
                )
        if status:
            reporter.ok(
                f"training status reason={status.get('reason')} epoch={final_epoch} "
                f"iteration={status.get('iteration')}"
            )
        if latest:
            reporter.ok(
                f"latest checkpoint loads: epoch={latest[0]} iteration={latest[1]}"
            )

        for record_name, checkpoint_name in RECORDS.items():
            record = read_json(run_dir / record_name, reporter)
            checkpoint = validate_checkpoint(
                models / checkpoint_name, reporter, checkpoint_cache, args.deep_checkpoints
            )
            if record is None:
                reporter.error(f"missing selection record: {record_name}")
                continue
            if checkpoint and int(record.get("epoch", -1)) != checkpoint[0]:
                reporter.error(
                    f"{record_name} epoch {record.get('epoch')} != "
                    f"{checkpoint_name} epoch {checkpoint[0]}"
                )
            elif checkpoint:
                reporter.ok(
                    f"{record.get('metric')} checkpoint loads at epoch {checkpoint[0]}"
                )

        temporary = list(models.glob("*.tmp")) + list(models.glob(".runtime_states/*"))
        if temporary:
            reporter.warning(f"interrupted checkpoint temporary files: {len(temporary)}")

        performance = read_jsonl_epochs(run_dir / "performance.jsonl", reporter)
        metric_epochs = read_jsonl_epochs(run_dir / "validation-metrics.jsonl", reporter)
        expected_performance = set(range(1, final_epoch + 1))
        expected_metrics = set(range(5, final_epoch + 1, 5))
        # A clean terminal epoch is evaluated even when it is not aligned with
        # the interval.  An interrupted latest checkpoint is not: for example,
        # epoch 492 in a 500-epoch run should only expect metrics through 490.
        if status and final_epoch and final_epoch not in expected_metrics:
            expected_metrics.add(final_epoch)
        report_epoch_series("local performance history", performance, expected_performance, reporter)
        report_epoch_series("local validation metrics", metric_epochs, expected_metrics, reporter)

        transaction_files = sorted(
            path
            for path in (run_dir / "wandb").glob("offline-run-*/run-*.wandb*")
            if path.is_file()
        )
        reporter.ok(
            f"offline W&B transactions={len(transaction_files)} "
            f"bytes={sum(path.stat().st_size for path in transaction_files)}"
        )
        expected_transactions = EXPECTED_TRANSACTIONS[run_name]
        if len(transaction_files) != expected_transactions:
            reporter.error(
                f"expected {expected_transactions} offline W&B transactions, "
                f"found {len(transaction_files)}"
            )
        empty_transactions = [path for path in transaction_files if path.stat().st_size == 0]
        if empty_transactions:
            reporter.error(
                "empty offline W&B transactions: "
                + ", ".join(path.name for path in empty_transactions)
            )

        transaction_union = {
            "performance/epoch": set(),
            "validation/epoch": set(),
            "metrics/epoch": set(),
        }
        if args.scan_transactions:
            for number, transaction in enumerate(transaction_files, 1):
                epochs, records, exits, error = scan_transaction(transaction)
                for key, values in epochs.items():
                    transaction_union[key].update(values)
                ranges = " ".join(
                    f"{key.split('/')[0]}={compact(values)}"
                    for key, values in epochs.items()
                )
                print(
                    f"  transaction {number:02d}: records={records} exits={exits or '-'} "
                    f"{ranges}"
                )
                if records == 0:
                    reporter.error(f"empty/unreadable transaction: {transaction}")
                if error:
                    reporter.warning(f"transaction tail issue in {transaction.name}: {error}")
            comparisons = (
                ("performance/epoch", set(performance)),
                ("metrics/epoch", set(metric_epochs)),
                ("validation/epoch", set(metric_epochs)),
            )
            for key, local in comparisons:
                missing = local - transaction_union[key]
                if missing:
                    reporter.error(f"offline W&B {key} missing local epochs: {compact(missing)}")
                else:
                    reporter.ok(f"offline W&B {key} covers all authoritative local epochs")

        if args.cloud:
            if not run_id:
                reporter.error("cannot query cloud without a run ID")
                continue
            try:
                cloud_run = cloud_api.run(f"{args.entity}/{args.project}/{run_id}")
            except Exception as error:
                reporter.error(f"cloud run not found: {error}")
                continue
            print(f"  cloud state={cloud_run.state} url={cloud_run.url}")
            comparisons = (
                ("performance/epoch", set(performance)),
                ("metrics/epoch", set(metric_epochs)),
                ("validation/epoch", set(metric_epochs)),
            )
            for key, local in comparisons:
                try:
                    online_values = cloud_epochs(cloud_run, key)
                except Exception as error:
                    reporter.error(f"cannot read cloud {key}: {error}")
                    continue
                online = set(online_values)
                missing = local - online
                extra = online - local
                duplicates = len(online_values) - len(online)
                if missing:
                    reporter.error(f"cloud {key} missing epochs: {compact(missing)}")
                if extra:
                    reporter.warning(f"cloud {key} has unexpected epochs: {compact(extra)}")
                if duplicates:
                    reporter.warning(f"cloud {key} has {duplicates} duplicate epoch records")
                if not missing and not extra:
                    reporter.ok(f"cloud {key} matches authoritative local coverage")

    print("\n" + "=" * 100)
    print(f"AUDIT RESULT: errors={reporter.errors} warnings={reporter.warnings}")
    if reporter.errors:
        print("FAIL: do not treat the W&B campaign as complete yet.")
        return 1
    print("PASS: all requested integrity checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
