#!/usr/bin/env python3
"""Build and optionally upload gap-free canonical W&B campaign histories.

The original offline runs were interrupted and resumed many times.  This tool
does not mutate or re-sync them.  It reconstructs one authoritative epoch row
at a time, choosing the latest complete replay of each training epoch and using
the durable JSONL metric files for validation metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import yaml

from audit_boundary_gamma_campaign import RUNS


ENTITY = "alessandrobenvenuti2002-politecnico-di-torino"
PROJECT = "focal-loss"
GROUP = "boundary-gamma-sweep-500-seed364505-canonical"
HISTORY_NAME = "canonical-wandb-history.jsonl"
MANIFEST_NAME = "canonical-wandb-manifest.json"


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Mapping:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_jsonl_last(path: Path) -> dict[int, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                epoch = int(record["epoch"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                raise RuntimeError(f"malformed {path}:{line_number}: {error}") from error
            # Resume replays are intentional.  The later completed execution is
            # the state that produced the retained checkpoint.
            result[epoch] = dict(record)
    return result


def checkpoint_header(path: Path) -> tuple[int, int]:
    import torch

    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    payload = torch.load(str(path), **options)
    if not isinstance(payload, Mapping) or "net" not in payload:
        raise RuntimeError(f"invalid checkpoint: {path}")
    return int(payload["epoch"]), int(payload["iteration"])


def item_key(item) -> str:
    key = getattr(item, "key", "")
    if not key and getattr(item, "nested_key", None):
        key = "/".join(item.nested_key)
    return key


def item_value(item):
    try:
        return json.loads(getattr(item, "value_json", ""))
    except (TypeError, ValueError):
        return None


def numeric_payload(history) -> dict[str, float | int]:
    payload = {}
    for item in history.item:
        key = item_key(item)
        value = item_value(item)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[key] = value
    return payload


def scan_segment(path: Path):
    """Return complete-epoch train aggregates and validation loss rows."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    last_values: dict[int, dict[str, float]] = defaultdict(dict)
    train_rows: dict[int, int] = defaultdict(int)
    completed_epochs: set[int] = set()
    validation: dict[int, dict] = {}
    records = 0
    tail_error = None
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
            if record.WhichOneof("record_type") != "history":
                continue
            payload = numeric_payload(record.history)
            if "train/epoch" in payload:
                epoch = int(payload["train/epoch"])
                train_rows[epoch] += 1
                for key, value in payload.items():
                    if not key.startswith("train/"):
                        continue
                    if key in {"train/epoch", "train/iteration"}:
                        continue
                    sums[epoch][key] += float(value)
                    counts[epoch][key] += 1
                    last_values[epoch][key] = float(value)
            if "performance/epoch" in payload:
                completed_epochs.add(int(payload["performance/epoch"]))
            if "validation/epoch" in payload:
                validation[int(payload["validation/epoch"])] = payload
    except Exception as error:
        # Slurm may truncate only the final record.  Coverage checks below
        # decide whether the readable prefix is sufficient.
        tail_error = f"{type(error).__name__}: {error}"
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    aggregates = {}
    for epoch in completed_epochs:
        row = {}
        for key, total in sums.get(epoch, {}).items():
            count = counts[epoch][key]
            if count:
                row[key] = (
                    last_values[epoch][key]
                    if key == "train/learning_rate"
                    else total / count
                )
        aggregates[epoch] = {
            "values": row,
            "rows": train_rows.get(epoch, 0),
        }
    return aggregates, validation, records, tail_error


def require_exact(label: str, observed, expected) -> None:
    observed = set(observed)
    expected = set(expected)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"{label}: missing={missing} extra={extra}")


def selection_summary(run_dir: Path) -> dict:
    summary = {}
    for filename in ("best-metric.json", "best-node-f1.json", "best-edge-f1.json"):
        record = load_json(run_dir / filename)
        summary[f"checkpoints/{record['metric']}"] = dict(record)
    return summary


def prepare_run(run_dir: Path, configured_epochs: int) -> dict:
    status = load_json(run_dir / "training-status.json")
    if status.get("reason") not in {"max_epochs", "early_stopping"}:
        raise RuntimeError(f"run is not terminal: {run_dir.name}: {status}")
    final_epoch = int(status["epoch"])
    if status.get("reason") == "max_epochs" and final_epoch != configured_epochs:
        raise RuntimeError(
            f"{run_dir.name}: max_epochs status ends at {final_epoch}, expected {configured_epochs}"
        )
    if not (run_dir / "training-complete").is_file():
        raise RuntimeError(f"missing training-complete marker: {run_dir.name}")

    latest_epoch, latest_iteration = checkpoint_header(
        run_dir / "models" / "latest_checkpoint.pt"
    )
    if latest_epoch != final_epoch or int(status["iteration"]) != latest_iteration:
        raise RuntimeError(
            f"latest checkpoint/status mismatch in {run_dir.name}: "
            f"checkpoint=({latest_epoch},{latest_iteration}) status="
            f"({status.get('epoch')},{status.get('iteration')})"
        )
    if latest_iteration % final_epoch:
        raise RuntimeError(
            f"cannot infer a fixed iterations-per-epoch in {run_dir.name}: "
            f"{latest_iteration}/{final_epoch}"
        )
    iterations_per_epoch = latest_iteration // final_epoch

    performance = load_jsonl_last(run_dir / "performance.jsonl")
    metrics = load_jsonl_last(run_dir / "validation-metrics.jsonl")
    expected_epochs = set(range(1, final_epoch + 1))
    expected_metric_epochs = set(range(5, final_epoch + 1, 5))
    if final_epoch not in expected_metric_epochs:
        expected_metric_epochs.add(final_epoch)
    require_exact(f"{run_dir.name} performance", performance, expected_epochs)
    require_exact(f"{run_dir.name} validation metrics", metrics, expected_metric_epochs)

    transaction_files = sorted(
        path
        for path in (run_dir / "wandb").glob("offline-run-*/run-*.wandb*")
        if path.is_file()
    )
    if not transaction_files:
        raise RuntimeError(f"no W&B transactions: {run_dir.name}")

    # Keep every candidate until we know it is complete.  Later complete
    # replays override earlier ones; partial timeout epochs are never selected.
    train_candidates: dict[int, list[dict]] = defaultdict(list)
    validation = {}
    segment_report = []
    for number, transaction in enumerate(transaction_files, 1):
        train, segment_validation, records, tail_error = scan_segment(transaction)
        for epoch, candidate in train.items():
            train_candidates[epoch].append(candidate)
        validation.update(segment_validation)
        segment_report.append({
            "number": number,
            "path": str(transaction.relative_to(run_dir)),
            "bytes": transaction.stat().st_size,
            "records": records,
            "tail_error": tail_error,
            "completed_train_epochs": sorted(train),
            "validation_epochs": sorted(segment_validation),
        })

    train = {}
    incomplete = {}
    for epoch in range(1, final_epoch + 1):
        candidates = train_candidates.get(epoch, [])
        complete = [item for item in candidates if item["rows"] == iterations_per_epoch]
        if not complete:
            incomplete[epoch] = [item["rows"] for item in candidates]
            continue
        train[epoch] = complete[-1]["values"]
    if incomplete:
        raise RuntimeError(
            f"{run_dir.name}: no complete W&B train replay for epochs {incomplete}"
        )
    require_exact(f"{run_dir.name} validation loss", validation, expected_metric_epochs)

    rows = []
    for epoch in range(1, final_epoch + 1):
        row = {
            "epoch": epoch,
            "train/epoch": epoch,
            "train/iteration": epoch * iterations_per_epoch,
        }
        row.update(train[epoch])
        row.update({
            f"performance/{key}": value
            for key, value in performance[epoch].items()
            if key != "epoch" and isinstance(value, (int, float))
        })
        row["performance/epoch"] = epoch
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
                if key not in {"epoch", "iteration"} and isinstance(value, (int, float))
            })
            row["metrics/epoch"] = epoch
            row["metrics/iteration"] = int(metrics[epoch]["iteration"])
        rows.append(row)

    history_path = run_dir / HISTORY_NAME
    atomic_text(
        history_path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )
    metadata = load_json(run_dir / "wandb-run.json")
    manifest = {
        "schema_version": 1,
        "run_name": run_dir.name,
        "source_run_id": metadata["id"],
        "canonical_run_id": "c" + str(metadata["id"]),
        "final_epoch": final_epoch,
        "latest_iteration": latest_iteration,
        "iterations_per_epoch": iterations_per_epoch,
        "history": HISTORY_NAME,
        "history_rows": len(rows),
        "metric_epochs": sorted(metrics),
        "validation_epochs": sorted(validation),
        "selection_summary": selection_summary(run_dir),
        "transactions": segment_report,
    }
    atomic_text(
        run_dir / MANIFEST_NAME,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def read_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cloud_key_set(run, key: str) -> set[int]:
    values = set()
    for row in run.scan_history(keys=[key], page_size=1000):
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.add(int(value))
    return values


def verify_cloud_run(run, manifest: Mapping) -> None:
    final_epoch = int(manifest["final_epoch"])
    all_epochs = set(range(1, final_epoch + 1))
    metric_epochs = set(int(value) for value in manifest["metric_epochs"])
    checks = {
        "epoch": all_epochs,
        "train/epoch": all_epochs,
        "performance/epoch": all_epochs,
        "validation/epoch": metric_epochs,
        "metrics/epoch": metric_epochs,
    }
    for key, expected in checks.items():
        require_exact(f"cloud {key} for {manifest['run_name']}", cloud_key_set(run, key), expected)


def wait_for_cloud_run(path: str, manifest: Mapping, *, attempts: int = 18, delay: int = 10):
    """Wait for W&B's eventually consistent history index, then verify it."""
    import wandb

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            run = wandb.Api(timeout=120).run(path)
            verify_cloud_run(run, manifest)
            if run.state != "finished":
                raise RuntimeError(f"state={run.state}, expected finished")
            return run
        except Exception as error:
            last_error = error
            if attempt < attempts:
                print(
                    f"  cloud indexing not complete ({attempt}/{attempts}): {error}; "
                    f"retrying in {delay}s"
                )
                time.sleep(delay)
    raise RuntimeError(
        f"cloud verification did not converge for {path}: {last_error}"
    )


def upload_run(run_dir: Path, manifest: Mapping, entity: str, project: str, group: str):
    import wandb

    rows = read_history(run_dir / str(manifest["history"]))
    expected = set(range(1, int(manifest["final_epoch"]) + 1))
    canonical_id = str(manifest["canonical_run_id"])
    path = f"{entity}/{project}/{canonical_id}"
    api = wandb.Api(timeout=120)
    existing = None
    try:
        existing = api.run(path)
    except Exception:
        pass
    # A just-finished W&B run can be visible before its history index is fully
    # populated.  Verify/wait before interpreting a short prefix as missing;
    # otherwise a retry could append duplicate epochs to a complete run.
    if existing is not None and existing.state == "finished":
        try:
            verified = wait_for_cloud_run(path, manifest, attempts=6, delay=10)
        except Exception as error:
            raise RuntimeError(
                f"existing canonical run is finished but could not be proven complete; "
                f"refusing to append: {path}: {error}"
            ) from error
        print(f"  already complete: {verified.url}")
        return verified.url

    uploaded = cloud_key_set(existing, "epoch") if existing is not None else set()
    if uploaded:
        maximum = max(uploaded)
        require_exact(f"existing canonical prefix for {run_dir.name}", uploaded, range(1, maximum + 1))
        if uploaded == expected:
            verify_cloud_run(existing, manifest)
            print(f"  already complete: {existing.url}")
            return existing.url
        start_epoch = maximum + 1
    else:
        start_epoch = 1

    config = yaml.safe_load((run_dir / "resolved-config.yaml").read_text(encoding="utf-8"))
    config["canonical_reconstruction"] = {
        "source_run_id": manifest["source_run_id"],
        "history_rows": manifest["history_rows"],
        "deduplication": "latest complete replay per epoch",
        "validation_metrics_source": "validation-metrics.jsonl",
    }
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        id=canonical_id,
        resume="allow",
        name=run_dir.name + "-canonical",
        tags=("gnbm", "canonical-reconstruction", "gap-free"),
        config=config,
        mode="online",
    )
    run.define_metric("epoch")
    for namespace in ("train", "performance", "validation", "metrics"):
        run.define_metric(namespace + "/*", step_metric="epoch")
    for name, summary in (
        ("metrics/node_mAP", "max"),
        ("metrics/node_mAR", "max"),
        ("metrics/edge_mAP", "max"),
        ("metrics/edge_mAR", "max"),
        ("metrics/node_f1", "max"),
        ("metrics/edge_f1", "max"),
        ("metrics/beta0_absolute_error", "min"),
        ("metrics/beta1_absolute_error", "min"),
        ("metrics/smd", "min"),
        ("validation/total", "min"),
    ):
        run.define_metric(name, step_metric="epoch", summary=summary, overwrite=True)
    for row in rows:
        epoch = int(row["epoch"])
        if epoch >= start_epoch:
            run.log(row, step=epoch)
    for key, value in manifest["selection_summary"].items():
        run.summary[key] = value
    run.summary["canonical/source_run_id"] = manifest["source_run_id"]
    run.summary["canonical/final_epoch"] = manifest["final_epoch"]
    url = run.url
    run.finish(exit_code=0)

    verified = wait_for_cloud_run(path, manifest)
    print(f"  verified gap-free epochs 1..{manifest['final_epoch']}: {url}")
    return url


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload prepared canonical runs; without this flag only local files are built",
    )
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--group", default=GROUP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifests = []
    print(f"Preparing canonical histories under {root}")
    for run_name, configured_epochs in RUNS:
        print(f"\n===== {run_name} =====")
        manifest = prepare_run(root / run_name, configured_epochs)
        manifests.append((root / run_name, manifest))
        tail_warnings = sum(bool(item["tail_error"]) for item in manifest["transactions"])
        print(
            f"  prepared {manifest['history_rows']} contiguous epoch rows; "
            f"transactions={len(manifest['transactions'])}; "
            f"truncated_tail_warnings={tail_warnings}"
        )
        print(f"  canonical id={manifest['canonical_run_id']}")

    print("\nAll ten canonical histories passed local coverage checks.")
    if not args.upload:
        print("No network writes were made. Re-run with --upload after reviewing this output.")
        return 0

    print(f"\nUploading to {args.entity}/{args.project}, group={args.group}")
    for run_dir, manifest in manifests:
        print(f"\n===== upload {run_dir.name} =====")
        upload_run(run_dir, manifest, args.entity, args.project, args.group)
    print("\nAll canonical W&B runs uploaded and verified with contiguous epoch coverage.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
