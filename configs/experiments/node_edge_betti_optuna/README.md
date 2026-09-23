# Node/edge Betti Optuna campaign

This directory defines a validation-only, paired hyperparameter study around the
existing `train.py`. The controller writes a concrete YAML per trial and monitors
the trainer's `validation-metrics.jsonl`; it does not duplicate the training loop
or change the mathematical Betti implementation.

## Scientific contract

Run one fixed control first. The control and every trial use the same model-only
checkpoint, seed, deterministic train subset, validation set, model, optimizer,
scheduler, and ordinary node/edge focal loss. The frozen control reference stores
a compatibility fingerprint and the dataset-manifest hash. Optimization refuses
an incompatible reference, and a completed trial is rejected if its manifest is
not byte-identical to the control manifest.

The test split is never loaded by this infrastructure. In particular, the ten
manually inspected test patches must not be used here. Final test results must not
be used to revise hyperparameters.

Beta errors alone are unsafe: deleting predicted edges can improve topology while
damaging the graph. Each post-warm-up validation epoch therefore minimizes the
normalized mean of beta0/beta1 absolute error plus a strong penalty for exceeding
the configured node-F1, edge-F1, node-mAP, or edge-mAP degradation tolerance.
The report independently enforces those four hard constraints. Among feasible
trials it minimizes the unpenalized normalized topology score, then breaks ties by
edge F1 and node F1. If none is feasible, it says so and reports the least total
constraint violation rather than presenting an infeasible winner.

The control reference is the **last control validation epoch**. A denominator
epsilon (`1e-8` by default) protects zero control beta errors. The selected trial
epoch is the best composite validation value after its Betti warm-up; epochs where
both Betti losses are inactive are not reported to Optuna. The report also records
the best edge-mAP epoch and last epoch. This first implementation does not save a
special best-composite checkpoint: selected hyperparameters are retrained later
from the common initialization.

Alpha stays at 0.5 because hybrid filtration already reduces to ordinary edge
confidence between matched nodes; tuning it would mostly add unmatched-node
degrees of freedom. H1 false-negative weight remains 1.0. Warm-up, gradual ramping,
small weights, and ordinary focal supervision remain fixed parts of the protocol.
The known critical-edge attribution limitation remains unresolved.

## Local infrastructure use

Install the additional dependency in the active project environment:

```bash
python -m pip install -r requirements/optuna.txt
```

The controller has two explicit modes. Both commands for one campaign must use the
same config and initial checkpoint:

```bash
python scripts/optimize_node_edge_betti.py control \
  --config configs/experiments/node_edge_betti_optuna/search.yaml \
  --output "$GNBM_OUTPUT_DIR" \
  --initial-weights "$GNBM_INITIAL_WEIGHTS"

python scripts/optimize_node_edge_betti.py optimize \
  --config configs/experiments/node_edge_betti_optuna/search.yaml \
  --output "$GNBM_OUTPUT_DIR" \
  --initial-weights "$GNBM_INITIAL_WEIGHTS"

python scripts/summarize_node_edge_betti_optuna.py \
  --output "$GNBM_OUTPUT_DIR" \
  --study-name node-edge-betti-optuna
```

Outputs include `control-reference.json`, `study.sqlite3`, exact trial configs,
unique `runs/trial_NNNN/` directories, partial or complete Optuna histories,
`summary.json`, `trials.csv`, and (only when one exists) `best-feasible.yaml`.

SQLite is supported only with one controller. The exclusive `.controller.lock`
prevents concurrent controllers for the same output directory. Do not start
multiple SQLite-backed controllers; parallel trials require a concurrency-safe
storage service and are outside this implementation. A normal SIGTERM/SIGINT
terminates the child and releases the lock. After an uncatchable kill, first verify
that no controller is running before manually removing a stale lock. On restart,
the study loads by name with `load_if_exists=True`, retains complete/pruned trials
and their artifacts, and conservatively marks stale RUNNING records failed.

## Tiny Jean Zay smoke test

The smoke test uses 256 deterministic training patches, 128 validation patches,
five epochs, validation every epoch, two trials, one A100, offline W&B, and no
pruning. Its metrics have **no scientific meaning**. It only checks imports,
SQLite persistence, control loading, generated YAML, real `train.py`, incremental
monitoring, completion, summary generation, and safe resumption.

On a Jean Zay login node:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
git switch features-gnbm-nodes-edges
git pull --ff-only origin features-gnbm-nodes-edges
source cluster/jean_zay/env_a100.sh
python -m pip install -r requirements/optuna.txt

export SYNTHETIC_MRI_DATASET="/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-node-edge-betti-optuna-smoke-a100"
export GNBM_INITIAL_WEIGHTS="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/pretrain_boundary_mixed_node_focal_seed364505/models/best_metric_checkpoint.pt"
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh smoke
```

Monitor and inspect (replace `JOBID`):

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -f "$WORK/logs/graph-native-betti-matching/a100/gnbm-betti-optuna-JOBID.out"
tail -f "$WORK/logs/graph-native-betti-matching/a100/gnbm-betti-optuna-JOBID.err"
find "$GNBM_OUTPUT_DIR" -maxdepth 3 -type f | sort
python scripts/summarize_node_edge_betti_optuna.py \
  --output "$GNBM_OUTPUT_DIR" --study-name node-edge-betti-optuna-smoke
```

To resume after timeout, export the same three paths and submit the same command:

```bash
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh smoke
```

The frozen control is reused and completed/pruned trials count toward the requested
two; they are not repeated. A failed/interrupted trial gets a new trial number and
its old directory is preserved.

## Future real search (do not launch yet)

Phase A runs the paired fixed control. Phase B starts with about 20 trials on a
representative deterministic training subset and preferably the complete validation
split, with enough epochs to finish warm-up/ramp and pruning only after several
observations. Phase C retrains the best feasible configuration plus one or two
nearby alternatives from the common initialization with a larger budget and at
least two seeds if possible. Only after validation freezes the protocol does Phase
D train on full data and evaluate the untouched test split once, reporting both
ordinary detection and topology metrics. Optuna is a selection mechanism, not
evidence by itself that the loss generalizes.
