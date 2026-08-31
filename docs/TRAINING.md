# Training

The supported entry point is `train.py`. Dataset and output paths remain
environment-specific, while scientific settings stay in version-controlled
YAML.

```bash
export PLANTS_DATASET=/data/scavone/plants_3d2cut/patches_3d
export SYNTHETIC_MRI_DATASET=/data/scavone/syntheticMRI/patches/syntheticMRI
export GNBM_OUTPUT_DIR=/data/benvenut/gnbm-runs

python train.py --config configs/pretrain_mixed.yaml
```

The baseline uses images as model input, AdamW, and the effective legacy
per-iteration polynomial schedule with linear warmup. The configured legacy
gradient clipping field was removed because the established trainer never
executed it. The configured minimum learning rate was also removed because the
legacy LambdaLR did not consume it. Likewise, inactive 3D encoder/decoder freeze
flags, automatic-resume metadata, and EMA fields were removed instead of being
kept as misleading no-ops. Focal unmatched-candidate mining scores candidates
with the live relation head and is configured inside the focal edge objective.

Operational overrides are limited to device, output directory, run name, batch
size, worker count, and checkpoint paths. The fully resolved YAML is written to
the run directory.

## Experiment tracking

W&B tracking is enabled by `tracking` in `configs/base.yaml`. Training logs all
loss components and learning rate against `train/iteration`, and validation
losses against `validation/epoch`. Each call records the metrics for one
training iteration together, avoiding artificial steps between loss components.
TensorBoard is not used.

Set credentials and operational destinations through the environment:

```bash
wandb login
source cluster/jean_zay/wandb_env.sh
export WANDB_RUN_GROUP=finetune-mri-ablation  # optional
```

The repository W&B destination is
`alessandrobenvenuti2002-politecnico-di-torino/focal-loss`. The shared shell
file contains no API key and may be overridden with `WANDB_ENTITY` or
`WANDB_PROJECT` before it is sourced.

Do not store `WANDB_API_KEY` in configuration files. `WANDB_MODE=offline`
buffers a run for later `wandb sync`; `WANDB_MODE=disabled` suppresses tracking
for a smoke test.

When `--resume` targets the same run directory, `wandb-run.json` supplies the
original W&B run ID and online metrics continue in that run. W&B does not
support resuming while offline; sync an offline run before continuing it
online. Training state still comes exclusively from the checkpoint passed to
`--resume`.

## Checkpoints

Every saved training checkpoint contains strict model, optimizer, scheduler,
epoch, and iteration state. Resume an interrupted run with:

```bash
python train.py \
  --config configs/pretrain_mixed.yaml \
  --resume /absolute/path/to/checkpoint_epoch=20.pt
```

Load a trusted legacy checkpoint as initialization without restoring its
optimizer or scheduler using:

```bash
python train.py \
  --config configs/finetune_synthetic_mri.yaml \
  --initial-weights "$GNBM_MRI_CHECKPOINT"
```

`best_checkpoint.pt` tracks minimum validation loss. When metric selection is
enabled, `best_metric_checkpoint.pt` independently tracks the configured
validation metric (edge mAP in the full-data recipes, **not recall**).
`best_only` keeps replaceable best slots plus `latest_checkpoint.pt`, rather
than accumulating an epoch archive. See below for the optional F1 slots.

## Epoch patience and independent F1 checkpoints

The full-dataset MRI fine-tuning recipes enable these settings; older recipes
default to disabled for backward compatibility:

```yaml
training:
  epochs: 250
  early_stopping:
    enabled: true
    monitor: edge_mAP
    mode: max
    patience_epochs: 50
    min_epochs: 0
    min_delta: 0.0
evaluation:
  training_metrics:
    enabled: true
    selection_metric: edge_mAP
    selection_mode: max
    save_best_checkpoint: true
    save_f1_checkpoints: true
  protocol:
    f1_iou_threshold: 0.5
```

Patience counts **epochs**, not validation calls. Full-data fine-tuning uses a
maximum of **250 total MRI epochs**, with **no minimum** (`min_epochs: 0`);
pretraining is unchanged. If the last improvement is at epoch 105, training
stops at epoch 155 absent further gains when validation is every five epochs.
Ties do not improve the monitor. A positive `min_delta`
requires a gain strictly greater than that amount relative to the last
significant improvement; small cumulative gains can eventually qualify.
With irregular intervals, stopping happens at the first validation at least
50 epochs after the last improvement. The first finite validation initializes
the reference. Missing/non-finite monitored metrics raise an error rather than
silently deciding convergence.

The default monitor remains edge AP for continuity with previous experiments.
To stop on edge F1, set `training.early_stopping.monitor: edge_f1`; this does not
change the independent AP checkpoint slot. `node_f1` is also supported. A
separate node-F1 gain does not reset an edge-AP/edge-F1 patience clock.

F1 uses **micro-aggregated TP/FP/FN over the whole validation split** at the
configured node and edge confidence thresholds (normally 0.5). Matching is
confidence-ordered, one-to-one box IoU at `f1_iou_threshold`; node boxes and
edge boxes use the existing evaluation geometry. All retained predictions
are included, without the AP protocol's top-40 cap. Node F1 and edge F1 are
`2 TP / (2 TP + FP + FN)`, not the harmonic mean of AP and AR. False positives
on empty-GT patches count; zero-denominator precision, recall, or F1 is defined
as zero. Edge F1 is **spatial edge-box detection F1**, not an endpoint-correct
adjacency metric and not Hungarian training-assignment F1. AP/AR are unchanged.
Changing thresholds changes F1 and its selected winner; choose thresholds on
validation, never on test.

| Checkpoint slot | Selection | Metadata |
|---|---|---|
| `best_checkpoint.pt` | Lowest validation loss | Training checkpoint state |
| `best_metric_checkpoint.pt` | Configured metric, normally edge AP | `best-metric.json` |
| `best_node_f1_checkpoint.pt` | Highest validation node F1 | `best-node-f1.json` |
| `best_edge_f1_checkpoint.pt` | Highest validation edge F1 | `best-edge-f1.json` |
| `latest_checkpoint.pt` | Latest scheduled save; always saved on early stop | Training checkpoint state |

Each is a **complete model and resume state**, not an independently swappable
head. F1 metadata includes the epoch, iteration, metric value, thresholds, and
all validation metrics at the winning epoch. Slots are replaced atomically;
if several win in one epoch they share a hard-linked payload when supported.
With `best_only`, there are at most five checkpoint slots, potentially about
6.8 GB if all five are from different epochs for the current model. Temporary
writes need additional headroom. The new F1 pair can require about 2.7 GB extra.

`validation-metrics.jsonl` and W&B include `node_precision`, `node_recall`,
`node_f1`, and their edge equivalents. JSON summaries additionally contain
`node_tp_total`, `node_fp_total`, `node_fn_total`, and edge totals; un-suffixed
count fields remain per-patch means. W&B records selected epochs under
`checkpoints/<metric>` in the run summary and logs the patience clock under
`stopping/*`. W&B checkpoint *values* are logged, not the large weight files.

On early stop, rank zero broadcasts the decision so all DDP workers stop
together after the final `latest_checkpoint.pt` is saved. `early-stopping.json`
records the clock and `training-status.json` reports `early_stopping` versus
`max_epochs`. The checkpoint stores patience and F1 winners, so interruption
resume does not reset them. Resume from `latest_checkpoint.pt` in the same run
directory. Changed stopping settings are rejected on strict resume; an already
early-stopped checkpoint does not restart itself. Older checkpoints lacking
patience/F1 state initialize those trackers at the first new validation.

**Budget is not continuation policy.** Full-data fine-tuning configs now cap at
250 total MRI epochs, with no early-stopping floor. This permits up to 150
additional epochs when continuing at epoch 100; patience may stop it earlier.
For new runs the scheduler is built for 250 epochs. Extending a completed
100-epoch run still requires an explicitly chosen continuation learning-rate
policy; simply changing `epochs` while restoring the old
scheduler is not a validated extension strategy. Loading `--initial-weights`
instead is a new fine-tuning experiment, not a true resume. Keep the existing
results intact and do not rerun pretraining merely to enable patience.

Focused CPU checks (the distributed test needs two local Gloo workers):

```bash
python -m unittest -v tests.test_early_stopping tests.test_f1_metrics \
  tests.test_training_selection tests.test_training tests.test_tracking
```

## Fast end-to-end cluster smoke run

The smoke configuration keeps the full model and loss path, but limits each
dataset to two training samples and one validation sample:

```bash
python train.py --config configs/smoke_mixed.yaml
```

It is an integration check, not a scientific experiment. The resolved
configuration and both the interval and best checkpoint should appear under
`$GNBM_OUTPUT_DIR/smoke_mixed/`.

## Cluster smoke tests

After building deformable attention, run the real CUDA forward and backward
tests:

```bash
python -m unittest -v tests.test_model_cluster.CudaModelTests
```

This covers the CUDA operator, model forward, complete baseline criterion
backward, and combined focal + H0/H1 Betti backward. Dataset integration tests
remain documented in `docs/DATA_LOADING.md`.

To extend the already-passing checkpoint forward comparison through the
baseline criterion, run:

```bash
python scripts/compare_model_forward.py \
  --legacy-root /data/benvenut/3d \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --compare-losses
```

The command fixes the relation-candidate random seed and compares the five
weighted baseline components, cardinality, and total loss in addition to model
outputs. Focal and Betti objectives are verified by their focused gradient
tests because they are opt-in deviations rather than baseline parity targets.
