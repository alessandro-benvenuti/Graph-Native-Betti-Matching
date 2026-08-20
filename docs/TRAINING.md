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

The default checkpoint policy writes interval checkpoints and
`best_checkpoint.pt`. As in the effective legacy evaluator, "best" means the
lowest mean validation total loss; SMD and AP were not key checkpoint metrics.
Storage-constrained runs may use `best_only` to overwrite a single
`best_checkpoint.pt`, or `none` to disable checkpoint output entirely.

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
