# Controlled MRI finetuning experiments

The first supported ablation matrix changes only graph-loss behavior. All four
runs inherit the same 3D model, synthetic-MRI split, augmentation, optimizer,
schedule, 10 epochs, dataset caps, validation interval, seed `364505`, and
initialization checkpoint.

| Experiment | Node loss | Edge loss | Unmatched edge candidates | H0/H1 |
|---|---|---|---|---|
| `baseline` | weighted CE | CE + ratio upsampling | off | off |
| `focal` | focal | focal | on | off |
| `betti` | weighted CE | CE + ratio upsampling | off | on |
| `focal_betti` | focal | focal | on | on |

The Betti runs use a pilot target coefficient of `0.1` for H0 and `0.1` for H1,
after 2 spatial-only epochs and a 5-epoch linear ramp. No historical trusted
configuration established these coefficients. They are therefore experimental
choices, recorded explicitly in YAML, and must not be described as baseline
settings. Validation evaluates enabled topology losses at their full target
coefficient from the beginning, making its objective stationary while training
ramps the regularizer.

## Required initialization

For a controlled comparison, all four runs must start from the exact same
plants + synthetic-MRI pretraining checkpoint. The MRI finetuning checkpoint
used for parity testing is not a valid common initialization because it has
already seen the finetuning task.

On the Magnolia cluster, the likely historical initialization is:

```text
/data/scavone/cross-dim_i2g_3d/runs/pretraining_plants_synth_baseline_20/models/checkpoint_epoch=50.pt
```

Confirm that checkpoint before treating the experiment as a reproduction.

## Smoke test

Exercise the combined focal + Betti path through real mixed loaders before
submitting pilot jobs:

```bash
python train.py --config configs/smoke_mixed_focal_betti.yaml
```

This is an integration test and should not be included in scientific results.

## Fixed-sample overfit check

Before the ablation pilot, run focal + Betti on eight fixed MRI training
samples for ten epochs. Augmentation is disabled, batch size is two, and focal
unmatched candidates plus H0/H1 are at full strength from the first epoch:

```bash
bash scripts/run_finetune_mri_experiment.sh \
  overfit \
  "$GNBM_PRETRAIN_CHECKPOINT"
```

Read `train.log` and confirm a clear reduction in training loss. The two
validation samples are a pipeline check, not evidence that such a tiny run
generalizes. If the training loss cannot decrease on this fixed set, do not
launch the four pilots.

## Running one experiment

```bash
bash scripts/run_finetune_mri_experiment.sh \
  baseline \
  "$GNBM_PRETRAIN_CHECKPOINT"
```

Replace `baseline` with `focal`, `betti`, or `focal_betti`. The launcher checks
the checkpoint and required environment variables and refuses to overwrite a
run that already has a resolved configuration. It also records the Git commit
and the initialization checkpoint path and SHA-256 checksum in the run folder.
Git is optional on the cluster: when metadata is unavailable, the launcher
records that fact and writes `source-manifest.sha256`, containing checksums for
the actual Python, YAML, shell, CUDA/C++, and compiled-extension files used by
the copied directory. If Git is available, its commit and working-tree status
are recorded as additional provenance, but a dirty tree does not block a run.

Each debugging pilot uses at most 256 MRI training samples and 32 validation
samples, runs for 10 epochs, validates every epoch, and uses
`checkpoint.policy: best_only`. Consequently each run directory contains at
most one approximately 1.5 GB checkpoint. Four completed pilots require about
6 GB for checkpoints rather than tens of gigabytes. The launcher also archives
stdout/stderr in `train.log`. W&B is the primary loss-curve record; `train.log`
remains the independent cluster record and fallback when tracking is offline or
disabled.

Run each experiment as a separate cluster job. Do not run the four pilot jobs
sequentially inside one allocation unless that allocation was sized for the
total runtime.

## Required records

Archive the following together for every run:

- Git commit/status when available, plus `source-manifest.sha256`;
- initialization-checkpoint absolute path and checksum;
- `resolved-config.yaml`;
- the single best checkpoint;
- stdout/stderr and cluster job metadata;
- W&B run ID from `wandb-run.json`;
- validation total loss at each validation epoch.

Validation total loss is suitable for checkpoint selection within a run. It is
not a cross-experiment performance metric because the four runs intentionally
optimize different objectives. Cross-experiment conclusions must use the same
task-level evaluation metrics on every selected checkpoint.

One seed is sufficient for pipeline validation, not for a final scientific
claim. After confirming the four pilot runs and evaluation behavior, repeat the
selected comparisons with multiple predeclared seeds.
