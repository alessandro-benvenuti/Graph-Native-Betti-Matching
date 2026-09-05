# Seven-recipe focal matrix

Each recipe is trained in two stages with seed `364505`:

1. 100 epochs of balanced mixed Plants + synthetic-MRI pretraining;
2. 600 epochs of synthetic-MRI specialization initialized from that recipe's
   lowest-validation-loss mixed checkpoint.

The seven recipes are baseline, node focal immediate, node focal curriculum,
edge focal immediate, edge focal curriculum, combined immediate, and combined
curriculum. Focal class weights are `[1, 1]`. Curricula keep gamma at zero
through 40% of each stage, ramp linearly to two at 70%, and remain at two.

Baseline edges use CE, matched pairs, and ratio upsampling to `0.15`.
Focal edges use no ratio upsampling and include up to 256 hard relation
negatives constructed from at most 32 unmatched queries with object
probability at least `0.25`. These candidates have weight one from the first
epoch in both immediate and curriculum runs; only gamma differs.

Mixed pretraining stores its current validation-loss-best checkpoint. MRI
specialization stores both validation-loss-best and task-metric-best
checkpoints. Every checkpoint contains model, optimizer, scheduler, epoch, and
iteration state, so each is a valid (possibly conservative) resume point. The
complete matrix is therefore bounded to twenty-one large checkpoint files.

At each validation interval, both stages also run the common graph metric
protocol on the complete synthetic-MRI validation split (currently 1,000
patches). Node/edge mAP and mAR, Betti errors, SMD, and graph counts are logged
under the W&B `metrics/` namespace and appended to
`validation-metrics.jsonl`. MRI specialization additionally stores
`best_metric_checkpoint.pt`, selected by maximum validation edge mAP, alongside
the loss-selected `best_checkpoint.pt`. Test data must not be used to choose
between them; the metric-selected checkpoint is the primary task model and the
loss-selected checkpoint is retained as a diagnostic.

The production matrix removes all dataset caps. Mixed pretraining makes all
25,923 Plants training patches and all 4,000 MRI training patches eligible,
and validates on all 6,421 Plants plus all 1,000 MRI validation patches. MRI
specialization uses all 4,000 MRI training patches and all 1,000 validation
patches. The balanced mixed sampler draws with replacement, so every training
record is eligible but an individual record is not guaranteed to appear in
every epoch. Test splits remain held out from training and model selection.

Submit from a Jean Zay login node after setting both dataset paths:

```bash
source cluster/jean_zay/env_h100.sh
export PLANTS_DATASET="$SCRATCH/datasets/plants_3d2cut/patches_3d"
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"

bash cluster/jean_zay/submit_focal_matrix_600.sh
```

Before submitting the matrix, verify the complete mixed/focal/checkpoint/W&B
path with one 45-minute development job:

```bash
wandb login --verify
bash cluster/jean_zay/submit_focal_matrix_smoke.sh
```

The smoke run uses its own `focal-matrix-600-smoke` W&B group and therefore is
not one of the fourteen scientific runs. A successful run performs four
optimizer steps, validates on four samples, writes `best_checkpoint.pt` and
`wandb-run.json`, and exits normally. Jean Zay compute nodes have no external
network access, so both matrix launchers force `WANDB_MODE=offline`. Upload
completed runs from a login node with:

```bash
bash cluster/jean_zay/sync_wandb_offline.sh "$GNBM_OUTPUT_DIR"
```

`env_h100.sh` loads the non-secret W&B entity and project from
`cluster/jean_zay/wandb_env.sh`. The matrix launcher assigns every pretraining
and finetuning run to the W&B group `focal-matrix-600-seed364505`. W&B login is
a one-time per-user cluster setup and its API key must remain outside Git.

Set `GNBM_MATRIX_DRY_RUN=1` to perform every preflight check and print the
matrix without calling `sbatch`. Pretraining and finetuning requests can be
configured independently with `GNBM_PRETRAIN_QOS`, `GNBM_PRETRAIN_WALLTIME`,
`GNBM_FINETUNE_QOS`, and `GNBM_FINETUNE_WALLTIME`.
