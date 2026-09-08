# FGW branch pilot

This pilot initializes the model weights from the 50-epoch mixed Plants + MRI
node-focal + matched-edge-focal pretraining run, which used the Hungarian
matcher. It deliberately does not resume the pretraining optimizer or scheduler
because MRI fine-tuning is a new optimization stage.
The common MRI Hungarian trunk then runs through epoch 5 and writes a
full-state checkpoint. Four jobs resume that exact checkpoint (model, optimizer,
scheduler, RNG, data-loader generator, epoch, and iteration) and run through
epoch 15:

- Hungarian control;
- FGW with target alpha 0.4;
- FGW with target alpha 0.6;
- FGW with target alpha 0.8.

Each FGW branch ramps linearly from zero during global epochs 6--10, then stays
at its target through epoch 15. Alpha 0.9 is deliberately excluded from this
first training pilot because the frozen-model diagnostic is not evidence that
it improves optimization.

All stages retain `training.epochs: 50`. `stop_after_epoch` pauses execution at
5 or 15 without shortening the polynomial learning-rate schedule. Do not
replace it with a smaller `epochs` value.

The pilot trains on the same seeded random subset of 1,024 MRI graphs in every
branch and validates on all 200 validation graphs. This reduces training cost
by about 75% without making model selection depend on a very small validation
sample. The subset is appropriate for screening, not for a final result: the
winning FGW setting must be compared with Hungarian on the full dataset and
multiple seeds.

On the Jean Zay login node, after loading the FGW A100 environment, submit the
complete dependency graph with:

```bash
bash cluster/jean_zay/submit_fgw_branch_pilot.sh
```

By default, the trunk loads model weights from:

```text
/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/pretrain_full_mixed_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt
```

Set `GNBM_INITIAL_WEIGHTS` before launching to explicitly select another
pretraining checkpoint. The variable is removed before the dependent jobs are
submitted, because those jobs strictly resume the trunk's complete state.

The launcher refuses to proceed if any of the five run directories already
exists. The continuation jobs are submitted with an `afterok` dependency and
start in parallel only after the trunk succeeds.

## Final replication

After the seed-364505 pilot, submit two independent training replications with:

```bash
bash cluster/jean_zay/submit_fgw_replications.sh
```

This creates Hungarian trunks for experiment seeds 364506 and 364507. Each
trunk has three dependent continuations: Hungarian, FGW alpha 0.4, and FGW
alpha 0.8. The dataset subset remains fixed with `sample_cap_seed: 364505`, so
the replications measure training-order and augmentation variability rather
than dataset-membership variability. Alpha 0.6 is excluded because the first
pilot left it largely dominated by 0.4, except for beta-1 error.
