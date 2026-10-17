# Boundary gamma 500->750 continuation

This experiment continues three selected MRI fine-tuning runs from epoch 500:
the baseline, node focal, and node plus matched--matched edge focal gamma 0.5.
The source runs remain unchanged.

The continuation restores model and AdamW moment state, resets early-stopping
patience, and deliberately starts a new polynomial learning-rate tail at
`1e-5`, decaying to zero over at most 250 epochs. Validation remains every five
epochs, with 50-epoch edge-mAP patience. Best edge-mAP, node-F1, and edge-F1
checkpoints and their provenance records are seeded from the source runs, so a
tail checkpoint replaces them only after a genuine improvement.

The first continuation segment explicitly migrates from two to four A100s,
discarding only the old two-rank RNG state. Model weights and AdamW moments are
retained, and batch 8 per GPU preserves the original global batch of 32. Later
segments strictly resume with four A100s. The launcher submits six resumable
20-hour segments per variant; surplus segments exit after the completion marker
is created.

```bash
bash cluster/jean_zay/submit_boundary_gamma_extension_750.sh
```

Set `GNBM_EXTENSION_DRY_RUN=1` to validate and print the plan without creating
directories or submitting jobs.

For the deadline fallback on two H100s, use `GNBM_EXTENSION_HARDWARE=h100`.
This selects batch 16 per GPU, keeping global batch 32 and world size 2,
uses a separate `gnbm-boundary-gamma-extension-750-h100` output root, and
submits eight six-hour segments per recipe.

Each new offline W&B run is seeded from the source run's validated canonical
epoch-1-to-500 history. After all three continuations finish, first reconstruct
and validate the complete histories locally, then upload them:

```bash
python cluster/jean_zay/rebuild_wandb_extension.py \
  "$SCRATCH/checkpoints/gnbm-boundary-gamma-extension-750-a100" \
  --source-root "$SCRATCH/checkpoints/gnbm-boundary-gamma-sweep-500-a100"

python cluster/jean_zay/rebuild_wandb_extension.py \
  "$SCRATCH/checkpoints/gnbm-boundary-gamma-extension-750-a100" \
  --source-root "$SCRATCH/checkpoints/gnbm-boundary-gamma-sweep-500-a100" \
  --upload
```

Do not use raw `wandb sync` for the final reported runs. The reconstruction
rejects incomplete epochs from interrupted Slurm segments and verifies that
the uploaded `focal-loss` histories contain every epoch without gaps.
