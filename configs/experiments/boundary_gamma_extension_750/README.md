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
