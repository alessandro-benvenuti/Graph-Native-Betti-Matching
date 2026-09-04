# Boundary-data edge-focal gamma sweep

Five controlled recipes use seed `364505`:

1. baseline node weighted cross-entropy and edge cross-entropy;
2. unweighted node focal loss with gamma 2 and edge cross-entropy;
3. node focal gamma 2 plus matched--matched edge focal gamma 0.5;
4. node focal gamma 2 plus matched--matched edge focal gamma 1.0;
5. node focal gamma 2 plus matched--matched edge focal gamma 2.0.

Every recipe runs 100 epochs of balanced mixed pretraining using 25,900 Plants
and a deterministic 4,000-patch MRI subset. It then initializes MRI-only
fine-tuning on every materialized MRI training patch for at most 500 epochs.
Fine-tuning stops after 50 epochs without an improvement in validation edge
mAP. Validation runs every five epochs.

Both stages retain rolling checkpoints for best edge mAP, best node F1, and
best edge F1. Fine-tuning is initialized from the pretraining edge-mAP winner.
Betti H0 and H1 losses are explicitly disabled with zero weight. Their
evaluation errors are still reported as metrics.

The dataset loader reads the materialized
`train/val/test/{raw,seg,vtp}` directories. Split CSV files are provenance
metadata and are not read during training.

Submit all five dependent pipelines on Jean Zay with:

```bash
bash cluster/jean_zay/submit_boundary_gamma_sweep_500.sh
```

Set `GNBM_BOUNDARY_SWEEP_DRY_RUN=1` to validate the complete matrix and print
the intended jobs without calling `sbatch`.

