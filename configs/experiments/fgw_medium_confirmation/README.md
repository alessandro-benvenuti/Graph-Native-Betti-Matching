# Medium-scale FGW confirmation

This experiment uses a fixed seeded subset of 8,192 MRI training patches and
the complete validation set. For each of three training seeds, a five-epoch
Hungarian trunk initialized from the focal mixed-pretraining checkpoint forks
into Hungarian and FGW alpha-0.4 continuations through epoch 100. FGW ramps from
zero to 0.4 during epochs 6--10.

All jobs use the same 100-epoch polynomial-scheduler horizon. `stop_after_epoch`
ends the trunks at epoch 5 and both comparison branches at epoch 100.
Only the final archived checkpoint and a replace-in-place recovery checkpoint
are retained in each run directory.

Submit all three dependency trees with:

```bash
bash cluster/jean_zay/submit_fgw_medium_confirmation.sh
```

After completion, compare the paired epoch-100 results with:

```bash
python scripts/summarize_fgw_medium_confirmation.py \
  --output-root "$GNBM_OUTPUT_DIR"
```
