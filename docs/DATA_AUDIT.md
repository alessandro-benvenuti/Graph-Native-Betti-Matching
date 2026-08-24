# SyntheticMRI candidate-grid audit

`scripts/audit_synthetic_mri_grid.py` reads the original SyntheticMRI
`raw/`, `seg/`, `graphs/`, and split CSV without modifying them. It writes
only reports and never creates image or graph patches.

The default `full` grid uses the inherited `64^3` patch, five-voxel padding,
and `0.25` overlap. The corresponding source crop is `54^3`, the stride is 40,
and both ends of every scan axis are included. Ground truth describes every
candidate after the grid has been fixed; it never controls candidate inclusion.

The optional `legacy` grid reproduces the inherited random offset and cyclic
starting position. It enumerates the entire candidate grid rather than stopping
at a quota because the exact historical generator invocation is not verified.

## One-volume smoke audit

Run this inside an allocated Jean Zay compute job, not on a login node:

```bash
data_root=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI

python -u scripts/audit_synthetic_mri_grid.py \
  --root "$data_root" \
  --splits "$data_root/splits.csv" \
  --output-dir "$data_root/results/patch_audit_stride40_smoke" \
  --grid both \
  --patient-id 1
```

## Complete audit

After checking the smoke reports, submit the complete audit in a sufficiently
long CPU allocation:

```bash
data_root=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI

python -u scripts/audit_synthetic_mri_grid.py \
  --root "$data_root" \
  --splits "$data_root/splits.csv" \
  --output-dir "$data_root/results/patch_audit_stride40" \
  --grid both
```

Use `--grid full` when only the deployment-style grid is required. The full
audit loads one scan at a time, but graph cropping is CPU-intensive; retain the
stdout/stderr log together with the reports.

## Reports

- `scan_inventory.csv`: scan shape, spacing, dtype, intensity statistics,
  foreground occupancy, and full-volume graph size;
- `candidate_audit.csv`: one row per candidate and grid, including foreground,
  node/edge counts, GT-SNR, coordinate validity, and the inherited rejection
  reason;
- `audit_summary_by_split.csv`: empty, graph-empty, `<3`-node, disagreement,
  and inherited-acceptance counts and percentages by grid and legacy split;
- `audit_summary.json`: configuration, patient list, provenance notes, and raw
  summary counters.

The audit deliberately does not simulate the historical quota. Quota effects
can be reconstructed later from the ordered `legacy` rows once the historical
target counts and per-volume cap are confirmed.
