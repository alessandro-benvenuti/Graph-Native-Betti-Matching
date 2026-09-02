# Interactive SyntheticMRI graph visualization

`scripts/visualize_graph_prediction_3d.py` creates a self-contained Plotly HTML
report for one exported evaluation prediction. The report embeds its JavaScript,
so it can be generated on Jean-Zay, copied elsewhere, and opened without a
network connection. It contains orthogonal MRI slices, a segmentation surface,
the ground-truth graph, the confidence-filtered prediction, provenance, counts,
thresholds, and the metrics stored in `predictions.json`. Click a legend entry
to toggle its complete layer.

The script reads existing prediction exports only. A checkpoint or training
directory is not enough; run `evaluate.py` first if `predictions.json` has not
been exported.

## Coordinates and identifiers

Both VTP ground-truth points and exported `nodes_dhw` are interpreted in the
repository's canonical **normalized D/H/W** order. They are converted to plot
coordinates only at the visualization boundary:

```text
x = w * W
y = h * H
z = d * D
```

They must not be interpreted directly as VTK X/Y/Z. The evaluator's sequential
`sample_id` (such as `sample_000123`) is display metadata; dataset files are
always resolved with `source_sample_id` (such as `sample_000109_0021`). Use
`--evaluation-id` for the former or `--sample-id` for the latter.

## Jean-Zay usage

Load the repository environment, then set paths explicitly. `$SCRATCH` is only
a shell convenience and is never required while importing project modules.

Find prediction exports:

```bash
find "$SCRATCH/experiments/gnbm" -type f -name predictions.json -print
```

List all samples in an export (omit both sample selectors):

```bash
python scripts/visualize_graph_prediction_3d.py \
  --predictions "$EVALUATION_DIR/predictions.json"
```

List the 20 worst samples by combined beta-0/beta-1 absolute error, then by
combined node/edge count error:

```bash
python - "$EVALUATION_DIR/predictions.json" <<'PY'
import json, sys
records = json.load(open(sys.argv[1], encoding="utf-8"))
def value(record, name):
    return float((record.get("metrics") or {}).get(name) or 0)
def key(record):
    beta = value(record, "beta0_absolute_error") + value(record, "beta1_absolute_error")
    count = value(record, "node_count_absolute_error") + value(record, "edge_count_absolute_error")
    return beta, count
for record in sorted(records, key=key, reverse=True)[:20]:
    beta, count = key(record)
    print(f"{record['source_sample_id']}\t{record['sample_id']}\tbeta={beta:g}\tcount={count:g}")
PY
```

Generate one interactive report:

```bash
python scripts/visualize_graph_prediction_3d.py \
  --dataset-root "$SCRATCH/datasets/syntheticMRI/new_patches" \
  --split test \
  --predictions "$EVALUATION_DIR/predictions.json" \
  --sample-id sample_000109_0021 \
  --output-dir "$SCRATCH/experiments/gnbm/visualizations" \
  --node-threshold 0.5 \
  --edge-threshold 0.5 \
  --error-analysis
```

`--error-analysis` uses a one-to-one minimum-distance node assignment gated by
`--match-distance` (default `0.1` in normalized D/H/W space). It marks unmatched
nodes, edges incident to unmatched predicted nodes, edges between mapped nodes
that are absent from GT, and GT edges missing between mapped endpoints. These
are explicitly visualization-time diagnostics. They are not the official
IoU-based evaluation matches and are not the training Hungarian assignment.

Generate a gallery for the ten worst beta/count samples:

```bash
mkdir -p "$SCRATCH/experiments/gnbm/visualizations/worst"
python - "$EVALUATION_DIR/predictions.json" <<'PY' |
import json, sys
records = json.load(open(sys.argv[1], encoding="utf-8"))
names = ("beta0_absolute_error", "beta1_absolute_error", "node_count_absolute_error", "edge_count_absolute_error")
score = lambda r: sum(float((r.get("metrics") or {}).get(n) or 0) for n in names)
for record in sorted(records, key=score, reverse=True)[:10]:
    print(record["source_sample_id"])
PY
while IFS= read -r sample; do
  python scripts/visualize_graph_prediction_3d.py \
    --dataset-root "$SCRATCH/datasets/syntheticMRI/new_patches" \
    --split test \
    --predictions "$EVALUATION_DIR/predictions.json" \
    --sample-id "$sample" \
    --output-dir "$SCRATCH/experiments/gnbm/visualizations/worst" \
    --error-analysis
done
```

For a storage-bounded comparison that first chooses ten reproducible random
test patches, runs inference for only those patches through the baseline,
node-focal, and node+matched-edge-focal full-data checkpoints, and then renders
all 30 HTML reports, use:

```bash
bash scripts/evaluate_random_patches_3d.sh \
  /path/to/baseline/best_metric_checkpoint.pt \
  /path/to/node_focal/best_metric_checkpoint.pt \
  /path/to/node_edge_focal/best_metric_checkpoint.pt \
  "$SCRATCH/experiments/gnbm/random10_three_models"
```

The script records the shared IDs in `source_sample_ids.txt` and refuses to
overwrite an existing output directory. Override its defaults with
`GNBM_RANDOM_PATCH_COUNT`, `GNBM_RANDOM_PATCH_SEED`,
`GNBM_EVALUATION_BATCH_SIZE`, and `GNBM_WORKERS`. Production model inference
uses one allocated GPU; the subsequent Plotly HTML rendering is CPU-only.

The default is `--html`. Add `--png` for a static image; this requires a
Plotly-compatible Kaleido installation and a working headless rendering
backend. Use `--no-html --png` for PNG only. Layer defaults can be changed with
`--show-mri`/`--hide-mri`, `--show-segmentation`/`--hide-segmentation`,
`--show-ground-truth`/`--hide-ground-truth`, and
`--show-prediction`/`--hide-prediction`.
