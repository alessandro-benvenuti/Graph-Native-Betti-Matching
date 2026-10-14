# Validation prediction analysis

This analysis uses only the configured **validation** split, without augmentation
or batch-normalization recalibration. It does not train a model. All checkpoints
use the same ordered patch manifest. The Jean Zay launcher defaults to the
corrected `new_patches_boundary` root and joins every sample to its complete
`patch_index.csv` provenance row.

Run a bounded smoke job first:

```bash
export GNBM_ANALYSIS_OUTPUT="$WORK/analysis/vascular-corruptions-smoke"
export GNBM_ANALYSIS_MAX_SAMPLES=10
sbatch cluster/jean_zay/analyze_validation_predictions_a100.slurm
```

Inspect the log and outputs, then run the full validation set in a new directory:

```bash
unset GNBM_ANALYSIS_MAX_SAMPLES
export GNBM_ANALYSIS_OUTPUT="$WORK/analysis/vascular-corruptions-full"
sbatch cluster/jean_zay/analyze_validation_predictions_a100.slurm
```

The job contains the supplied epoch 200, 300, 400, and 500 checkpoints. Optional
overrides are `GNBM_ANALYSIS_CONFIG`, `GNBM_ANALYSIS_DATASET`,
`GNBM_ANALYSIS_BATCH_SIZE`, and `GNBM_REPO_DIR`. Batch size defaults to 4; only
increase it after the smoke run establishes available A100 memory.

Completed checkpoint directories contain `COMPLETED.json` and are skipped when a
job is resubmitted. If a job stopped inside a checkpoint, inspect the partial
directory and set `GNBM_ANALYSIS_REPLACE_INCOMPLETE=1` to regenerate that
checkpoint. This flag never replaces a completed checkpoint.

## Outputs

- `validation_manifest.json`: ordered patch IDs, source paths, patient ID, patch
  origin, and the other `patch_index.csv` fields.
- `resolved_config.yaml` and `analysis_settings.json`: the exact resolved run
  configuration and analysis parameters.
- `edge_pairs.pt`: the shared canonical ordering of all 7,140 unordered query
  pairs.
- Per checkpoint, `raw/shard_*.pt`: lossless CPU tensors for all 120 query
  coordinates and logits, both directional edge logits, the GT graph, Hungarian
  assignment, and model input shape. Target adjacency and displacement are
  exactly reconstructible from the GT and assignment, so they are not duplicated
  in every record. Edge probability is recovered as the softmax of the mean
  directional logits, matching production inference.
- `nodes.csv.gz`: matching, confidence, displacement vector and magnitude,
  nearest GT node and segment, segment projection, boundary distance, GT degree,
  raw/effective soft degree, and duplicate information.
- `edge_events.csv.gz`: every GT edge and each non-GT edge above the event
  threshold. It includes raw/effective confidence, endpoint matching, geometric
  length, whether the endpoints already share a GT path, and whether the active
  edge belongs to a predicted cycle. Lower-confidence edges remain in raw shards.
- `patches.csv` and the root `checkpoint_comparison.csv`: candidate node,
  coordinate, raw-edge, and effective-edge costs; normalized variants;
  confidence quantiles; and thresholded topology and error counts.
- `calibration.csv`: reliability bins for node confidence, raw edge confidence,
  and confidence-weighted edge presence.
- `ranked_cases.csv`: the 100 largest-error patches for each candidate cost.
- `plots/`: deterministic overlays for the first requested patches, and PNG
  histograms of each candidate cost.

Coordinates are normalized D/H/W. `model_input_shape` converts them to model-input
voxels; original voxels require the original volume shape. All query slots remain
in the export. Unmatched nodes receive no coordinate penalty.

Matching follows the configured matcher and is recomputed independently at every
checkpoint. Query indices are not treated as persistent node identities. The run
aborts if any GT graph exceeds the 120-query capacity. Candidate cost variants are
exported separately and do not choose the eventual embedded-loss definition.

## Interpretation

Duplicate, split-node, and subdivision counts are geometric **candidates**, not
confirmed diagnoses. A subdivision candidate is an active, unmatched degree-two
node near a GT segment. A split-node candidate is an active, nearby unmatched node
that is disconnected from the query matched to its nearest GT node. A false loop
edge is a false active edge that belongs to a cycle in the thresholded prediction;
this does not identify a unique causal edit.

Use `ranked_cases.csv`, raw shards, and visual overlays to validate these candidate
definitions before choosing corruption frequencies. The final test set remains
untouched.

## Detailed CPU-only inspection

Once the full export is complete, inspect the spatial context, complete score
distributions, subdivision lengths, and exact co-occurrences without rerunning
the model:

```bash
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching"
export GNBM_ANALYSIS_ROOT="$WORK/analysis/vascular-corruptions-full"
export GNBM_DETAILED_ANALYSIS_OUTPUT="$GNBM_ANALYSIS_ROOT/detailed_summary"
mkdir -p "$WORK/logs/graph-native-betti-matching"

sbatch \
  --output="$WORK/logs/graph-native-betti-matching/gnbm-corruption-inspect-%j.out" \
  --error="$WORK/logs/graph-native-betti-matching/gnbm-corruption-inspect-%j.out" \
  cluster/jean_zay/inspect_validation_corruptions_cpu.slurm
```

The job reads the existing raw shards on CPU and writes `detailed_summary.md`,
`detailed_summary.json`, fine-bin `distribution_histograms.csv`, exact
`subdivision_candidates.csv`, `cooccurrence_pairwise.csv`, and
`cooccurrence_patterns.csv`. The edge distributions cover every candidate pair;
they are not truncated at the edge-event export threshold.
