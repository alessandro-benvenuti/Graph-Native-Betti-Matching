# Inference and evaluation

`evaluate.py` is the supported checkpoint-evaluation entry point. It performs
model inference, converts retained object queries into an undirected graph,
computes graph metrics, optionally recalibrates BatchNorm statistics, exports
predictions as JSON, and writes headless PNG comparisons.

## Baseline-compatible inference

The decoder deliberately retains the effective baseline decisions:

- object queries are retained by foreground-class argmax unless
  `evaluation.node_threshold` is set;
- every unordered pair of retained nodes is considered;
- relation logits are evaluated in both endpoint orders and averaged before
  softmax;
- edges are retained by class argmax unless `evaluation.edge_threshold` is
  set;
- non-maximum suppression is not applied by the trusted baseline protocol.

Each exported prediction contains normalized D/H/W nodes, all six predicted
center-size box values, node/query scores, local undirected edges, and edge
scores.

Model-to-graph parity against the original repository can be checked without
involving the corrected metrics:

```bash
python scripts/compare_model_forward.py \
  --legacy-root /data/benvenut/3d \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --compare-inference
```

This compares retained nodes, all node-box values, node scores, local edge
indices, and relation scores after applying each repository's own inference
implementation.

## Metrics

The evaluator reports:

- node and edge mAP/mAR over IoU 0.50:0.05:0.95 with at most 40 detections;
- node and edge micro precision/recall/F1 at fixed confidence thresholds and
  `protocol.f1_iou_threshold` (default IoU 0.5), without a detection-count cap;
- Street Mover Distance (SMD);
- absolute beta-0 and beta-1 errors;
- target and predicted beta-0/beta-1 values;
- target and predicted node/edge counts and their absolute errors.

As in the baseline evaluator, detections are pooled by confidence within each
contiguous fold before AP/AR is computed; the reported value is the mean across
folds. It is not an average of independently computed per-image AP values.
Metrics that are mathematically undefined are serialized as JSON `null` and
excluded from aggregate means (for example, edge AP for a sample with no target
edges, or SMD when either graph has no non-zero-length edge).

The fixed target node size (`0.2`), edge half-width (`0.1`), IoU thresholds,
SMD settings, fold count, and detection cap are explicit under
`evaluation.protocol` in `configs/base.yaml`.

F1 uses summed TP/FP/FN over all patches, rather than averaging per-patch F1 or
combining AP with AR. Greedy score-ordered one-to-one box matching counts
duplicates as false positives, including on patches with no GT. Zero
denominators return zero. Edge F1 measures spatial edge-box agreement, not
endpoint-correct graph adjacency. Precision/recall use all predictions retained
by inference; node and edge confidence cuts are saved in the resolved config.
See [training selection](TRAINING.md#epoch-patience-and-independent-f1-checkpoints)
for checkpoint selection and patience behavior.

The implementation intentionally fixes four logic errors in the old report
script:

1. node center-size boxes are converted to corners exactly once;
2. every sample is included when the dataset size is not divisible by five;
3. SMD samples the endpoints named by each edge, rather than `nodes[i-j-1]`;
4. plotting uses the repository's declared D/H/W coordinate order.

Consequently newly computed metrics are the supported values for comparisons
between refactored experiments, but they must not be presented as numerically
identical to CSV files produced by the buggy legacy evaluator.

## Magnolia command

For a small validation evaluation with four plots:

```bash
python evaluate.py \
  --config configs/finetune_synthetic_mri.yaml \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --output-dir "$GNBM_OUTPUT_DIR/evaluation_mri" \
  --dataset synthetic_mri \
  --split val \
  --max-samples 32 \
  --batch-size 4 \
  --visualizations 4
```

The dataset root must contain the requested `val/` or `test/` split. Evaluation
never enables random data augmentation. It sorts discovered source sample IDs
before applying `--max-samples`, so a smoke-test subset is filesystem-independent.

Outputs are:

- `resolved-config.yaml`: exact evaluation configuration;
- `metadata.json`: checkpoint path/size, split, sample count, and calibration;
- `summary.json`: aggregate means and contiguous-fold standard deviations;
- `predictions.json`: graph predictions, source IDs, and per-sample metrics;
- `per-patch-metrics.csv`: one readable metrics row per source patch;
- `plots/sample_*.png`: optional segmentation/target/prediction comparisons.

Use `--no-export-predictions` when only aggregate metrics are needed.

## BatchNorm calibration

BatchNorm calibration is disabled by default. For a historical checkpoint whose
documented protocol requires it, set the number of calibration batches either
in YAML or operationally:

```bash
python evaluate.py \
  --config configs/finetune_synthetic_mri.yaml \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --output-dir "$GNBM_OUTPUT_DIR/evaluation_mri_bncal" \
  --dataset synthetic_mri \
  --split test \
  --bn-calibration-batches 100
```

Calibration uses the configured dataset's validation split, updates BatchNorm
running statistics without changing learned parameters, disables dropout, and
then restores evaluation mode. It never calibrates on the test split.

## Tests

Dependency-level tests:

```bash
python -m unittest -v \
  tests.test_inference \
  tests.test_evaluation_metrics \
  tests.test_evaluator
```

Optional two-sample Magnolia test using the real MRI dataset and checkpoint:

```bash
export SYNTHETIC_MRI_DATASET=/data/scavone/syntheticMRI/patches/syntheticMRI
export GNBM_MRI_CHECKPOINT=/absolute/path/to/checkpoint.pt

python -m unittest -v tests.test_evaluation_cluster
```

The cluster test exercises real loading, CUDA inference, graph metrics, JSON
export, and PNG rendering together.
