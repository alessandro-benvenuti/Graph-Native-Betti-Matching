#!/usr/bin/env bash
# Evaluate three full-data models on one shared random patch set and render HTML.
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "Usage: $0 BASELINE_CHECKPOINT NODE_FOCAL_CHECKPOINT NODE_EDGE_FOCAL_CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi

baseline_checkpoint="$1"
node_focal_checkpoint="$2"
combined_checkpoint="$3"
dataset_root="${SYNTHETIC_MRI_DATASET:-${SCRATCH:-}/datasets/syntheticMRI/new_patches}"
output_root="${4:-${GNBM_OUTPUT_DIR:-${SCRATCH:-}/experiments/gnbm}/random10_three_models}"
split="${GNBM_RANDOM_PATCH_SPLIT:-test}"
sample_count="${GNBM_RANDOM_PATCH_COUNT:-10}"
sample_seed="${GNBM_RANDOM_PATCH_SEED:-364505}"
batch_size="${GNBM_EVALUATION_BATCH_SIZE:-2}"
workers="${GNBM_WORKERS:-4}"

if [[ -z "${SCRATCH:-}" && -z "${SYNTHETIC_MRI_DATASET:-}" ]]; then
  echo "Set SYNTHETIC_MRI_DATASET or run inside the Jean-Zay environment with SCRATCH set." >&2
  exit 2
fi
if [[ ! -d "$dataset_root/$split/raw" ]]; then
  echo "SyntheticMRI split not found: $dataset_root/$split" >&2
  exit 2
fi
for checkpoint in "$baseline_checkpoint" "$node_focal_checkpoint" "$combined_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done
case "$sample_count" in
  ''|*[!0-9]*|0) echo "GNBM_RANDOM_PATCH_COUNT must be a positive integer." >&2; exit 2 ;;
esac
case "$sample_seed" in
  ''|*[!0-9]*) echo "GNBM_RANDOM_PATCH_SEED must be a non-negative integer." >&2; exit 2 ;;
esac
if [[ -e "$output_root" ]]; then
  echo "Output already exists; refusing to overwrite prediction exports: $output_root" >&2
  exit 2
fi

python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "CUDA is required by this production evaluation script")'

mkdir -p "$output_root"
sample_list="$output_root/source_sample_ids.txt"

python - "$dataset_root" "$split" "$sample_count" "$sample_seed" "$sample_list" <<'PY'
import random
import sys
from pathlib import Path

from data.loaders.discovery import discover_synthetic_mri

dataset_root, split, count, seed, output = sys.argv[1:]
records = discover_synthetic_mri(Path(dataset_root), split)
sample_ids = [record.sample_id for record in records]
count = int(count)
if len(sample_ids) < count:
    raise RuntimeError(f"Requested {count} patches, but {split} contains only {len(sample_ids)}")
selected = random.Random(int(seed)).sample(sample_ids, count)
Path(output).write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"Selected {count} of {len(sample_ids)} {split} patches with seed {seed}:")
print("\n".join(selected))
PY

names=(baseline node_focal nodefocal_edgefocal_mm)
configs=(
  configs/experiments/full_dataset_comparison/finetune_baseline.yaml
  configs/experiments/full_dataset_node_focal/finetune.yaml
  configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml
)
checkpoints=(
  "$baseline_checkpoint"
  "$node_focal_checkpoint"
  "$combined_checkpoint"
)

for index in "${!names[@]}"; do
  name="${names[$index]}"
  evaluation_dir="$output_root/evaluations/$name"
  echo "Evaluating $name on the shared $sample_count-patch manifest..."
  python evaluate.py \
    --config "${configs[$index]}" \
    --checkpoint "${checkpoints[$index]}" \
    --output-dir "$evaluation_dir" \
    --dataset synthetic_mri \
    --split "$split" \
    --sample-list "$sample_list" \
    --batch-size "$batch_size" \
    --workers "$workers" \
    --device cuda

  python - "$evaluation_dir/predictions.json" "$sample_list" <<'PY'
import json
import sys
from pathlib import Path

records = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
actual = [record.get("source_sample_id") for record in records]
if actual != expected:
    raise RuntimeError(f"Prediction IDs do not match the shared manifest: {actual!r}")
PY

  visualization_dir="$output_root/html/$name"
  while IFS= read -r sample_id; do
    python scripts/visualize_graph_prediction_3d.py \
      --dataset-root "$dataset_root" \
      --split "$split" \
      --predictions "$evaluation_dir/predictions.json" \
      --sample-id "$sample_id" \
      --output-dir "$visualization_dir" \
      --error-analysis
  done < "$sample_list"
done

html_count="$(find "$output_root/html" -type f -name '*.html' | wc -l)"
echo "Finished: $html_count interactive reports"
echo "Shared patch manifest: $sample_list"
echo "Download this directory to your Mac: $output_root/html"
