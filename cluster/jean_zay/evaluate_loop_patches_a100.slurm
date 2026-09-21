#!/usr/bin/env bash
# Compare the paired best checkpoints on every validation patch with a GT loop.
#SBATCH --job-name=gnbm-loop-val
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u

required=(
  GNBM_LOOP_CONTROL_RUN
  GNBM_LOOP_BETTI_RUN
  GNBM_LOOP_OUTPUT
  SYNTHETIC_MRI_DATASET
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing batch environment variable: $name" >&2
    exit 2
  fi
done

python_bin="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "A100 Python is not executable: $python_bin" >&2
  exit 2
fi

control_config="$GNBM_LOOP_CONTROL_RUN/resolved-config.yaml"
betti_config="$GNBM_LOOP_BETTI_RUN/resolved-config.yaml"
control_checkpoint="$GNBM_LOOP_CONTROL_RUN/models/best_metric_checkpoint.pt"
betti_checkpoint="$GNBM_LOOP_BETTI_RUN/models/best_metric_checkpoint.pt"
for path in \
  "$control_config" "$betti_config" \
  "$control_checkpoint" "$betti_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Required run artifact is missing: $path" >&2
    exit 2
  fi
done

batch_size="${GNBM_LOOP_BATCH_SIZE:-32}"
workers="${GNBM_LOOP_WORKERS:-8}"
node_distance="${GNBM_LOOP_MAX_NODE_DISTANCE:-0.1}"
selection_dir="$GNBM_LOOP_OUTPUT/selection"
control_output="$GNBM_LOOP_OUTPUT/control"
betti_output="$GNBM_LOOP_OUTPUT/betti"
comparison_output="$GNBM_LOOP_OUTPUT/comparison"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "dataset=$SYNTHETIC_MRI_DATASET"
echo "control=$control_checkpoint"
echo "betti=$betti_checkpoint"
echo "output=$GNBM_LOOP_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$GNBM_LOOP_OUTPUT"
"$python_bin" "$repo_dir/scripts/find_gt_loop_patches.py" \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --split val \
  --output-dir "$selection_dir"

for method in control betti; do
  if [[ "$method" == control ]]; then
    config="$control_config"
    checkpoint="$control_checkpoint"
    evaluation_output="$control_output"
  else
    config="$betti_config"
    checkpoint="$betti_checkpoint"
    evaluation_output="$betti_output"
  fi
  echo "Evaluating $method on GT-loop validation patches..."
  "$python_bin" "$repo_dir/evaluate.py" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$evaluation_output" \
    --dataset synthetic_mri \
    --split val \
    --sample-list "$selection_dir/source_sample_ids.txt" \
    --batch-size "$batch_size" \
    --workers "$workers" \
    --device cuda \
    --visualizations 0
done

"$python_bin" "$repo_dir/scripts/compare_loop_patch_predictions.py" \
  --control-predictions "$control_output/predictions.json" \
  --betti-predictions "$betti_output/predictions.json" \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --split val \
  --max-node-distance "$node_distance" \
  --output-dir "$comparison_output"

touch "$GNBM_LOOP_OUTPUT/evaluation-complete"
echo "Loop-patch evaluation complete: $GNBM_LOOP_OUTPUT"
