#!/usr/bin/env bash
# Full-test qualitative screening on the A100 development QoS.
#SBATCH --job-name=gnbm-focal-figure
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread

set -eo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u

root="$GNBM_FOCAL_FIGURE_OUTPUT"
evaluation_root="$root/evaluations"
selection_root="$root/selection"
dataset_root="$SYNTHETIC_MRI_DATASET"
mkdir -p "$evaluation_root" "$selection_root"

names=(baseline node_focal node_edge_focal)
configs=(
  configs/experiments/boundary_gamma_extension_750/baseline.yaml
  configs/experiments/boundary_gamma_extension_750/node_focal.yaml
  configs/experiments/boundary_gamma_extension_750/node_edge_focal_g05.yaml
)
checkpoints=(
  "$GNBM_FOCAL_BASELINE_CHECKPOINT"
  "$GNBM_FOCAL_NODE_CHECKPOINT"
  "$GNBM_FOCAL_NODE_EDGE_CHECKPOINT"
)
source_runs=(
  "$GNBM_EXTENSION_SOURCE_ROOT/finetune_boundary_mri500_baseline_seed364505"
  "$GNBM_EXTENSION_SOURCE_ROOT/finetune_boundary_mri500_node_focal_seed364505"
  "$GNBM_EXTENSION_SOURCE_ROOT/finetune_boundary_mri500_node_edge_focal_g05_seed364505"
)

evaluate_one() {
  local index="$1"
  local gpu="$2"
  local output="$evaluation_root/${names[$index]}"
  if [[ -s "$output/predictions.json" && -s "$output/per-patch-metrics.csv" ]]; then
    echo "Reusing complete prediction export: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    mv "$output" "${output}.incomplete-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  GNBM_EXTENSION_SOURCE_RUN="${source_runs[$index]}" \
  "$GNBM_VENV/bin/python" -u evaluate.py \
    --config "${configs[$index]}" \
    --checkpoint "${checkpoints[$index]}" \
    --output-dir "$output" \
    --dataset synthetic_mri \
    --split test \
    --batch-size 32 \
    --workers 4 \
    --device cuda \
    --visualizations 0
}

echo "Evaluating baseline and node focal in parallel..."
evaluate_one 0 0 & baseline_pid=$!
evaluate_one 1 1 & node_pid=$!
wait "$baseline_pid"
wait "$node_pid"

echo "Evaluating node+edge focal..."
evaluate_one 2 0

"$GNBM_VENV/bin/python" scripts/select_focal_visual_examples.py \
  --baseline "$evaluation_root/baseline/per-patch-metrics.csv" \
  --node-focal "$evaluation_root/node_focal/per-patch-metrics.csv" \
  --node-edge-focal "$evaluation_root/node_edge_focal/per-patch-metrics.csv" \
  --output-dir "$selection_root" \
  --limit "${GNBM_FOCAL_FIGURE_LIMIT:-5}"

while IFS= read -r sample_id; do
  [[ -n "$sample_id" ]] || continue
  for name in "${names[@]}"; do
    "$GNBM_VENV/bin/python" scripts/visualize_graph_prediction_3d.py \
      --dataset-root "$dataset_root" \
      --split test \
      --predictions "$evaluation_root/$name/predictions.json" \
      --sample-id "$sample_id" \
      --output-dir "$root/html/$sample_id/$name" \
      --error-analysis
  done
done < "$selection_root/source_sample_ids.txt"

touch "$root/complete"
echo "Qualitative screening complete: $root"
echo "Selection: $selection_root/selection.json"
echo "Editable predictions: $evaluation_root/*/predictions.json"
echo "HTML: $root/html"
