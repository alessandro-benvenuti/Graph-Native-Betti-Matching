#!/usr/bin/env bash
# Mechanistic train-set topology overfit from the mature node-focal checkpoint.
#SBATCH --job-name=gnbm-topo-overfit
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

required=(GNBM_MECH_CHECKPOINT GNBM_MECH_SOURCE_CONFIG GNBM_MECH_OUTPUT SYNTHETIC_MRI_DATASET)
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
for path in "$GNBM_MECH_CHECKPOINT" "$GNBM_MECH_SOURCE_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Required source artifact is missing: $path" >&2
    exit 2
  fi
done

output="$GNBM_MECH_OUTPUT"
pool="$output/pool"
screening="$output/screening"
selection="$output/selection"
runs="$output/runs"
evaluation="$output/evaluation"
comparison="$output/comparison"
pool_total="${GNBM_MECH_POOL_TOTAL:-512}"
selected_total="${GNBM_MECH_SELECTED_TOTAL:-25}"
match_distance="${GNBM_MECH_MATCH_DISTANCE:-0.1}"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "checkpoint=$GNBM_MECH_CHECKPOINT"
echo "dataset=$SYNTHETIC_MRI_DATASET"
echo "output=$output"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$output"
cd "$repo_dir"

"$python_bin" -m unittest \
  tests.test_topology_overfit_experiment \
  tests.test_inference \
  tests.test_interactive_visualization

"$python_bin" scripts/build_topology_overfit_pool.py \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --output-dir "$pool" \
  --total "$pool_total" \
  --loop-fraction 0.5 \
  --seed 364505

"$python_bin" evaluate.py \
  --config "$GNBM_MECH_SOURCE_CONFIG" \
  --checkpoint "$GNBM_MECH_CHECKPOINT" \
  --output-dir "$screening" \
  --dataset synthetic_mri \
  --split train \
  --sample-list "$pool/source_sample_ids.txt" \
  --batch-size 32 \
  --workers 8 \
  --device cuda \
  --export-all-edge-scores \
  --visualizations 0

"$python_bin" scripts/select_topology_overfit_patches.py \
  --predictions "$screening/predictions.json" \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --output-dir "$selection" \
  --total "$selected_total" \
  --max-node-distance "$match_distance"

export GNBM_OUTPUT_DIR="$runs"
export GNBM_OVERFIT_SAMPLE_LIST="$selection/source_sample_ids.txt"

for method in control betti; do
  config="configs/experiments/topology_overfit_node_focal/${method}.yaml"
  run="topology_overfit_node_focal_${method}_seed364505"
  echo "Training mechanistic $method arm on selected train patches..."
  "$python_bin" -u train.py \
    --config "$config" \
    --initial-weights "$GNBM_MECH_CHECKPOINT" \
    --output-dir "$runs" \
    --run-name "$run" \
    --device cuda \
    --workers 8
done

"$python_bin" evaluate.py \
  --config "$GNBM_MECH_SOURCE_CONFIG" \
  --checkpoint "$GNBM_MECH_CHECKPOINT" \
  --output-dir "$evaluation/before" \
  --dataset synthetic_mri \
  --split train \
  --sample-list "$selection/source_sample_ids.txt" \
  --batch-size 25 --workers 8 --device cuda \
  --export-all-edge-scores --visualizations 0

for method in control betti; do
  run="topology_overfit_node_focal_${method}_seed364505"
  checkpoint="$runs/$run/models/checkpoint_epoch=100.pt"
  "$python_bin" evaluate.py \
    --config "$runs/$run/resolved-config.yaml" \
    --checkpoint "$checkpoint" \
    --output-dir "$evaluation/$method" \
    --dataset synthetic_mri \
    --split train \
    --sample-list "$selection/source_sample_ids.txt" \
    --batch-size 25 --workers 8 --device cuda \
    --export-all-edge-scores --visualizations 0
done

"$python_bin" scripts/summarize_topology_overfit.py \
  --selection "$selection/selection.json" \
  --before "$evaluation/before/predictions.json" \
  --control "$evaluation/control/predictions.json" \
  --betti "$evaluation/betti/predictions.json" \
  --output-dir "$comparison"

if [[ "${GNBM_MECH_RENDER:-1}" == "1" ]]; then
  "$python_bin" scripts/visualize_topology_overfit.py \
    --selection "$selection/selection.json" \
    --before "$evaluation/before/predictions.json" \
    --control "$evaluation/control/predictions.json" \
    --betti "$evaluation/betti/predictions.json" \
    --dataset-root "$SYNTHETIC_MRI_DATASET" \
    --output-dir "$output/visualizations" \
    --per-category 2 \
    --soft-edge-threshold 0.25 \
    --match-distance "$match_distance"
fi

touch "$output/experiment-complete"
echo "Mechanistic topology overfit complete: $output"
