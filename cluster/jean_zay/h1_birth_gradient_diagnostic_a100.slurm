#!/usr/bin/env bash
# Frozen-checkpoint H1 birth-edge gradient diagnosis on genuine-loop patches.
#SBATCH --job-name=gnbm-h1-birth-grad
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

required=(GNBM_MECH_OUTPUT SYNTHETIC_MRI_DATASET)
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

experiment="$GNBM_MECH_OUTPUT"
pool="$experiment/pool/pool.json"
selection="$experiment/selection/selection.json"
runs="$experiment/runs"
output="$experiment/h1-gradient-diagnostic"
sample_selection="$output/selection"
unselected_limit="${GNBM_H1_DIAGNOSTIC_UNSELECTED:-32}"

for path in "$pool" "$selection"; do
  if [[ ! -f "$path" ]]; then
    echo "Required completed overfit artifact is missing: $path" >&2
    exit 2
  fi
done

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "experiment=$experiment"
echo "output=$output"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$output"
cd "$repo_dir"

"$python_bin" -m unittest \
  tests.test_betti_h1 \
  tests.test_node_edge_betti_diagnostic \
  tests.test_topology_overfit_experiment

if [[ ! -f "$sample_selection/source_sample_ids.txt" ]]; then
  "$python_bin" scripts/select_h1_gradient_diagnostic_patches.py \
    --pool "$pool" \
    --selection "$selection" \
    --output-dir "$sample_selection" \
    --unselected-limit "$unselected_limit" \
    --seed 364505
fi

for method in control betti; do
  run="topology_overfit_node_focal_${method}_seed364505"
  config="$runs/$run/resolved-config.yaml"
  checkpoint="$runs/$run/models/checkpoint_epoch=100.pt"
  method_output="$output/$method"
  for path in "$config" "$checkpoint"; do
    if [[ ! -f "$path" ]]; then
      echo "Required $method artifact is missing: $path" >&2
      exit 2
    fi
  done
  if [[ -f "$method_output/summary.json" ]]; then
    echo "Reusing completed $method H1 diagnostic"
    continue
  fi
  "$python_bin" scripts/diagnose_node_edge_betti.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$method_output" \
    --dataset synthetic_mri \
    --split train \
    --sample-list "$sample_selection/source_sample_ids.txt" \
    --batch-size 1 \
    --workers 4 \
    --device cuda \
    --aggregation hybrid \
    --alpha 0.5 \
    --unmatched-object-threshold 0.25 \
    --max-active-unmatched 8 \
    --normalization matched_mean \
    --mode node_aware
done

"$python_bin" scripts/summarize_h1_gradient_diagnostic.py \
  --manifest "$sample_selection/manifest.json" \
  --control "$output/control/per-sample.json" \
  --betti "$output/betti/per-sample.json" \
  --output-dir "$output/comparison"

touch "$output/diagnostic-complete"
echo "H1 birth-edge gradient diagnostic complete: $output"
