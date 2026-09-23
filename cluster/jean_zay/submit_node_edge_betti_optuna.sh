#!/usr/bin/env bash
# Submit the single-controller node/edge Betti Optuna campaign.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "search" ) ]]; then
  echo "Usage: $0 smoke|search" >&2
  exit 2
fi
campaign="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
for name in WORK SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR GNBM_INITIAL_WEIGHTS; do
  if [[ -z "${!name:-}" ]]; then echo "$name is not set." >&2; exit 2; fi
done
if [[ ! -f "$GNBM_INITIAL_WEIGHTS" ]]; then
  echo "Initial checkpoint does not exist: $GNBM_INITIAL_WEIGHTS" >&2; exit 2
fi

venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
config="configs/experiments/node_edge_betti_optuna/$campaign.yaml"
study_name="node-edge-betti-optuna${campaign/smoke/-smoke}"
if [[ "$campaign" == "search" ]]; then study_name="node-edge-betti-optuna"; fi

"$python_bin" -c 'import optuna; print("optuna=" + optuna.__version__)'
(
  cd "$repo_dir"
  "$python_bin" -c \
    'import sys; from scripts.optimize_node_edge_betti import load_campaign; load_campaign(__import__("pathlib").Path(sys.argv[1]))' \
    "$config"
)

export GNBM_REPO_DIR="$repo_dir"
export GNBM_VENV="$venv"
export GNBM_OPTUNA_CONFIG="$config"
export GNBM_OPTUNA_STUDY_NAME="$study_name"
export WANDB_MODE="offline"
if [[ "$campaign" == "smoke" ]]; then
  qos="${GNBM_QOS:-qos_gpu_a100-dev}"
  walltime="${GNBM_WALLTIME:-02:00:00}"
else
  qos="${GNBM_QOS:-qos_gpu_a100-t3}"
  walltime="${GNBM_WALLTIME:-20:00:00}"
fi
log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
submission="$(sbatch --qos="$qos" --time="$walltime" \
  --chdir="$repo_dir" --output="$log_dir/%x-%j.out" --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/node_edge_betti_optuna_a100.slurm")"
echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Logs:  $log_dir/gnbm-betti-optuna-$job_id.{out,err}"
