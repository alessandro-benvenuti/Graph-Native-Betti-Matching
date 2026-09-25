#!/usr/bin/env bash
# A100 entry point for one control or one bounded Pareto worker.
#SBATCH --job-name=gnbm-betti-optuna
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -euo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
export WANDB_MODE="${WANDB_MODE:-offline}"

python_bin="$GNBM_VENV/bin/python"
common=(
  --config "$GNBM_OPTUNA_CONFIG"
  --output "$GNBM_OUTPUT_DIR"
  --initial-weights "$GNBM_INITIAL_WEIGHTS"
  --storage "$GNBM_OPTUNA_STORAGE"
)

echo "job=$SLURM_JOB_ID host=$(hostname) config=$GNBM_OPTUNA_CONFIG"
"$python_bin" -c 'import optuna; print("optuna=" + optuna.__version__)'

case "$GNBM_OPTUNA_ACTION" in
  control)
    "$python_bin" -u scripts/optimize_node_edge_betti.py control "${common[@]}"
    ;;
  worker)
    "$python_bin" -u scripts/optimize_node_edge_betti.py worker "${common[@]}" \
      --max-trials "$GNBM_OPTUNA_MAX_TRIALS"
    ;;
  *) echo "Unknown GNBM_OPTUNA_ACTION=$GNBM_OPTUNA_ACTION" >&2; exit 2 ;;
esac
