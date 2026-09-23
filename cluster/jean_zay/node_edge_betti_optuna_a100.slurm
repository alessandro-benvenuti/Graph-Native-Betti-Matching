#!/usr/bin/env bash
# Single-controller A100 entry point for the paired Optuna smoke/search campaign.
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
)

echo "job=$SLURM_JOB_ID host=$(hostname) config=$GNBM_OPTUNA_CONFIG"
"$python_bin" -c 'import optuna; print("optuna=" + optuna.__version__)'

if [[ ! -f "$GNBM_OUTPUT_DIR/control-reference.json" ]]; then
  "$python_bin" -u scripts/optimize_node_edge_betti.py control "${common[@]}"
else
  echo "Reusing frozen control reference: $GNBM_OUTPUT_DIR/control-reference.json"
fi
"$python_bin" -u scripts/optimize_node_edge_betti.py optimize "${common[@]}"
"$python_bin" -u scripts/summarize_node_edge_betti_optuna.py \
  --output "$GNBM_OUTPUT_DIR" --study-name "$GNBM_OPTUNA_STUDY_NAME"
