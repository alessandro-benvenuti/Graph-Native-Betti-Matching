#!/usr/bin/env bash
# A100 entry point for one staged Betti prefix, screening worker, or final continuation.
#SBATCH --job-name=gnbm-staged-betti
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
export WANDB_PROJECT=gnbm
export WANDB_RUN_GROUP=staged-betti-activation
if [[ "$GNBM_STAGED_CONFIG" == *"/smoke.yaml" || "$GNBM_STAGED_CONFIG" == "smoke.yaml" ]]; then
  export WANDB_MODE=offline
fi

python_bin="$GNBM_VENV/bin/python"
common=(
  --config "$GNBM_STAGED_CONFIG"
  --output "$GNBM_OUTPUT_DIR"
  --initial-weights "$GNBM_INITIAL_WEIGHTS"
)

echo "job=$SLURM_JOB_ID host=$(hostname) action=$GNBM_STAGED_ACTION branch=${GNBM_STAGED_BRANCH:-none}"
"$python_bin" -c 'import optuna; print("optuna=" + optuna.__version__)'

case "$GNBM_STAGED_ACTION" in
  prefix)
    exec "$python_bin" -u scripts/staged_betti_activation.py prefix "${common[@]}"
    ;;
  screen)
    resume_args=()
    if [[ "${GNBM_STAGED_RESUME:-0}" == "1" ]]; then
      resume_args+=(--resume-running)
    fi
    exec "$python_bin" -u scripts/staged_betti_activation.py screen "${common[@]}" \
      --branch "$GNBM_STAGED_BRANCH" "${resume_args[@]}"
    ;;
  final)
    exec "$python_bin" -u scripts/staged_betti_activation.py final "${common[@]}" \
      --branch "$GNBM_STAGED_BRANCH" --trial "$GNBM_STAGED_TRIAL"
    ;;
  test)
    exec "$python_bin" -u scripts/staged_betti_activation.py test "${common[@]}" \
      --branch "$GNBM_STAGED_BRANCH" --trial "$GNBM_STAGED_TRIAL"
    ;;
  *)
    echo "Unknown GNBM_STAGED_ACTION=$GNBM_STAGED_ACTION" >&2
    exit 2
    ;;
esac
