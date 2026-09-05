#!/usr/bin/env bash
# A100 batch entry point; submit_train_a100.sh supplies QoS and resources.
#SBATCH --job-name=gnbm-a100-train
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

args=(
  --config "$GNBM_CONFIG"
  --output-dir "$GNBM_OUTPUT_DIR"
  --run-name "$GNBM_RUN_NAME"
  --device cuda
)
if [[ -n "${GNBM_BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "$GNBM_BATCH_SIZE")
fi
if [[ -n "${GNBM_WORKERS:-}" ]]; then
  args+=(--workers "$GNBM_WORKERS")
fi

latest_checkpoint="$GNBM_OUTPUT_DIR/$GNBM_RUN_NAME/models/latest_checkpoint.pt"
if [[ -n "${GNBM_RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume "$GNBM_RESUME_CHECKPOINT")
elif [[ "${GNBM_AUTO_RESUME:-0}" == "1" && -f "$latest_checkpoint" ]]; then
  echo "Auto-resuming $latest_checkpoint"
  args+=(--resume "$latest_checkpoint")
elif [[ -n "${GNBM_INITIAL_WEIGHTS:-}" ]]; then
  args+=(--initial-weights "$GNBM_INITIAL_WEIGHTS")
fi

gpus="${GNBM_GPUS:-1}"
python_bin="$GNBM_VENV/bin/python"
echo "job=$SLURM_JOB_ID host=$(hostname) nodes=$SLURM_NNODES gpus=$gpus architecture=a100"
echo "config=$GNBM_CONFIG run=$GNBM_RUN_NAME output=$GNBM_OUTPUT_DIR"
echo "wandb_entity=${WANDB_ENTITY:-default} wandb_project=${WANDB_PROJECT:-focal-loss} wandb_mode=${WANDB_MODE:-online}"
"$python_bin" -c 'import sys, torch; print("python_executable=" + sys.executable); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if (( gpus == 1 )); then
  "$python_bin" -u train.py "${args[@]}"
else
  if (( SLURM_NNODES != 1 )); then
    echo "This launcher supports A100 DDP on one node only." >&2
    exit 2
  fi
  echo "Checking the custom CUDA operator on all $gpus visible A100s..."
  "$python_bin" cluster/jean_zay/check_multi_gpu_ops.py --devices "$gpus"
  "$python_bin" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$gpus" \
    train.py --distributed "${args[@]}"
fi
