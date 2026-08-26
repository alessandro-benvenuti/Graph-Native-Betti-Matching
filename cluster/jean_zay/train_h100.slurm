#!/usr/bin/env bash
# submit_train.sh supplies QoS, walltime, working directory, and log paths.
#SBATCH --job-name=gnbm-train
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

# Jean Zay's module stack runs Conda deactivation hooks while loading PyTorch.
# Some hooks probe variables that may be unset, so enable nounset only after
# the module-backed project environment has been initialized.
set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
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
gpus_per_node="${GNBM_GPUS_PER_NODE:-$gpus}"
echo "job=$SLURM_JOB_ID host=$(hostname) nodes=$SLURM_NNODES gpus=$gpus"
echo "config=$GNBM_CONFIG run=$GNBM_RUN_NAME output=$GNBM_OUTPUT_DIR"
echo "wandb_entity=${WANDB_ENTITY:-default} wandb_project=${WANDB_PROJECT:-focal-loss} wandb_mode=${WANDB_MODE:-online}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if (( gpus == 1 )); then
  # The batch shell already owns the allocated GPU. Avoid wrapping Python in
  # srun: Jean Zay's OpenMPI stack interprets the resulting step metadata as a
  # direct MPI launch even though this project uses no MPI.
  python -u train.py "${args[@]}"
else
  if (( SLURM_NNODES != 1 )); then
    echo "This launcher currently supports multi-GPU training on one node only." >&2
    exit 2
  fi
  # torchrun creates one process per visible H100 and uses c10d/NCCL. Running
  # directly from the batch shell avoids all PMIx/OpenMPI initialization.
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$gpus" \
    train.py --distributed "${args[@]}"
fi
