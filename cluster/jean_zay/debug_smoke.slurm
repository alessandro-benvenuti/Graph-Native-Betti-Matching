#!/usr/bin/env bash
#SBATCH --job-name=gnbm-debug
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=00:45:00
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

# Debug allocations verify the training path without creating cloud runs.
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "=== Allocation ==="
echo "job=$SLURM_JOB_ID host=$(hostname) account=${SLURM_JOB_ACCOUNT:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Runtime ==="
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
if torch.cuda.get_device_capability(0) != (9, 0):
    raise SystemExit("Expected an H100 with compute capability 9.0")
PY

echo "=== Build deformable attention ==="
scripts/build_deformable_attention.sh

echo "=== Configuration and W&B tests ==="
python -m unittest -v tests.test_tracking tests.test_experiment_configs

echo "=== CUDA/model tests ==="
python -m unittest -v tests.test_model_cluster.CudaModelTests

echo "=== Real checkpoint compatibility ==="
python -m unittest -v \
  tests.test_model_cluster.RealCheckpointTests.test_mri_checkpoint_matches_all_active_model_tensors

echo "=== Real MRI loader ==="
AUGMENTATION_DATASET_SAMPLES=1 python -m unittest -v \
  tests.test_data_loader_datasets.SyntheticMRILoaderIntegrationTests

echo "=== One-epoch full-loss training smoke test ==="
run_name="jean_zay_debug_${SLURM_JOB_ID}"
python -u train.py \
  --config configs/smoke_synthetic_mri_focal_betti.yaml \
  --initial-weights "$GNBM_MRI_CHECKPOINT" \
  --output-dir "$GNBM_OUTPUT_DIR" \
  --run-name "$run_name" \
  --batch-size 2 \
  --workers 0

checkpoint="$GNBM_OUTPUT_DIR/$run_name/models/best_checkpoint.pt"
if [[ ! -s "$checkpoint" ]]; then
  echo "Smoke training did not create $checkpoint" >&2
  exit 1
fi

echo "=== Resume integrity check ==="
python - <<'PY'
import os
import torch

path = os.path.join(
    os.environ["GNBM_OUTPUT_DIR"],
    "jean_zay_debug_" + os.environ["SLURM_JOB_ID"],
    "models",
    "best_checkpoint.pt",
)
checkpoint = torch.load(path, map_location="cpu")
required = {"net", "optimizer", "scheduler", "epoch", "iteration"}
missing = sorted(required.difference(checkpoint))
if missing:
    raise SystemExit("Checkpoint is missing: " + ", ".join(missing))
print("Checkpoint:", path)
print("epoch:", checkpoint["epoch"], "iteration:", checkpoint["iteration"])
PY

echo "Debug smoke test completed successfully."
