#!/usr/bin/env bash
#SBATCH --job-name=gnbm-a100-debug
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -eo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u
export WANDB_MODE=disabled

echo "=== A100 allocation and runtime ==="
echo "job=$SLURM_JOB_ID host=$(hostname) account=${SLURM_JOB_ACCOUNT:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
if torch.cuda.get_device_capability(0) != (8, 0):
    raise SystemExit("Expected an A100 with compute capability 8.0")
PY

echo "=== Build sm_80 deformable attention ==="
bash scripts/build_deformable_attention.sh
echo "=== Configuration and tracking tests ==="
python -m unittest -v tests.test_tracking tests.test_experiment_configs
echo "=== CUDA forward/backward tests ==="
python -m unittest -v tests.test_model_cluster.CudaModelTests
echo "=== Checkpoint compatibility ==="
python -m unittest -v tests.test_model_cluster.RealCheckpointTests.test_mri_checkpoint_matches_all_active_model_tensors
echo "=== MRI loader ==="
AUGMENTATION_DATASET_SAMPLES=1 python -m unittest -v tests.test_data_loader_datasets.SyntheticMRILoaderIntegrationTests

echo "=== One-epoch training and checkpoint ==="
run_name="jean_zay_a100_debug_${SLURM_JOB_ID}"
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
python - "$checkpoint" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu")
required = {"net", "optimizer", "scheduler", "epoch", "iteration"}
missing = sorted(required.difference(checkpoint))
if missing:
    raise SystemExit("Checkpoint is missing: " + ", ".join(missing))
print("Checkpoint resume payload is complete:", sys.argv[1])
PY
echo "A100 debug smoke test completed successfully."
