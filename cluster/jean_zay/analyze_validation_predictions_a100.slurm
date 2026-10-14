#!/usr/bin/env bash
#SBATCH --job-name=gnbm-corruption-analysis
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
set -eo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u
export SYNTHETIC_MRI_DATASET="${SYNTHETIC_MRI_DATASET:-/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary}"
analysis_output="${GNBM_ANALYSIS_OUTPUT:?Set GNBM_ANALYSIS_OUTPUT to a new output directory}"
# The inherited training config requires this variable during interpolation,
# although this read-only analysis never writes training checkpoints there.
export GNBM_OUTPUT_DIR="${GNBM_OUTPUT_DIR:-$analysis_output/config-runtime-unused}"
checkpoint_root=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-betti-staged-activation-ddp2-a100/shared-prefix/checkpoints
args=()
if [[ -n "${GNBM_ANALYSIS_MAX_SAMPLES:-}" ]]; then
  args+=(--max-samples "$GNBM_ANALYSIS_MAX_SAMPLES")
fi
if [[ "${GNBM_ANALYSIS_REPLACE_INCOMPLETE:-0}" == "1" ]]; then
  args+=(--replace-incomplete)
fi
for required in \
  "$SYNTHETIC_MRI_DATASET/patch_index.csv" \
  "$checkpoint_root/epoch_0200.pt" \
  "$checkpoint_root/epoch_0300.pt" \
  "$checkpoint_root/epoch_0400.pt" \
  "$checkpoint_root/epoch_0500.pt"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done
python analyze_validation_predictions.py \
  --config "${GNBM_ANALYSIS_CONFIG:-configs/experiments/staged_betti_activation/shared_prefix.yaml}" \
  --dataset "${GNBM_ANALYSIS_DATASET:-synthetic_mri}" \
  --patch-index "$SYNTHETIC_MRI_DATASET/patch_index.csv" \
  --output-dir "$analysis_output" \
  --batch-size "${GNBM_ANALYSIS_BATCH_SIZE:-4}" \
  --workers 8 \
  --checkpoint "$checkpoint_root/epoch_0200.pt" \
  --checkpoint "$checkpoint_root/epoch_0300.pt" \
  --checkpoint "$checkpoint_root/epoch_0400.pt" \
  --checkpoint "$checkpoint_root/epoch_0500.pt" "${args[@]}"
