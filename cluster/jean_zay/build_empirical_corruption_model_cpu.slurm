#!/usr/bin/env bash
# CPU-only construction of the empirical RelationFormer corruption model.
#SBATCH --job-name=gnbm-corruption-model
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/scratch_paths.sh"
module purge
module load pytorch-gpu/py3/2.3.0
set -u

analysis_root="${GNBM_ANALYSIS_ROOT:-$WORK/analysis/vascular-corruptions-full}"
output_dir="${GNBM_CORRUPTION_MODEL_OUTPUT:-$analysis_root/empirical_corruption_model}"
venv_dir="${GNBM_A100_VENV:-$SCRATCH/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv_dir/bin/python"

for required in "$analysis_root/edge_pairs.pt" "$analysis_root/validation_manifest.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing input: $required" >&2
    exit 2
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "Python is not executable: $python_bin" >&2
  exit 2
fi

cd "$repo_dir"
source "$venv_dir/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "analysis_root=$analysis_root"
echo "output_dir=$output_dir"

if [[ -n "${GNBM_CORRUPTION_REFIT_SOURCE:-}" ]]; then
  "$python_bin" -u scripts/refit_empirical_corruption_model.py \
    --study-root "$GNBM_CORRUPTION_REFIT_SOURCE" --output-dir "$output_dir"
  exit 0
fi

args=(
  --analysis-root "$analysis_root"
  --output-dir "$output_dir"
  --primary-epoch 500
  --threshold 0.5
  --spatial-tolerance 0.03
  --holdout-fraction 0.2
)
if [[ -n "${GNBM_CORRUPTION_MODEL_MAX_RECORDS:-}" ]]; then
  args+=(--max-records-per-epoch "$GNBM_CORRUPTION_MODEL_MAX_RECORDS")
fi
if [[ "${GNBM_CORRUPTION_MODEL_REPLACE:-0}" == "1" ]]; then
  args+=(--replace-output)
fi

"$python_bin" -u scripts/build_empirical_corruption_model.py "${args[@]}"
