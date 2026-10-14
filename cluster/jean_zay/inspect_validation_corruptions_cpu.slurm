#!/usr/bin/env bash
# CPU-only post-processing of existing validation-prediction raw shards.
#SBATCH --job-name=gnbm-corruption-inspect
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --mem=32G

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
module purge
module load pytorch-gpu/py3/2.3.0
set -u

analysis_root="${GNBM_ANALYSIS_ROOT:-$WORK/analysis/vascular-corruptions-full}"
output_dir="${GNBM_DETAILED_ANALYSIS_OUTPUT:-$analysis_root/detailed_summary}"
venv_dir="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv_dir/bin/python"

if [[ ! -f "$analysis_root/checkpoint_comparison.csv" || ! -f "$analysis_root/edge_pairs.pt" ]]; then
  echo "Existing analysis is incomplete: $analysis_root" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python is not executable: $python_bin" >&2
  exit 2
fi

mkdir -p "$output_dir"
cd "$repo_dir"
source "$venv_dir/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "analysis_root=$analysis_root"
echo "output_dir=$output_dir"

"$python_bin" -u scripts/inspect_validation_corruptions.py \
  --analysis-root "$analysis_root" \
  --output-dir "$output_dir" \
  --primary-epoch 500 \
  --threshold 0.5 \
  --spatial-tolerance 0.03
