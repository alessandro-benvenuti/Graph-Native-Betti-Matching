#!/usr/bin/env bash
# Submit separate one- and two-H100 full-MRI development benchmarks.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
for name in WORK SCRATCH SYNTHETIC_MRI_DATASET; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set; source cluster/jean_zay/env.sh first." >&2
    exit 2
  fi
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="$WORK/logs/graph-native-betti-matching/full-dataset-benchmarks/$stamp"
output_root="$SCRATCH/gnbm-full-dataset-benchmarks/$stamp"
mkdir -p "$report_root" "$output_root"

submit_case() {
  local case_name="$1"
  local gpus="$2"
  local batch_size="$3"
  local case_report="$report_root/$case_name"
  mkdir -p "$case_report"
  sbatch \
    --chdir="$repo_dir" \
    --job-name="gnbm-${case_name}" \
    --qos=qos_gpu_h100-dev \
    --time="${GNBM_BENCHMARK_WALLTIME:-02:00:00}" \
    --gres="gpu:$gpus" \
    --cpus-per-task="$((10 * gpus))" \
    --output="$case_report/slurm-%j.out" \
    --error="$case_report/slurm-%j.err" \
    --export=ALL,GNBM_REPO_DIR="$repo_dir",GNBM_BENCHMARK_REPORT_DIR="$report_root",GNBM_BENCHMARK_OUTPUT_DIR="$output_root",GNBM_BENCHMARK_CASE="$case_name",GNBM_BENCHMARK_GPUS="$gpus",GNBM_BENCHMARK_BATCH_SIZE="$batch_size" \
    "$repo_dir/cluster/jean_zay/h100_full_dataset_benchmark.slurm"
}

one_submission="$(submit_case 1gpu_batch32 1 32)"
two_submission="$(submit_case 2gpu_batch16 2 16)"
one_job="${one_submission##* }"
two_job="${two_submission##* }"

echo "$one_submission"
echo "$two_submission"
echo "Queue:  squeue -j $one_job,$two_job"
echo "Report: $report_root"
echo "After both jobs finish:"
echo "  $GNBM_VENV/bin/python $repo_dir/cluster/jean_zay/summarize_full_dataset_benchmark.py $report_root | tee $report_root/comparison.txt"
