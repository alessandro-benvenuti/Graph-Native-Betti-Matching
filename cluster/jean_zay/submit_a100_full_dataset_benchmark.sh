#!/usr/bin/env bash
# Submit equivalent global-batch-32 layouts to the A100 development QoS.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on Jean Zay." >&2
  exit 2
fi
for name in WORK SCRATCH SYNTHETIC_MRI_DATASET GNBM_VENV; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set; source env_a100.sh first." >&2
    exit 2
  fi
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="$WORK/logs/graph-native-betti-matching/a100-benchmarks/$stamp"
output_root="$SCRATCH/gnbm-a100-benchmarks/$stamp"
mkdir -p "$report_root" "$output_root"

submit_case() {
  local case_name="$1" gpus="$2" batch_size="$3"
  local case_report="$report_root/$case_name"
  mkdir -p "$case_report"
  sbatch \
    --chdir="$repo_dir" \
    --job-name="gnbm-a100-${gpus}x${batch_size}" \
    --qos=qos_gpu_a100-dev \
    --time=02:00:00 \
    --gres="gpu:$gpus" \
    --cpus-per-task="$((8 * gpus))" \
    --output="$case_report/slurm-%j.out" \
    --error="$case_report/slurm-%j.err" \
    --export=ALL,GNBM_REPO_DIR="$repo_dir",GNBM_BENCHMARK_REPORT_DIR="$report_root",GNBM_BENCHMARK_OUTPUT_DIR="$output_root",GNBM_BENCHMARK_CASE="$case_name",GNBM_BENCHMARK_GPUS="$gpus",GNBM_BENCHMARK_BATCH_SIZE="$batch_size" \
    "$repo_dir/cluster/jean_zay/a100_full_dataset_benchmark.slurm"
}

one="$(submit_case 1gpu_batch32 1 32)"
two="$(submit_case 2gpu_batch16 2 16)"
four="$(submit_case 4gpu_batch8 4 8)"
one_id="${one##* }"
two_id="${two##* }"
four_id="${four##* }"
echo "$one"
echo "$two"
echo "$four"
echo "Queue: squeue -j $one_id,$two_id,$four_id"
echo "Report: $report_root"
echo "After all jobs finish:"
echo "  $GNBM_VENV/bin/python $repo_dir/cluster/jean_zay/summarize_a100_benchmark.py $report_root | tee $report_root/comparison.txt"
