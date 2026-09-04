#!/usr/bin/env bash
# Submit the two-H100 development batch-8 versus batch-32 benchmark.
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
report_dir="$WORK/logs/graph-native-betti-matching/batch-benchmarks/$stamp"
output_dir="$SCRATCH/gnbm-batch-benchmarks/$stamp"
mkdir -p "$report_dir" "$output_dir"

submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos=qos_gpu_h100-dev \
  --time="${GNBM_BENCHMARK_WALLTIME:-02:00:00}" \
  --output="$report_dir/slurm-%j.out" \
  --error="$report_dir/slurm-%j.err" \
  --export=ALL,GNBM_REPO_DIR="$repo_dir",GNBM_BENCHMARK_REPORT_DIR="$report_dir",GNBM_BENCHMARK_OUTPUT_DIR="$output_dir" \
  "$repo_dir/cluster/jean_zay/h100_batch_benchmark.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Report: $report_dir"
echo "Result: cat $report_dir/comparison.txt"
