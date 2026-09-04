#!/usr/bin/env bash
#SBATCH --job-name=gnbm-batch-bench
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
set -u

report_dir="${GNBM_BENCHMARK_REPORT_DIR:?}"
output_root="${GNBM_BENCHMARK_OUTPUT_DIR:?}"
config="configs/experiments/boundary_gamma_sweep_500/benchmark_node_edge_focal_g20.yaml"
mkdir -p "$report_dir" "$output_root"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "python=$GNBM_VENV/bin/python"
  echo "config=$config"
  echo "dataset=$SYNTHETIC_MRI_DATASET"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  "$GNBM_VENV/bin/python" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("cudnn=" + str(torch.backends.cudnn.version()))'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$report_dir/system-info.txt" 2>&1

git -C "$repo_dir" status --short > "$report_dir/git-status.txt"
cp "$repo_dir/$config" "$report_dir/benchmark-config.yaml"

run_case() {
  local batch_size="$1"
  local case_name="batch${batch_size}_per_gpu"
  local case_report="$report_dir/$case_name"
  local case_output="$output_root/$case_name"
  local telemetry_pid
  mkdir -p "$case_report" "$case_output"

  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits \
    -l 2 > "$case_report/gpu-telemetry.csv" &
  telemetry_pid=$!

  export GNBM_CONFIG="$config"
  export GNBM_RUN_NAME="h100_${case_name}_job${SLURM_JOB_ID}"
  export GNBM_OUTPUT_DIR="$case_output"
  export GNBM_GPUS=2
  export GNBM_GPUS_PER_NODE=2
  export GNBM_BATCH_SIZE="$batch_size"
  export WANDB_MODE=disabled
  unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

  local start_seconds end_seconds exit_code
  start_seconds=$(date +%s)
  set +e
  /usr/bin/time -v bash "$repo_dir/cluster/jean_zay/train_h100.slurm" \
    > "$case_report/train.log" 2> "$case_report/time-and-errors.log"
  exit_code=$?
  set -e
  end_seconds=$(date +%s)

  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true

  {
    echo "case=$case_name"
    echo "batch_size_per_gpu=$batch_size"
    echo "global_batch_size=$((batch_size * 2))"
    echo "wall_seconds=$((end_seconds - start_seconds))"
    echo "allocated_gpu_hours=$(awk -v seconds="$((end_seconds - start_seconds))" 'BEGIN {printf "%.6f", seconds * 2 / 3600}')"
    echo "exit_code=$exit_code"
  } > "$case_report/summary.txt"

  find "$case_output" -name performance.jsonl -exec cp {} "$case_report/performance.jsonl" \;
  find "$case_output" -name resolved-config.yaml -exec cp {} "$case_report/resolved-config.yaml" \;
  find "$case_output" -name dataset-manifest.json -exec cp {} "$case_report/dataset-manifest.json" \;

  if [[ "$exit_code" -ne 0 ]]; then
    echo "$case_name failed; see $case_report/time-and-errors.log" >&2
    return "$exit_code"
  fi
}

run_case 8
run_case 32

"$GNBM_VENV/bin/python" "$repo_dir/cluster/jean_zay/summarize_batch_benchmark.py" \
  "$report_dir" > "$report_dir/comparison.txt"
cat "$report_dir/comparison.txt"
echo "Benchmark report: $report_dir"
