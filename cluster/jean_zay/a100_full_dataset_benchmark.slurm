#!/usr/bin/env bash
#SBATCH --job-name=gnbm-a100-bench
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

set -eo pipefail
repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u

gpus="${GNBM_BENCHMARK_GPUS:?}"
batch_size="${GNBM_BENCHMARK_BATCH_SIZE:?}"
case_name="${GNBM_BENCHMARK_CASE:?}"
report_dir="${GNBM_BENCHMARK_REPORT_DIR:?}/$case_name"
output_dir="${GNBM_BENCHMARK_OUTPUT_DIR:?}/$case_name"
config="configs/experiments/boundary_gamma_sweep_500/benchmark_full_mri_node_edge_focal_g20.yaml"
mkdir -p "$report_dir" "$output_dir"

{
  echo "job_id=$SLURM_JOB_ID"
  echo "case=$case_name"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "config=$config"
  echo "dataset=$SYNTHETIC_MRI_DATASET"
  echo "gpus=$gpus"
  echo "batch_size_per_gpu=$batch_size"
  echo "global_batch_size=$((gpus * batch_size))"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  "$GNBM_VENV/bin/python" -c 'import sys, torch; print("python=" + sys.version.split()[0]); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("cudnn=" + str(torch.backends.cudnn.version()))'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
} > "$report_dir/system-info.txt" 2>&1
cp "$repo_dir/$config" "$report_dir/benchmark-config.yaml"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv,noheader,nounits -l 2 > "$report_dir/gpu-telemetry.csv" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

export GNBM_CONFIG="$config"
export GNBM_RUN_NAME="a100_full_${case_name}_job${SLURM_JOB_ID}"
export GNBM_OUTPUT_DIR="$output_dir"
export GNBM_GPUS="$gpus"
export GNBM_GPUS_PER_NODE="$gpus"
export GNBM_BATCH_SIZE="$batch_size"
export WANDB_MODE=disabled
unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

start_seconds=$(date +%s)
set +e
/usr/bin/time -v bash "$repo_dir/cluster/jean_zay/train_a100.slurm" \
  > "$report_dir/train.log" 2> "$report_dir/time-and-errors.log"
exit_code=$?
set -e
end_seconds=$(date +%s)
cleanup
trap - EXIT

{
  echo "case=$case_name"
  echo "gpus=$gpus"
  echo "batch_size_per_gpu=$batch_size"
  echo "global_batch_size=$((gpus * batch_size))"
  echo "wall_seconds=$((end_seconds - start_seconds))"
  echo "allocated_gpu_hours=$(awk -v seconds="$((end_seconds - start_seconds))" -v gpus="$gpus" 'BEGIN {printf "%.6f", seconds * gpus / 3600}')"
  echo "exit_code=$exit_code"
} > "$report_dir/summary.txt"
find "$output_dir" -name performance.jsonl -exec cp {} "$report_dir/performance.jsonl" \;
find "$output_dir" -name resolved-config.yaml -exec cp {} "$report_dir/resolved-config.yaml" \;
find "$output_dir" -name dataset-manifest.json -exec cp {} "$report_dir/dataset-manifest.json" \;
if [[ "$exit_code" -ne 0 ]]; then
  echo "$case_name failed; see $report_dir/time-and-errors.log" >&2
  exit "$exit_code"
fi
echo "Completed $case_name; report: $report_dir"
