#!/usr/bin/env bash
# Upload completed offline W&B runs from a networked Jean Zay login node.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${1:-${GNBM_OUTPUT_DIR:-}}"
if [[ -z "$root" ]]; then
  echo "Usage: $0 [OUTPUT_ROOT], or export GNBM_OUTPUT_DIR." >&2
  exit 2
fi
if [[ ! -d "$root" ]]; then
  echo "Output root does not exist: $root" >&2
  exit 2
fi

cd "$repo_dir"
source cluster/jean_zay/env_a100.sh
unset WANDB_MODE
sync_entity="${GNBM_WANDB_SYNC_ENTITY:-alessandrobenvenuti2002-politecnico-di-torino}"
sync_project="${GNBM_WANDB_SYNC_PROJECT:-focal-loss}"

mapfile -t run_files < <(
  # Successful `wandb sync` transactions are renamed to *.wandb.synced, so
  # selecting only *.wandb makes retries skip segments already uploaded.
  find "$root" -type f -name 'run-*.wandb' \
    -path '*/wandb/offline-run-*/run-*.wandb' -print | sort
)
if [[ "${#run_files[@]}" -eq 0 ]]; then
  echo "No unsynchronized offline W&B transactions found under $root"
  exit 0
fi

echo "Found ${#run_files[@]} unsynchronized offline W&B transaction(s) under $root"
echo "Destination: $sync_entity/$sync_project"
failed_runs=()
for run_file in "${run_files[@]}"; do
  echo
  # A preceding legacy sync can rename a transaction while this process is
  # running. Re-check the snapshot entry instead of failing on a stale path.
  if [[ ! -f "$run_file" ]]; then
    echo "Transaction is no longer pending; skipping: $run_file"
    continue
  fi
  run="${run_file%/*}"
  echo "Syncing $run_file"

  # Wall-time termination can prevent the SDK from writing its final summary.
  # The legacy sender requires the file even though history is stored in the
  # transaction. An empty summary preserves all recorded metric history and a
  # later segment from the same run can supply the final summary values.
  summary_file="$run/files/wandb-summary.json"
  if [[ ! -f "$summary_file" ]]; then
    mkdir -p "$run/files"
    printf '{}\n' > "$summary_file"
    echo "Created missing terminal metadata: $summary_file"
  fi

  run_id="${run_file##*/run-}"
  run_id="${run_id%.wandb}"
  # Every invocation is retry-safe: --append can create the destination when
  # absent and appends when an earlier attempt or segment already created it.
  # The legacy reader tolerates the truncated tail commonly left when Slurm
  # terminates an offline process at its wall-time limit.
  if ! wandb sync --legacy --append --id "$run_id" \
    --entity "$sync_entity" --project "$sync_project" "$run_file"; then
    echo "W&B sync failed; continuing with the remaining runs: $run" >&2
    failed_runs+=("$run")
  fi
done

if (( ${#failed_runs[@]} > 0 )); then
  echo >&2
  echo "Failed to synchronize ${#failed_runs[@]} offline run(s):" >&2
  printf '  %s\n' "${failed_runs[@]}" >&2
  exit 1
fi
