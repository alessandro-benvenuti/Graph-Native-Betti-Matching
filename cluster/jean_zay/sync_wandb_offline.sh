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

mapfile -t runs < <(
  find "$root" -type d -path '*/wandb/offline-run-*' -print | sort
)
if [[ "${#runs[@]}" -eq 0 ]]; then
  echo "No offline W&B runs found under $root"
  exit 0
fi

echo "Found ${#runs[@]} offline W&B run(s) under $root"
echo "Destination: $sync_entity/$sync_project"
declare -A seen_run_ids=()
for run in "${runs[@]}"; do
  echo
  echo "Syncing $run"
  run_file="$(find "$run" -maxdepth 1 -type f -name 'run-*.wandb' -print -quit)"
  if [[ -z "$run_file" ]]; then
    echo "No W&B run file found in $run" >&2
    exit 1
  fi
  run_id="${run_file##*/run-}"
  run_id="${run_id%.wandb}"
  if [[ -n "${seen_run_ids[$run_id]:-}" ]]; then
    wandb sync --append --id "$run_id" \
      --entity "$sync_entity" --project "$sync_project" "$run"
  else
    wandb sync --entity "$sync_entity" --project "$sync_project" "$run"
    seen_run_ids[$run_id]=1
  fi
done
