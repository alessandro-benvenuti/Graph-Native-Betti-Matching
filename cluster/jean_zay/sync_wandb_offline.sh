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
source cluster/jean_zay/env_h100.sh
unset WANDB_MODE

mapfile -t runs < <(
  find "$root" -type d -path '*/wandb/offline-run-*' -print | sort
)
if [[ "${#runs[@]}" -eq 0 ]]; then
  echo "No offline W&B runs found under $root"
  exit 0
fi

echo "Found ${#runs[@]} offline W&B run(s) under $root"
for run in "${runs[@]}"; do
  echo
  echo "Syncing $run"
  wandb sync "$run"
done
