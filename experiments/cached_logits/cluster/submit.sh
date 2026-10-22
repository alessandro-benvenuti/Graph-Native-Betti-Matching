#!/usr/bin/env bash
# Login-node submission: perform Git checks once, before allocating resources.
set -euo pipefail
: "${WORK:?WORK is required}"
repo_dir="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
source "${BASH_SOURCE[0]%/*}/git_env.sh"
cached_logit_require_git
stage="${1:-smoke}"
case "$stage" in
  smoke|cache-main) batch_script=cache.slurm ;;
  benchmark|main|resume) batch_script=optimize.slurm ;;
  *) echo 'Usage: submit.sh {smoke|cache-main|benchmark|main|resume}' >&2; exit 2 ;;
esac
command -v sbatch >/dev/null || { echo 'Run submit.sh on a Jean Zay login node' >&2; exit 2; }
# Standard-library-only helper: no network/model/dataset imports on the login node.
export GNBM_CODE_PROVENANCE_FILE
GNBM_CODE_PROVENANCE_FILE="$(python3 "$repo_dir/experiments/cached_logits/provenance.py" capture --repo "$repo_dir")"
case "$stage" in
  smoke) export CACHED_LOGIT_STAGE=smoke ;;
  cache-main) export CACHED_LOGIT_STAGE=main ;;
  *) export CACHED_LOGIT_CPU_STAGE="$stage" ;;
esac
root=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
mkdir -p "$root/logs"
cd "$repo_dir"
printf 'Verified code provenance: %s\n' "$GNBM_CODE_PROVENANCE_FILE"
sbatch --export=ALL --output="$root/logs/%x-%j.out" \
  "$repo_dir/experiments/cached_logits/cluster/$batch_script"
