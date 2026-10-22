#!/usr/bin/env bash
# Run on a Jean Zay login node. Never switches a campaign checkout.
set -euo pipefail
: "${WORK:?WORK is required}"
repo_dir="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
remote_url=https://github.com/alessandro-benvenuti/Graph-Native-Betti-Matching.git
branch=codex/cached-logit-betti-test
if [[ ! -e "$repo_dir" ]]; then
  git clone --single-branch --branch "$branch" "$remote_url" "$repo_dir"
  printf '%s\n' "$branch" > "$repo_dir/.git/cached-logit-disposable"
else
  [[ -d "$repo_dir/.git" && -f "$repo_dir/.git/cached-logit-disposable" ]] || { echo 'Refusing existing unrelated directory' >&2; exit 2; }
  [[ "$(git -C "$repo_dir" remote get-url origin)" == "$remote_url" ]] || exit 2
  [[ "$(git -C "$repo_dir" branch --show-current)" == "$branch" ]] || exit 2
  [[ -z "$(git -C "$repo_dir" status --porcelain)" ]] || { echo 'Refusing dirty disposable checkout' >&2; exit 2; }
  git -C "$repo_dir" fetch origin "$branch"
  git -C "$repo_dir" merge --ff-only "origin/$branch"
fi
[[ "$(git -C "$repo_dir" rev-parse --git-common-dir)" == .git ]] || exit 2
cd "$repo_dir"
git log -1 --format='%H %s'
printf 'Prepared %s\n' "$PWD"
