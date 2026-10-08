#!/usr/bin/env bash
# Submit the fixed-stream Betti component and subsampling profile.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "$name is not set." >&2; exit 2; }
done

export GNBM_REPO_DIR="$repo_dir"
export GNBM_PROFILE_CHECKPOINT="${GNBM_PROFILE_CHECKPOINT:-$GNBM_OUTPUT_DIR/shared-prefix/checkpoints/epoch_0300.pt}"
export GNBM_PROFILE_OUTPUT="${GNBM_PROFILE_OUTPUT:-$GNBM_OUTPUT_DIR/profiles/betti-sampling-epoch300}"

[[ -f "$GNBM_PROFILE_CHECKPOINT" ]] || {
  echo "Missing epoch-300 checkpoint: $GNBM_PROFILE_CHECKPOINT" >&2
  exit 2
}
mkdir -p "$GNBM_PROFILE_OUTPUT" "$WORK/logs/graph-native-betti-matching/a100"

echo "Betti sampling profile"
echo "  checkpoint: $GNBM_PROFILE_CHECKPOINT"
echo "  output: $GNBM_PROFILE_OUTPUT"
echo "  local batch: ${GNBM_PROFILE_BATCH_SIZE:-16}"
echo "  profile: warmup=${GNBM_PROFILE_WARMUP_STEPS:-10} measured=${GNBM_PROFILE_MEASURED_STEPS:-100} gradient_repeats=${GNBM_PROFILE_GRADIENT_REPEATS:-12}"

cd "$repo_dir"
sbatch \
  --output="$WORK/logs/graph-native-betti-matching/a100/%x-%j.out" \
  --error="$WORK/logs/graph-native-betti-matching/a100/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/profile_betti_sampling_a100.slurm"

