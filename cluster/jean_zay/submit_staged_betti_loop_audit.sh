#!/usr/bin/env bash
# Reuse the spatial loop comparison for control, O07 and S07 at epoch 400.
set -euo pipefail
[[ $# -le 1 ]] || { echo "Usage: $0 [gt-loops|all-validation]" >&2; exit 2; }
export GNBM_LOOP_AUDIT_SCOPE="${1:-gt-loops}"
case "$GNBM_LOOP_AUDIT_SCOPE" in
  gt-loops|all-validation) ;;
  *) echo "Scope must be gt-loops or all-validation" >&2; exit 2 ;;
esac
for name in WORK SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "$name is not set" >&2; exit 2; }
done
export GNBM_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GNBM_LOOP_AUDIT_OUTPUT="${GNBM_LOOP_AUDIT_OUTPUT:-$GNBM_OUTPUT_DIR/loop-audit-e0400-$GNBM_LOOP_AUDIT_SCOPE}"
for label in O07 S07; do
  run="$GNBM_OUTPUT_DIR/promotion-0400/$label"
  for artifact in training-complete resolved-config.yaml models/latest_checkpoint.pt; do
    [[ -f "$run/$artifact" ]] || { echo "Missing $run/$artifact" >&2; exit 2; }
  done
done
for artifact in resolved-config.yaml checkpoints/epoch_0400.pt; do
  [[ -f "$GNBM_OUTPUT_DIR/shared-prefix/$artifact" ]] || { echo "Missing control $artifact" >&2; exit 2; }
done
log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir"
submission="$(sbatch --chdir="$GNBM_REPO_DIR" \
  --qos="${GNBM_LOOP_QOS:-qos_gpu_a100-t3}" \
  --time="${GNBM_LOOP_WALLTIME:-04:00:00}" \
  --output="$log_dir/%x-%j.out" --error="$log_dir/%x-%j.err" \
  "$GNBM_REPO_DIR/cluster/jean_zay/analyze_staged_betti_loops_a100.slurm")"
echo "$submission"
echo "Scope: $GNBM_LOOP_AUDIT_SCOPE; one A100; wall time: ${GNBM_LOOP_WALLTIME:-04:00:00}"
echo "Report: $GNBM_LOOP_AUDIT_OUTPUT/report.md"
echo "Retry the same command after timeout; completed inference exports are reused."
