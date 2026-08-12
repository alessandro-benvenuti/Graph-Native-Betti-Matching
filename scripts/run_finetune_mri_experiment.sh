#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 {overfit|baseline|focal|betti|focal_betti} INITIAL_CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi

experiment="$1"
checkpoint="$2"
output_dir="${3:-${GNBM_OUTPUT_DIR:-}}"

case "$experiment" in
  overfit|baseline|focal|betti|focal_betti) ;;
  *)
    echo "Unknown experiment: $experiment" >&2
    exit 2
    ;;
esac

if [[ ! -f "$checkpoint" ]]; then
  echo "Initial checkpoint does not exist: $checkpoint" >&2
  exit 2
fi
if [[ -z "$output_dir" ]]; then
  echo "Set GNBM_OUTPUT_DIR or pass OUTPUT_DIR as the third argument." >&2
  exit 2
fi
if [[ -z "${SYNTHETIC_MRI_DATASET:-}" ]]; then
  echo "SYNTHETIC_MRI_DATASET is not set." >&2
  exit 2
fi
if [[ "$experiment" == "overfit" ]]; then
  config="configs/overfit_synthetic_mri_focal_betti.yaml"
  run_name="overfit_mri_focal_betti_seed364505"
else
  config="configs/experiments/finetune_mri/${experiment}.yaml"
  run_name="finetune_mri_${experiment}_seed364505"
fi
run_dir="${output_dir}/${run_name}"
if [[ -e "$run_dir/resolved-config.yaml" ]]; then
  echo "Refusing to overwrite an existing run: $run_dir" >&2
  echo "Resume it explicitly with train.py --resume, or choose another output directory." >&2
  exit 2
fi

mkdir -p "$run_dir"
checksum_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$run_dir/source-commit.txt"
  git status --short --untracked-files=normal > "$run_dir/source-status.txt"
else
  echo "unavailable: exported directory without Git metadata" > "$run_dir/source-commit.txt"
  echo "Git status unavailable; use source-manifest.sha256." > "$run_dir/source-status.txt"
fi

{
  for source_root in train.py boxes configs data models training scripts; do
    if [[ -f "$source_root" ]]; then
      echo "$source_root"
    elif [[ -d "$source_root" ]]; then
      find "$source_root" -type f \
        \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \
           -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.so' \) \
        ! -path '*/__pycache__/*'
    fi
  done
} | LC_ALL=C sort | while IFS= read -r source_file; do
  printf '%s  %s\n' "$(checksum_file "$source_file")" "$source_file"
done > "$run_dir/source-manifest.sha256"

checkpoint_checksum="$(checksum_file "$checkpoint")"
{
  echo "path: $checkpoint"
  echo "sha256: $checkpoint_checksum"
} > "$run_dir/initial-checkpoint.txt"

python train.py \
  --config "$config" \
  --initial-weights "$checkpoint" \
  --output-dir "$output_dir" \
  2>&1 | tee "$run_dir/train.log"
