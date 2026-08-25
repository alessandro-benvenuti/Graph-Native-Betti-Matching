# SyntheticMRI v2 dataset generation

`scripts/generate_synthetic_mri_dataset.py` creates a new dataset without
modifying the inherited `patches/` directory or `splits.csv` file. Its default
outputs are:

```text
syntheticMRI/
├── new_split.csv
└── new_patches/
    ├── train/{raw,seg,vtp}/
    ├── val/{raw,seg,vtp}/
    ├── test/{raw,seg,vtp}/
    ├── patch_index.csv
    ├── patient_features.csv
    ├── split_balance.csv
    ├── generation_config.json
    └── generation_summary.json
```

## Contract

- The split is patient-level and has exact 70/15/15-rounded sizes: 95 train,
  20 validation, and 21 test patients for the 136-volume dataset.
- Seed 42 and 20,000 deterministic candidate assignments are used by default.
  The selected assignment minimizes standardized mean and dispersion
  differences for foreground fraction, nodes, edges, bifurcations, beta0, and
  beta1.
- The effective source crop is 54 cubed, padded by five voxels per side to 64
  cubed.
- Candidate starts are distributed evenly between both scan boundaries, with
  40 voxels as the maximum step. The known scan shape produces 960 candidates
  per patient and 130,560 candidates in total.
- Every candidate is saved. Segmentation and graph GT are never used as
  selection filters.
- MRI normalization reproduces the inherited MAD clipping and scaling.
- VTP graph coordinates are stored in normalized patch coordinates, matching
  the current SyntheticMRI loader.

The generator is restartable per patient. A completed patient has a private
completion marker. With `--resume`, completed patients are skipped and an
incomplete patient's files are regenerated. Configuration fingerprints prevent
accidentally mixing incompatible grids or splits.

## 1. Plan and inspect the split

Run inside a CPU allocation from the repository root:

```bash
data_root=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI

python -u scripts/generate_synthetic_mri_dataset.py \
  --root "$data_root" \
  --plan-only
```

This writes `new_split.csv` and metadata below `new_patches/`, but no patch
triplets. Inspect the exact split and feature balance before continuing:

```bash
awk -F, 'NR > 1 {count[$2]++} END {for (split in count) print split, count[split]}' \
  "$data_root/new_split.csv"

column -s, -t < "$data_root/new_patches/split_balance.csv" | less -S
```

The raw and segmentation payload is approximately 159 GiB before NIfTI
compression, excluding VTP files and filesystem metadata. Check the project
quota before materialization.

## 2. One-patient smoke generation

The smoke output is part of the final dataset and will be skipped later:

```bash
python -u scripts/generate_synthetic_mri_dataset.py \
  --root "$data_root" \
  --resume \
  --patient-id 1 \
  --workers 1
```

Confirm that the three matching files exist and that the summary reports one
completed patient. A loader integration check may then point
`SYNTHETIC_MRI_DATASET` at `$data_root/new_patches`.

## 3. Full CPU generation

The task does not use a GPU. Parallelism is across patients; each worker loads
one complete scan, so use a conservative worker count initially. An example
Jean Zay submission is:

```bash
repo_dir="$(pwd)"
python_exe="$(command -v python)"
data_root=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI
log_dir="$data_root/results/new_patch_generation_logs"

mkdir -p "$log_dir"

sbatch \
  --job-name=generate_mri_v2 \
  --account="${IDRPROJ}@cpu" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --hint=nomultithread \
  --qos=qos_cpu-t4 \
  --time=48:00:00 \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  --wrap="cd '$repo_dir' && '$python_exe' -u scripts/generate_synthetic_mri_dataset.py \
    --root '$data_root' \
    --resume \
    --workers 8"
```

If memory or filesystem pressure is excessive, cancel and resume with four
workers. If the job reaches its time limit, submit the same command again; it
will skip completed patients.

## 4. Completion checks

```bash
python -m json.tool "$data_root/new_patches/generation_summary.json"

find "$data_root/new_patches/train/raw" -name '*_data.nii.gz' | wc -l
find "$data_root/new_patches/val/raw"   -name '*_data.nii.gz' | wc -l
find "$data_root/new_patches/test/raw"  -name '*_data.nii.gz' | wc -l
```

For the expected 95/20/21 split, the three counts should be 91,200, 19,200,
and 20,160. `generation_summary.json` must report `complete: true`, 136 completed
patients, and 130,560 indexed patches. It also reports the new image mean and
foreground fraction by split so training normalization can be updated rather
than silently reusing the legacy selected-patch statistic.

Do not delete or rename the legacy dataset until existing jobs have finished
and the new dataset has passed loader and graph-alignment checks.
