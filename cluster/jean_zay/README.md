# Jean Zay setup and execution

This directory targets the V100 environment that most closely reproduces the
working Gardenia stack:

- Python 3.10.4;
- PyTorch 1.12.1 supplied by Jean Zay;
- CUDA 11.2 supplied by Jean Zay;
- one V100 GPU per task;
- project account `vnc@v100`.

The virtual environment lives outside Git at
`$WORK/venvs/vascular-graph-extraction`; the repository `.venv` entry is only a
symbolic link. The dataset, initial checkpoint, and outputs live under
`$SCRATCH`.

## 1. Expected paths

Recommended layout:

```text
$WORK/projects/Graph-Native-Betti-Matching
$WORK/venvs/vascular-graph-extraction
$WORK/tools/uv
$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI
$SCRATCH/checkpoints/model_epoch_280
$SCRATCH/experiments
```

Set the real paths on the Jean Zay login node:

```bash
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI"
export GNBM_MRI_CHECKPOINT="$SCRATCH/checkpoints/model_epoch_280"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"
```

Confirm the dataset root contains `train/`, `val/`, and `test/` before running.

## 2. One-time environment setup

From the repository root on a login node:

```bash
bash cluster/jean_zay/setup_environment.sh
```

This is the only step that downloads Python packages. It does not request a GPU
or compile CUDA code. PyTorch is intentionally inherited from the
`pytorch-gpu/py3/1.12.1` module and must not be installed into the venv.

For an interactive shell after setup:

```bash
source cluster/jean_zay/env.sh
```

## 3. Bounded development test

Submit exactly one 45-minute V100 development job:

```bash
bash cluster/jean_zay/submit_debug.sh
```

The job uses `qos_gpu-dev`, builds deformable attention for V100 (`sm_70`),
runs CUDA forward/backward tests, checks the real checkpoint and MRI loader, and
trains one batch with the complete focal + H0/H1 objective. It writes one
checkpoint so resume integrity can be checked without duplicating large files.
After a successful installation, later debug jobs reuse the extension. Set
`GNBM_FORCE_REBUILD_OPS=1` before submission only when its C++/CUDA source or
the PyTorch/CUDA environment changes.

Monitor it with the commands printed by the submission wrapper, or:

```bash
squeue -u "$USER"
tail -f "$WORK/logs/graph-native-betti-matching/gnbm-debug-JOB_ID.out"
```

Cancel a mistaken job promptly:

```bash
scancel JOB_ID
```

## 4. Production training

Default 20-hour QoS:

```bash
export GNBM_INITIAL_WEIGHTS="$GNBM_MRI_CHECKPOINT"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"
bash cluster/jean_zay/submit_train.sh \
  configs/experiments/finetune_mri/baseline.yaml \
  finetune_mri_baseline_seed364505
```

For a V100 job longer than 20 hours and no longer than 100 hours:

```bash
export GNBM_QOS=qos_gpu-t4
export GNBM_WALLTIME=48:00:00
```

then invoke the same submission command. Long jobs should use a configuration
with interval checkpoints; `best_only` is not interruption-safe.

Resume a complete project checkpoint in a new job:

```bash
unset GNBM_INITIAL_WEIGHTS
export GNBM_RESUME_CHECKPOINT="$GNBM_OUTPUT_DIR/RUN_NAME/models/checkpoint_epoch=EPOCH.pt"
bash cluster/jean_zay/submit_train.sh CONFIG RUN_NAME
```

`--resume` restores model, optimizer, scheduler, epoch, and iteration. A legacy
checkpoint such as `model_epoch_280` is initialization only and must be supplied
through `GNBM_INITIAL_WEIGHTS`.

## Accounting and development QoS

The development job requests one GPU for 45 minutes. `qos_gpu-dev` is the
official QoS for code development and execution tests. It still consumes the
project's allocated GPU time, so use it only for bounded checks and cancel a
stuck or obviously incorrect job. Production training must use `qos_gpu-t3` or
`qos_gpu-t4`, never `qos_gpu-dev`.
