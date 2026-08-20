# Jean Zay setup and execution

This directory targets Jean Zay's dedicated H100 partition:

- `arch/h100` architecture modules;
- PyTorch 2.3.1 supplied by Jean Zay;
- the CUDA 12 runtime supplied with that PyTorch module;
- one H100 80 GB GPU per task;
- project account `vnc@h100` on partition `gpu_p6`.

The virtual environment lives outside Git at
`$WORK/venvs/vascular-graph-extraction-h100-torch231`; the repository `.venv`
entry is only a symbolic link. The dataset, initial checkpoint, and outputs
live under `$SCRATCH`.

## 1. Expected paths

Recommended layout:

```text
$WORK/projects/Graph-Native-Betti-Matching
$WORK/venvs/vascular-graph-extraction-h100-torch231
$WORK/tools/uv
$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI
$SCRATCH/checkpoints/checkpoint_epoch=280.pt
$SCRATCH/experiments/gnbm
```

Set the real paths on the Jean Zay login node:

```bash
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI"
export GNBM_MRI_CHECKPOINT="$SCRATCH/checkpoints/checkpoint_epoch=280.pt"
export GNBM_INITIAL_WEIGHTS="$GNBM_MRI_CHECKPOINT"
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
H100-specific `pytorch-gpu/py3/2.3.1` module and must not be installed into the
venv.

For an interactive shell after setup:

```bash
source cluster/jean_zay/env.sh
```

Log in to W&B once from the login node after activating the environment:

```bash
wandb login --verify
```

`wandb login` stores the credential outside the repository. Alternatively,
export `WANDB_API_KEY` in the submitting shell; never put the key in YAML or a
committed shell script. The non-secret entity and project are defined in
`cluster/jean_zay/wandb_env.sh`, which is loaded automatically by `env.sh` in
both interactive and batch environments. The configured project is:

```text
https://wandb.ai/alessandrobenvenuti2002-politecnico-di-torino/focal-loss
```

## 3. Bounded development test

Submit exactly one 45-minute H100 development job:

```bash
bash cluster/jean_zay/submit_debug.sh
```

The job uses `qos_gpu_h100-dev`, builds deformable attention for H100 (`sm_90`),
runs CUDA forward/backward tests, checks the real checkpoint and MRI loader, and
trains one batch with the complete focal + H0/H1 objective. It writes one
checkpoint so resume integrity can be checked without duplicating large files.
W&B is disabled for this smoke job unless `WANDB_MODE` is explicitly supplied.
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

For an H100 job longer than 20 hours and no longer than 100 hours:

```bash
export GNBM_QOS=qos_gpu_h100-t4
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

`--resume` restores model, optimizer, scheduler, epoch, and iteration. The
original `checkpoint_epoch=280.pt` is initialization only and must be supplied
through `GNBM_INITIAL_WEIGHTS`.

The run directory stores `wandb-run.json` beside the resolved configuration.
Reusing the same run name with `GNBM_RESUME_CHECKPOINT` reuses that W&B run ID,
so online metrics continue in the original cloud run. To buffer a job locally,
set `WANDB_MODE=offline` before submission. W&B does not support run resumption
while offline; sync the buffered data before continuing that run online.
Afterwards, activate the environment on the login node and upload it with:

```bash
wandb sync "$GNBM_OUTPUT_DIR/RUN_NAME/wandb/offline-run-"*
```

Use `WANDB_MODE=disabled` as an operational override for an untracked test.

## Accounting and development QoS

The development job requests one H100 for 45 minutes. `qos_gpu_h100-dev` is the
official QoS for code development and execution tests. It still consumes the
project's allocated GPU time, so use it only for bounded checks and cancel a
stuck or obviously incorrect job. Production training must use
`qos_gpu_h100-t3` or `qos_gpu_h100-t4`, never `qos_gpu_h100-dev`.
