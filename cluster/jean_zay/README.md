# Jean Zay setup and execution

For the complete training, evaluation, environment-variable, CLI, YAML, and
checkpoint reference, see [`../../docs/RUNNING_THE_MODEL.md`](../../docs/RUNNING_THE_MODEL.md).

This directory currently provides production launchers for Jean Zay's H100
partition and separate environment setup for the A100 migration:

- `arch/h100` architecture modules;
- PyTorch 2.3.1 supplied by Jean Zay;
- the CUDA 12 runtime supplied with that PyTorch module;
- one to four H100 GPUs on one node through torchrun/DDP;
- project account `vnc@h100` on partition `gpu_p6`.

The A100 environment uses `arch/a100`, PyTorch 2.3.0 with CUDA 12.2,
`$WORK/venvs/vascular-graph-extraction-a100-torch230`, and an independent
`sm_80` extension cache. Create and activate it with:

```bash
bash cluster/jean_zay/setup_environment_a100.sh
source cluster/jean_zay/env_a100.sh
```

Do not submit the H100 Slurm scripts from the A100 environment. Dedicated A100
launchers must use account `vnc@a100`, partition `gpu_p5`, constraint `a100`,
and either `qos_gpu_a100-dev` (two-hour limit) or `qos_gpu_a100-t3`
(twenty-hour limit).

After creating the A100 environment, run the migration smoke test and then the
full-data scaling benchmark:

```bash
source cluster/jean_zay/env_a100.sh
bash cluster/jean_zay/submit_debug_a100.sh
bash cluster/jean_zay/submit_a100_full_dataset_benchmark.sh
```

Do not launch the production gamma sweep until both checks pass.

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
$SCRATCH/datasets/syntheticMRI/new_patches
$SCRATCH/checkpoints/checkpoint_epoch=280.pt
$SCRATCH/experiments/gnbm
```

Set the real paths on the Jean Zay login node:

```bash
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches"
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
source cluster/jean_zay/env_h100.sh
```

Log in to W&B once from the login node after activating the environment:

```bash
wandb login --verify
```

`wandb login` stores the credential outside the repository. Alternatively,
export `WANDB_API_KEY` in the submitting shell; never put the key in YAML or a
committed shell script. The non-secret entity and project are defined in
`cluster/jean_zay/wandb_env.sh`, which is loaded automatically by `env_h100.sh` in
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

then invoke the same submission command. Long jobs should configure
`training.checkpoint.latest_interval_epochs`. This writes an atomic,
replace-in-place `latest_checkpoint.pt` independently of best-model selection.

Choose the total GPU count before submission. The validated launcher supports
one, two, or four GPUs on one node:

```bash
export GNBM_GPUS=4
```

`data.batch_size` and `GNBM_BATCH_SIZE` are per GPU. When comparing against a
global batch of 32, use 8 per GPU on four GPUs.

Resume a complete project checkpoint in a new job:

```bash
unset GNBM_INITIAL_WEIGHTS
export GNBM_RESUME_CHECKPOINT="$GNBM_OUTPUT_DIR/RUN_NAME/models/latest_checkpoint.pt"
bash cluster/jean_zay/submit_train.sh CONFIG RUN_NAME
```

`--resume` restores model, optimizer, scheduler, epoch, and iteration. The
original `checkpoint_epoch=280.pt` is initialization only and must be supplied
through `GNBM_INITIAL_WEIGHTS`.

Alternatively, reuse the same run name and let the launcher select its latest
checkpoint:

```bash
unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT
export GNBM_AUTO_RESUME=1
bash cluster/jean_zay/submit_train.sh CONFIG RUN_NAME
```

New checkpoints also preserve per-rank Python, NumPy, PyTorch, CUDA, and loader
generator states plus best-metric selection state. Resume with the same GPU
count and global batch size for the closest continuation.

The run directory stores `wandb-run.json` beside the resolved configuration.
Reusing the same run name with `GNBM_RESUME_CHECKPOINT` reuses that W&B run ID,
so online metrics continue in the original cloud run. To buffer a job locally,
set `WANDB_MODE=offline` before submission. W&B does not support run resumption
while offline; sync the buffered data before continuing that run online.
The focal-matrix launchers always select offline mode because Jean Zay H100
compute nodes cannot reach wandb.ai.
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

The generic training wrapper also accepts `qos_gpu_h100-dev` for bounded DDP
smoke tests, with a local safety limit of two H100s and two hours. Full-data
training continues to require `t3` or `t4`.
