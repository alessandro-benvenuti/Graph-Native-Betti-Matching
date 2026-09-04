# Running the 3D RelationFormer model

This guide describes the supported training and evaluation interfaces in this
repository, with Jean-Zay H100 examples. It separates three kinds of settings:

1. environment variables used by shell launchers and YAML expansion;
2. command-line arguments accepted by `train.py` and `evaluate.py`;
3. experiment settings stored in YAML files.

The checked-in experiment YAML should normally be the scientific record. Use
CLI and environment overrides mainly for paths, resources, and run identity.

## 1. Jean-Zay directory layout

The expected layout is:

```text
$WORK/projects/Graph-Native-Betti-Matching       repository
$WORK/venvs/vascular-graph-extraction-h100-torch231
$WORK/logs/graph-native-betti-matching           Slurm stdout/stderr
$SCRATCH/datasets/plants_3d2cut/patches_3d       Plants data
$SCRATCH/datasets/syntheticMRI/new_patches_boundary  boundary-corrected MRI patches
$SCRATCH/experiments/gnbm                        run outputs/checkpoints
```

Never run training or GPU evaluation on a login node. The login node is only
for preparing the environment, validating configuration, submitting jobs,
checking logs, and synchronizing offline W&B runs.

Initialize a login shell with:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
source cluster/jean_zay/env.sh

export PLANTS_DATASET="$SCRATCH/datasets/plants_3d2cut/patches_3d"
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"
```

Run `bash cluster/jean_zay/setup_environment.sh` once if the external virtual
environment has not been created. The environment inherits Jean-Zay's PyTorch
2.3.1 H100 module; do not install another PyTorch into it.

## 2. Configuration composition

A YAML file may inherit one or more other YAML files:

```yaml
defaults:
  - ../../finetune_synthetic_mri.yaml
  - ../../losses/node_focal_unweighted_immediate.yaml
```

Parents are loaded in order, then the child recursively overrides mappings.
Relative paths are resolved from the YAML containing `defaults`. Strings such
as `${SYNTHETIC_MRI_DATASET}` are expanded when the config is loaded; a missing
environment variable is an error. The final configuration is validated before
model or dataset construction and is saved as `resolved-config.yaml`.

## 3. Training from Python

The complete direct interface is:

```text
python train.py --config CONFIG
                [--resume CHECKPOINT]
                [--initial-weights CHECKPOINT]
                [--device DEVICE]
                [--output-dir DIRECTORY]
                [--run-name NAME]
                [--batch-size INTEGER]
                [--workers INTEGER]
                [--distributed]
```

Arguments:

| Argument | Meaning |
|---|---|
| `--config` | Required composed YAML recipe. |
| `--initial-weights` | Load model parameters only. Optimizer, scheduler, epoch, iteration, random states, patience, and best-model state start anew. Use this to initialize a new stage or experiment. |
| `--resume` | Restore complete training state. It is mutually exclusive with `--initial-weights`. Resume requires the same distributed world size. |
| `--device` | Override `runtime.device`, normally `cuda` or `cpu`. |
| `--output-dir` | Override the parent output directory. |
| `--run-name` | Override `experiment.name`; this is the subdirectory and W&B run name. |
| `--batch-size` | Override the **per-process/per-GPU** training batch. |
| `--workers` | Override DataLoader workers per process. Zero is valid. |
| `--distributed` | Enable DDP. Invoke it through `torchrun`; the Jean-Zay launcher does this automatically for multiple GPUs. |

Do not invoke multi-GPU `train.py` manually on Jean-Zay unless reproducing what
`train_h100.slurm` does. Use the submission wrapper below.

## 4. Generic Jean-Zay training launcher

Submit one training stage with:

```bash
bash cluster/jean_zay/submit_train.sh CONFIG_PATH UNIQUE_RUN_NAME
```

Required variables:

| Variable | Allowed/expected value | Purpose |
|---|---|---|
| `WORK` | Jean-Zay path | Supplied by Jean-Zay. Holds repository, venv and logs. |
| `SCRATCH` | Jean-Zay path | Supplied by Jean-Zay. Holds datasets and run outputs. |
| `SYNTHETIC_MRI_DATASET` | dataset root | Must contain materialized `train`, `val`, and `test` splits with `raw`, `seg`, and `vtp` directories. Current runs use `.../syntheticMRI/new_patches_boundary`. The sibling `new_patches.csv` is provenance metadata; the loader does not reconstruct splits from it. |
| `GNBM_OUTPUT_DIR` | directory | Parent of every run. Put it under `$SCRATCH`. |

Optional launcher variables:

| Variable | Values/default | Purpose |
|---|---|---|
| `GNBM_GPUS` | `1`, `2`, or `4`; default `1` | H100s on one node. |
| `GNBM_BATCH_SIZE` | positive integer | Per-GPU batch passed to `train.py`. A global batch of 32 uses `8` with four GPUs. |
| `GNBM_WORKERS` | non-negative integer | DataLoader workers per process. |
| `GNBM_QOS` | `qos_gpu_h100-dev`, `qos_gpu_h100-t3`, or `qos_gpu_h100-t4`; default `t3` | Queue class. Development is limited by this wrapper to at most two GPUs and two hours; `t3` to 20 hours; `t4` to 100 hours. |
| `GNBM_WALLTIME` | `HH:MM:SS`; default `20:00:00` | Slurm time limit. Use `100:00:00` for the maximum supported `t4` request. |
| `GNBM_INITIAL_WEIGHTS` | checkpoint path or unset | Model-only initialization. Cannot coexist with `GNBM_RESUME_CHECKPOINT`. |
| `GNBM_RESUME_CHECKPOINT` | full checkpoint path or unset | Explicit restoration of model, optimizer, scheduler, runtime states and training control state. |
| `GNBM_AUTO_RESUME` | `0` or `1`; default `0` | With `1`, reuse `RUN_NAME/models/latest_checkpoint.pt` if present. Explicit resume takes precedence, followed by auto-resume, then initial weights. |
| `GNBM_VENV` | path | Override the external project venv. |
| `GNBM_REPO_DIR` | path | Override the repository path used by the batch job. Normally set automatically. |

`GNBM_GPUS_PER_NODE` is an internal batch-launch variable and should normally
equal `GNBM_GPUS`. The current launcher supports one node only.

Additional repository scripts expose a few script-specific controls:

| Variable | Used by | Meaning |
|---|---|---|
| `GNBM_BOUNDARY_SWEEP_DRY_RUN` | `submit_boundary_gamma_sweep_500.sh` | `1` validates and prints the current five-pipeline sweep without submitting; `0` submits. |
| `GNBM_PRETRAIN_QOS`, `GNBM_FINETUNE_QOS` | current boundary sweep and historical focal/edge matrix launchers | Separate stage QoS values. |
| `GNBM_MATRIX_DRY_RUN` | `submit_focal_matrix_600.sh` | `1` validates and prints the matrix without submitting it; `0` submits. |
| `GNBM_ABLATION_DRY_RUN` | `submit_edge_candidate_ablation_600.sh` | Equivalent dry-run switch for the edge-candidate ablation. |
| `GNBM_MRI_CHECKPOINT` | debug smoke launcher | Required model checkpoint used by the bounded integration test. |
| `GNBM_FORCE_REBUILD_OPS` | CUDA-extension build script | `1` forces a rebuild; leave unset unless source or the PyTorch/CUDA environment changed. |

`env.sh` also permits overriding `UV_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`,
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `TMPDIR`, `TORCH_CUDA_ARCH_LIST`,
`OMP_NUM_THREADS`, and `MAX_JOBS`. These are runtime/cache/build controls, not
experiment hyperparameters. The checked defaults are appropriate for the
Jean-Zay H100 environment.

Example: a fresh four-H100 run initialized from model weights:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
source cluster/jean_zay/env.sh

export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"
export GNBM_INITIAL_WEIGHTS="$SCRATCH/checkpoints/checkpoint_epoch=280.pt"
unset GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

export GNBM_GPUS=4
export GNBM_BATCH_SIZE=8
export GNBM_QOS=qos_gpu_h100-t4
export GNBM_WALLTIME=100:00:00
export WANDB_MODE=offline
export WANDB_RUN_GROUP=example-full-mri

bash cluster/jean_zay/submit_train.sh \
  configs/experiments/full_dataset_node_focal/finetune.yaml \
  example_full_mri_node_focal_seed364505
```

The wrapper prints the job ID and log path. It performs only preflight checks
on the login node; training runs in Slurm.

## 5. Current boundary-data gamma sweep

The current controlled experiment set contains five two-stage pipelines:

1. baseline;
2. node focal with node gamma `2.0`;
3. node focal plus matched--matched edge focal with edge gamma `0.5`;
4. the same with edge gamma `1.0`;
5. the same with edge gamma `2.0`.

Each pipeline performs 100 mixed-pretraining epochs followed by up to 500
MRI-only fine-tuning epochs. Fine-tuning uses validation edge mAP for
50-epoch patience. Both stages retain rolling best-edge-mAP, best-node-F1,
and best-edge-F1 checkpoints. Betti losses are disabled and have weight zero.

Submit the complete set with:

```bash
bash cluster/jean_zay/submit_boundary_gamma_sweep_500.sh
```

The launcher defaults to one H100 for pretraining, four H100s for fine-tuning,
the `qos_gpu_h100-t4` QoS, and a 100-hour limit for both stages. Override these
with `GNBM_PRETRAIN_GPUS`, `GNBM_FINETUNE_GPUS`, `GNBM_PRETRAIN_QOS`,
`GNBM_FINETUNE_QOS`, `GNBM_PRETRAIN_WALLTIME`, and
`GNBM_FINETUNE_WALLTIME`. Set `GNBM_BOUNDARY_SWEEP_DRY_RUN=1` to validate and
print all ten stages without submitting them.

The launcher forces W&B offline mode because Jean-Zay compute nodes cannot
reach W&B. Synchronize the run directory from a login node after completion.

The dependent MRI job uses the matching pretraining run's
`best_metric_checkpoint.pt`, selected by maximum validation edge mAP, as
initial weights.

### Development-QoS batch-size benchmark

Before committing the full boundary-data sweep to a larger global batch, run
the controlled batch-8 versus batch-32 benchmark:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
source cluster/jean_zay/env.sh
bash cluster/jean_zay/submit_h100_batch_benchmark.sh
```

Both cases use two H100s under `qos_gpu_h100-dev`, the same seeded subset of
4,096 MRI training patches, 64 validation patches, two epochs, and the gamma-2
node+edge-focal recipe. They compare per-GPU batches 8 and 32 (global batches 16
and 64), preserving the fourfold per-GPU batch change intended for the eventual
four-H100 run without charging a normal training-QoS benchmark. The second epoch
is the representative warmed-up measurement. The submission command prints a report directory below
`$WORK/logs/graph-native-betti-matching/batch-benchmarks/`. Its
`comparison.txt` contains throughput and epoch-time speedups; each case also
preserves `performance.jsonl`, peak allocation, two-second GPU telemetry,
allocated GPU-hours, resolved configuration, dataset manifest, train log, and
software/hardware provenance.

### Historical two-stage pipelines

These launchers submit limited-MRI mixed pretraining and dependent full-MRI
fine-tuning automatically:

```bash
# Node-focal pipeline
bash cluster/jean_zay/submit_full_dataset_node_focal.sh

# Baseline and node-focal + edge-focal matched--matched pipelines
bash cluster/jean_zay/submit_full_dataset_comparisons.sh
```

They additionally accept:

| Variable | Values/default |
|---|---|
| `PLANTS_DATASET` | Required Plants root. |
| `GNBM_PRETRAIN_GPUS` | `1`, `2`, or `4`; default `1`. |
| `GNBM_FINETUNE_GPUS` | `1`, `2`, or `4`; default `4`. |
| `GNBM_PRETRAIN_WALLTIME` | Default `48:00:00`. |
| `GNBM_FINETUNE_WALLTIME` | Default `100:00:00`. |
| `WANDB_RUN_GROUP` | Group name; each launcher has a descriptive default. |

The dependent MRI job uses the pretraining run's
`best_metric_checkpoint.pt` as **initial weights**, not as a resume checkpoint.
Launchers refuse to overwrite an existing run directory. Choose a new
`GNBM_OUTPUT_DIR` for a clean repeated matrix, for example:

```bash
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm/full-data-250-$(date +%Y%m%d_%H%M%S)"
```

## 6. Checkpoints and outputs

A run is stored in:

```text
GNBM_OUTPUT_DIR/experiment.name/
```

Important files are:

| File | Meaning |
|---|---|
| `resolved-config.yaml` | Exact effective configuration. |
| `dataset-manifest.json` | Exact dataset membership and sampling provenance. |
| `validation-metrics.jsonl` | Validation metrics by selected evaluation epoch. |
| `best_checkpoint.pt` | Lowest validation total loss, when the policy includes best-loss saving. |
| `best_metric_checkpoint.pt` | Best configured task metric, commonly validation edge mAP. |
| `best_node_f1_checkpoint.pt` | Best node micro-F1 checkpoint when enabled. |
| `best_edge_f1_checkpoint.pt` | Best edge micro-F1 checkpoint when enabled. |
| `latest_checkpoint.pt` | Replace-in-place recovery state, independent of model selection. |
| `checkpoint_epoch=N.pt` | Periodic archive when an interval policy is selected. |
| `best-metric.json`, `best-node-f1.json`, `best-edge-f1.json` | Winner epoch, iteration, value and checkpoint provenance. |
| `early-stopping.json` | Current patience state. |
| `training-status.json` | Terminal epoch and either `early_stopping` or `max_epochs`. |
| `wandb-run.json` | W&B identity/provenance. |

All checkpoints are written below `GNBM_OUTPUT_DIR`, so setting that variable
under `$SCRATCH` keeps the large files off `$WORK`. Best/latest names are
replace-in-place slots; they do not accumulate one file per improvement.

Resume after interruption with the **latest**, not the best, checkpoint:

```bash
unset GNBM_INITIAL_WEIGHTS GNBM_AUTO_RESUME
export GNBM_RESUME_CHECKPOINT="$GNBM_OUTPUT_DIR/RUN_NAME/models/latest_checkpoint.pt"

bash cluster/jean_zay/submit_train.sh CONFIG_PATH RUN_NAME
```

Resume is a continuation of the same configured schedule. Changing total
epochs, scheduler semantics, selection metric, patience configuration, GPU
count, or global batch while resuming is not automatically a controlled
extension experiment.

## 7. Evaluation and prediction export

The complete evaluation interface is:

```text
python evaluate.py --config CONFIG --checkpoint CHECKPOINT --output-dir DIR
                   [--dataset NAME]
                   [--split val|test]
                   [--max-samples N | --sample-id ID ... | --sample-list FILE]
                   [--batch-size N] [--workers N] [--device DEVICE]
                   [--node-threshold P] [--edge-threshold P]
                   [--bn-calibration-batches N]
                   [--visualizations N]
                   [--no-export-predictions]
```

| Argument | Meaning |
|---|---|
| `--config` | Required model/dataset/evaluation recipe. It must match the checkpoint architecture. |
| `--checkpoint` | Required model checkpoint. Any of the selected checkpoint slots can be evaluated. |
| `--output-dir` | Required new evaluation directory. |
| `--dataset` | Configured dataset key. It may be omitted when exactly one dataset has role `target`. |
| `--split` | `val` or `test`; default `val`. Use validation to choose models/thresholds and test only for the final estimate. |
| `--max-samples` | Evaluate a positive-size prefix; mutually exclusive with sample-ID selection. Omit for the complete split. |
| `--sample-id` | Exact source dataset ID. Repeat the option to select several patches. |
| `--sample-list` | Text file with one source ID per line; blank lines and `#` comments are ignored. |
| `--batch-size` | Evaluation batch override; it also sets `validation_batch_size`. |
| `--workers` | DataLoader worker override. |
| `--device` | Normally `cuda`; `cpu` is possible but slow. |
| `--node-threshold` | Object probability in `[0,1]`. Without it, inference keeps node-class argmax predictions. |
| `--edge-threshold` | Edge probability in `[0,1]`. Without it, inference keeps edge-class argmax predictions. |
| `--bn-calibration-batches` | Non-negative number of validation batches used to update BatchNorm statistics before inference. Zero disables it. |
| `--visualizations` | Number of initial samples for static GT/prediction PNGs. Default `0`. |
| `--no-export-predictions` | Skip the potentially large `predictions.json`. |

Evaluation writes `summary.json`, `per-patch-metrics.csv`, `predictions.json`,
`metadata.json`, `resolved-config.yaml`, and optionally `plots/*.png`.
`predictions.json` is the lossless graph export used by interactive 3D tools.

Once predictions have been exported, generate a self-contained interactive
3D report for one source patch with:

```bash
python scripts/visualize_graph_prediction_3d.py \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --split test \
  --predictions "$EVAL_OUTPUT/predictions.json" \
  --sample-id sample_000109_0021 \
  --output-dir "$EVAL_OUTPUT/interactive" \
  --node-threshold 0.5 \
  --edge-threshold 0.5 \
  --error-analysis
```

Omit `--sample-id` to list the available evaluation/source IDs. The script can
also select `--evaluation-id`, hide individual MRI/segmentation/GT/prediction
layers, disable HTML with `--no-html`, or request a static image with `--png`.
The error overlay is visualization-time distance matching, not the training
Hungarian assignment. See [`GRAPH_PREDICTION_3D.md`](GRAPH_PREDICTION_3D.md)
for its complete interface and gallery workflow.

Example Slurm test-set evaluation:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
source cluster/jean_zay/env.sh

export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"

export EVAL_CONFIG="configs/experiments/full_dataset_node_focal/finetune.yaml"
export EVAL_CHECKPOINT="$GNBM_OUTPUT_DIR/RUN_NAME/models/best_metric_checkpoint.pt"
export EVAL_OUTPUT="$GNBM_OUTPUT_DIR/RUN_NAME/evaluation/test-best-edge-map"

sbatch \
  --job-name=gnbm-test \
  --account=vnc@h100 \
  --partition=gpu_p6 \
  --constraint=h100 \
  --qos=qos_gpu_h100-dev \
  --time=02:00:00 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=10 \
  --output="$WORK/logs/graph-native-betti-matching/%x-%j.out" \
  --error="$WORK/logs/graph-native-betti-matching/%x-%j.err" \
  --export=ALL,EVAL_CONFIG,EVAL_CHECKPOINT,EVAL_OUTPUT \
  --wrap='source "$WORK/projects/Graph-Native-Betti-Matching/cluster/jean_zay/env.sh" && "$GNBM_VENV/bin/python" -u evaluate.py --config "$EVAL_CONFIG" --checkpoint "$EVAL_CHECKPOINT" --output-dir "$EVAL_OUTPUT" --dataset synthetic_mri --split test --batch-size 32 --workers 4 --device cuda --visualizations 16'
```

If the complete test evaluation exceeds two hours, use `qos_gpu_h100-t3` and
an appropriate time up to `20:00:00`. Evaluation does not modify the checkpoint.

## 8. W&B settings

The repository stores no secret. Authenticate once on the login node:

```bash
source cluster/jean_zay/env.sh
wandb login --verify
```

| Variable | Values/default | Purpose |
|---|---|---|
| `WANDB_ENTITY` | default from `wandb_env.sh` | Destination entity. |
| `WANDB_PROJECT` | default `focal-loss` | Destination project. |
| `WANDB_RUN_GROUP` | any non-empty name | Groups related runs. The correct variable is not `WANDB_GROUP`. |
| `WANDB_MODE` | `online`, `offline`, `disabled`, or `shared` | Compute nodes have no internet, so production jobs normally use `offline`. |
| `WANDB_API_KEY` | secret, optional environment variable | Alternative authentication. Never save it in Git, YAML, or submission scripts. |

YAML `tracking.entity`, `project`, `group`, and `mode` describe the run; the
tracker also observes the standard W&B environment. The Jean-Zay full-matrix
launchers force offline mode unless explicitly configured before submission.
Synchronize completed offline runs from the login node:

```bash
bash cluster/jean_zay/sync_wandb_offline.sh "$GNBM_OUTPUT_DIR"
```

## 9. YAML reference

### Experiment and runtime

| Key | Supported values and meaning |
|---|---|
| `schema_version` | Must be `1`. |
| `experiment.name` | Non-empty run identity after launcher/CLI override. |
| `experiment.seed` | Non-negative integer. Rank is added for per-process random seeds. |
| `experiment.output_dir` | Parent output directory, normally `${GNBM_OUTPUT_DIR}`. |
| `runtime.device` | PyTorch device such as `cuda` or `cpu`. DDP assigns `cuda:LOCAL_RANK`. |
| `runtime.distributed` | Boolean. Normally enabled by `--distributed`, not manually in production YAML. |
| `runtime.deterministic` | Boolean; enables PyTorch deterministic algorithms and may reject nondeterministic operations. |
| `runtime.workers` | Non-negative DataLoader workers per process. |
| `runtime.pin_memory` | Boolean DataLoader pinned-memory control. |

### Tracking

| Key | Supported values |
|---|---|
| `tracking.enabled` | Boolean. |
| `tracking.project` | Non-empty string. |
| `tracking.entity`, `tracking.group` | `null` or non-empty string. |
| `tracking.mode` | `null`, `online`, `offline`, `disabled`, or `shared`. |
| `tracking.tags` | List of non-empty strings. |
| `tracking.save_code` | Boolean. |

### Data

| Key | Supported values and meaning |
|---|---|
| `data.spatial_dims` | Must be `3`. |
| `data.image_size` | Three positive integers in D/H/W order. Current experiments use `[64,64,64]`. |
| `data.batch_size` | Positive per-process training batch. |
| `data.validation_batch_size` | Optional positive validation batch. |
| `data.train_augmentation` | Boolean. |
| `data.mixed_sampling.balance_source_target` | Boolean. With source and target datasets, balances domain sampling when enabled. |
| `data.datasets` | Mapping containing `plants` and/or `synthetic_mri`; at least one dataset must have role `target`. |
| `data.datasets.NAME.role` | `source` or `target`. |
| `data.datasets.NAME.root` | Non-empty path, commonly an expanded environment variable. |
| `train_samples`, `validation_samples` | Positive integer cap or `null` for all discovered samples. |
| `plants.coordinate_order_on_disk` | Must be `[y, x]`. |
| `synthetic_mri.coordinate_space_on_disk` | `normalized` or `voxel`. |
| `synthetic_mri.foreground_mean` | Numeric intensity centering mean. |
| `synthetic_mri.sample_cap_selection` | `first` or `seeded_random`. |
| `synthetic_mri.sample_cap_seed` | Non-negative integer required by `seeded_random`. Selection is deterministic. |

### Augmentation

| Key | Supported values and meaning |
|---|---|
| `augmentation.coordinate_order` | Canonical experiments use `[D,H,W]`. |
| `normalization_denominator` | Must be `axis_size`. |
| `synthetic_mri.rotate_90.enabled` | Boolean; planes list the D/H/W rotation planes. |
| `synthetic_mri.zoom.enabled` | Boolean. |
| `synthetic_mri.zoom.range` | Two numbers `[min,max]`, with `min>0` and `max>=min`. |
| image/segmentation interpolation | Current supported recipe uses `trilinear`/`nearest`. |
| Gaussian-noise `enabled` | Boolean. |
| Gaussian-noise `probability` | Number in `[0,1]`. |
| Gaussian-noise `std_range` | Non-negative increasing pair. |
| `synthetic_mri.intensity_clamp` | `null` or increasing `[low,high]`. |
| `plants.projection_depth` | Positive odd integer. |
| `plants.padding` | Non-negative integer leaving a non-empty inner volume. |
| `plants.flip.probability_per_axis` | Number in `[0,1]`; axes use D/H/W. |
| `plants.intensity_scale` | Current recipe uses `[-0.5,0.5]`. |
| `plants.pad_to_image_size` | Boolean. |

### Model and Hungarian matcher

| Key | Supported values |
|---|---|
| `model.num_classes` | Positive integer; current binary object/background model uses `2`. |
| `encoder.type` | Must be `se_resnet`. |
| `encoder.input_channels` | Positive integer; current volumes use `1`. |
| `encoder.depths`, `encoder.strides` | Four positive integers. |
| `decoder.type` | Must be `deformable_detr`. |
| `decoder.hidden_dim` | Positive and divisible by `attention_heads`. |
| `decoder.attention_heads` | `6` or `26`. |
| `encoder_layers`, `decoder_layers`, `feedforward_dim`, `decoder_points`, `encoder_points`, `object_queries` | Positive integers. |
| `decoder.dropout` | Number in `[0,1)`. |
| `decoder.activation` | `relu`, `gelu`, or `glu`. |
| `decoder.feature_levels` | Must be `1`. |
| `decoder.relation_tokens`, `dummy_tokens` | Non-negative integers. Relation attention needs at least one relation token. |
| `decoder.relation_attention`, `use_cuda_extension` | Boolean. Production H100 runs use the CUDA extension. |
| `matcher.type` | Current implementation uses `hungarian`. |
| `matcher.class_cost`, `matcher.node_cost` | Matcher cost coefficients; use non-negative experimental values. |

Architecture changes are not operational overrides: they generally make old
checkpoints incompatible and require a new controlled training run.

### Training, optimizer and scheduler

| Key | Supported values and meaning |
|---|---|
| `training.input` | `image` or `segmentation`. |
| `training.start_epoch` | Normally `0`; checkpoint resume supplies the actual continuation epoch. |
| `training.epochs` | Positive total epoch count. |
| `training.warmup_epochs` | Scheduler warmup duration. |
| `optimizer.name` | Must be `adamw`. |
| `optimizer.base_lr`, `weight_decay`, `epsilon`, `betas` | AdamW parameters. Use positive epsilon/LR and betas in `[0,1)` in valid experiments. |
| `scheduler.name` | Must be `polynomial`. |
| `scheduler.warmup_lr` | Starting LR during linear warmup. |
| `scheduler.power` | Polynomial-decay exponent. |

The polynomial schedule is iteration-based and its horizon is calculated from
`training.epochs * iterations_per_epoch`. Consequently, total epochs and data
loader length are part of the schedule definition.

### Early stopping

| Key | Supported values |
|---|---|
| `enabled` | Boolean. |
| `monitor` | `edge_mAP`, `node_mAP`, `edge_f1`, `node_f1`, `validation_total`, `beta0_absolute_error`, `beta1_absolute_error`, or `smd`. |
| `mode` | Must be `max` for AP/F1 and `min` for loss, Betti errors, and SMD. |
| `patience_epochs` | Positive integer. Counts epochs, not validation events. |
| `min_epochs` | Non-negative floor; `0` means no minimum. It cannot exceed total epochs when stopping is enabled. |
| `min_delta` | Finite non-negative required improvement. |

Stopping is checked only at validation epochs. With validation every five
epochs, patience 50, no minimum, and the last improvement at epoch 105, the
first stop is epoch 155. Early stopping requires checkpointing; metric-based
stopping also requires training metrics.

### Checkpoint policy

| Key | Supported values |
|---|---|
| `policy` | `none`, `best_only`, `interval`, or `interval_and_best`. |
| `interval_epochs` | Positive interval for archived epoch checkpoints. |
| `latest_interval_epochs` | Positive recovery-checkpoint interval. |

`best_only` refers to validation-loss checkpoint behavior. Task-metric and F1
checkpoint selection is controlled separately below.

### Losses

`loss.enabled` is a subset of `boxes`, `class`, `cardinality`, `nodes`, and
`edges`. `loss.weights` supplies the coefficient for every enabled component.
The total loss is the weighted sum, plus enabled non-log-only topology terms.

Node classification:

| Key | Supported values |
|---|---|
| `loss.node.classification.name` | `weighted_cross_entropy` or `focal`. |
| `class_weights` | Two non-negative values `[background, object]` with positive sum. `[1,1]` is unweighted. |
| `focal_gamma` | Non-negative number. At gamma `0`, the implemented weighted focal loss reduces to weighted CE. |

Edge classification uses the same fields, but names are `cross_entropy` or
`focal`, and weights mean `[no-edge, edge]`.

For either node or edge focal curriculum:

| Key | Supported values |
|---|---|
| `curriculum.enabled` | Boolean. |
| `start_percent`, `end_percent` | Numbers satisfying `0 <= start <= end <= 100`. |

During training, enabled curriculum linearly changes gamma from `0` to
`focal_gamma` between these percentages of total training progress. Validation
always evaluates the configured final gamma.

Edge candidates:

| Key | Supported values and meaning |
|---|---|
| `max_per_graph` | Positive integer or `null`; cap on matched-pair candidates. |
| `positive_cap` | Must currently remain `null` in user configurations. Training internally applies its audited validation behavior. |
| `include_unmatched` | Boolean; include hard negative U-M and U-U pairs. |
| `unmatched_object_threshold` | Object probability in `[0,1]` for active unmatched queries. |
| `max_active_unmatched` | Non-negative cap; highest object probabilities are retained. |
| `max_unmatched_pairs_per_graph` | Non-negative cap on hardest pairs by predicted edge probability; `0` means uncapped. |
| `unmatched_weight` | Non-negative per-candidate weight. |
| `unmatched_warmup_epochs` | Non-negative zero-weight duration. |
| `unmatched_ramp_epochs` | Non-negative linear ramp duration after warmup; zero applies the target weight immediately. |

Edge balancing:

| Key | Supported values |
|---|---|
| `mode` | `none` or `ratio_upsample`. Focal edge loss requires `none`. |
| `positive_to_negative_ratio` | Positive target ratio when upsampling. |
| `tolerance` | Non-negative ratio tolerance. |

### Graph-native topology losses

Both `topology.betti_h0` and `topology.betti_h1` support:

| Key | Meaning |
|---|---|
| `enabled` | Compute the topology loss. |
| `log_only` | If true, log but do not add it to total loss. |
| `weight` | Non-negative final contribution coefficient. |
| `warmup_epochs`, `ramp_epochs` | Non-negative zero-weight and linear-ramp durations. |
| `diagonal_factor` | Non-negative matching parameter. |
| `normalize` | Boolean normalization control. |

H0 additionally has non-negative `unmatched_weight`. H1 additionally has
non-negative `false_positive_weight` and `false_negative_weight`. These losses
operate on the matched graph candidate space implemented by the criterion;
enable them only as explicit topology experiments.

### Evaluation and model selection

| Key | Supported values and meaning |
|---|---|
| `evaluation.interval_epochs` | Positive validation interval. |
| `node_threshold`, `edge_threshold` | `null` or probability in `[0,1]`. `null` uses binary class argmax. |
| `bn_calibration_batches` | Non-negative integer. |
| `training_metrics.enabled` | Evaluate task metrics during training. |
| `training_metrics.dataset` | A configured dataset key. |
| `training_metrics.max_samples` | Positive cap or `null` for the complete validation split. |
| `selection_metric` | Non-empty summary key, commonly `edge_mAP`. |
| `selection_mode` | `max` or `min`, consistent with the metric. |
| `save_best_checkpoint` | Save `best_metric_checkpoint.pt`; requires metrics enabled. |
| `save_f1_checkpoints` | Independently save best node and edge F1; requires metrics and a non-`none` checkpoint policy. |

Protocol fields:

| Key | Supported values |
|---|---|
| `f1_iou_threshold` | Number in `(0,1]`; micro-F1 spatial match threshold. |
| `iou_thresholds` | Strictly increasing probability list in `[0,1]` used by AP/AR. |
| `max_detections` | Positive AP/AR detection cap. |
| `target_node_size`, `edge_half_width` | Positive spatial box sizes used by graph detection metrics. |
| `smd_points`, `smd_iterations` | Positive SMD sampling/optimization counts. |
| `smd_epsilon` | Positive numerical stabilizer. |
| `folds` | Positive fold count used by the inherited metric aggregation. |

The F1 values are micro-F1 over all retained detections at the configured
confidence thresholds and `f1_iou_threshold`; they are not the harmonic mean
of mAP and mAR. Selecting a model, selecting confidence thresholds, and
reporting a final test score are distinct operations.

## 10. Monitoring and diagnosis

Queue:

```bash
squeue -u "$USER" -o "%.18i %.42j %.10T %.12M %.14l %.80R"
```

Logs:

```bash
tail -f "$WORK/logs/graph-native-betti-matching/JOB_NAME-JOB_ID.out"
tail -f "$WORK/logs/graph-native-betti-matching/JOB_NAME-JOB_ID.err"
```

Accounting after a job leaves the queue:

```bash
sacct -j JOB_ID \
  --format=JobID,JobName%42,State,ExitCode,Elapsed,MaxRSS,AllocTRES%60
```

Successful completion is `COMPLETED` with exit code `0:0`. `PENDING` consumes
no GPU time. A dependent fine-tuning job remains pending until pretraining ends
successfully; maintenance and priority reasons are scheduler conditions, not
model failures.
