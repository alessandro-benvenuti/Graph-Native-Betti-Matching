# H0-focused refinement of node + edge focal (edge gamma 0.5)

This study reuses the existing focal-only 500→750 continuation as its control.
It never submits or modifies that control. Every candidate starts from a frozen
copy of the original **epoch-500 latest checkpoint**, retaining AdamW moments.
The first launch deliberately restarts the same 250-epoch polynomial tail at
1e-5 and resets the same edge-mAP patience. The original checkpoint has world
size 2; the new branch explicitly allows the change to the control's world size
4. Subsequent segments and promotions restore optimizer, scheduler, patience,
and four-rank runtime state strictly, without any reset.

The implementation clones the actual control's resolved configuration and
checks its focal recipe, data, epoch-500 restart provenance (`extension-source.json`),
and dataset manifest. Preparation copies the source checkpoint, verifies its
checksum, and freezes the inputs. An incompatible existing control is an error,
not a reason to silently launch a second control. Nothing reads the test split.

## Search and finite promotions

Scientific settings are in `configs/experiments/focal_h0_study.yaml`:

- 8 H0-only trials: logarithmic H0 weight in [0.001, 0.1], with queued anchors
  0.001, 0.003, 0.01, 0.03, then seeded multi-objective TPE proposals.
- Node focal gamma 2, matched-only edge focal gamma 0.5, no ratio upsampling.
- Activation at completed epoch 500, fixed 10-epoch ramp, no extra warmup.
- H0 on every eligible local graph (batch 8 per rank). H1 is disabled, including
  validation H1 computation. This does not disable ordinary graph beta1 metrics.
- Four A100s per trial, global batch 32. Two concurrent workers by default.
- All 8 candidates to 525; manually promote 4 to 550, then 2 to 600.
- Compare exact control tails [515,520,525], [540,545,550], [590,595,600].
- Objectives: maximize node/edge mAP, minimize absolute beta0/beta1 errors.
  No scalar score, hard degradation threshold, or automatic "best" selection.

Those H0-only promotions remain available but are not automatic. Following the
observed H0/H1 trade-off at 525, the agreed next budget is the H1 ablation below,
not an obligatory H0-only continuation to 550.

Only the first-budget values enter Optuna. Promotions produce separate Pareto
reports and never overwrite those values or mix different budgets in a ranking.
Reports include all common numeric diagnostics, signed node/edge/component/cycle
count errors, absolute deltas and relative percentage changes. Empty percentage
cells mean a zero control denominator. Runtime is reported separately from
the four scientific objectives. With few trials, TPE is a small structured search,
not a guarantee of an optimum; asynchronous execution also affects proposal order.

The existing 50-epoch patience is retained. An early-stopped candidate that fails
to reach a later endpoint is explicitly excluded from that endpoint ranking, not
compared using a shorter tail. Infrastructure failures remain FAIL with reasons;
they are not silently replaced by new trials. Inspect failures before promotion.

## Local infrastructure tests

Use a Python environment with the project's dependencies and `requirements/optuna.txt`:

```bash
python -m unittest tests.test_focal_h0_study
```

These tests use tiny fake training to exercise checkpoint freezing, loss settings,
two concurrent workers, actual SIGTERM recovery, finite promotions, separate
fidelity reports, an isolated H1 ablation, and mocked Slurm dependency chains.
They do not validate real model learning or GPU throughput.

## Prepare on Jean Zay

Use the dedicated checkout requested for this experiment. If it was created as a
detached worktree, update it without switching the checkout used by other jobs:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching-h0-g05"
git fetch origin features-gnbm-nodes-edges
git switch --detach FETCH_HEAD
source cluster/jean_zay/env_a100.sh

export SYNTHETIC_MRI_DATASET="/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-edge-focal-g05-h0-study-a100"
export GNBM_H0_SOURCE_RUN="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g05_seed364505"
export GNBM_H0_REFERENCE_RUN="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-extension-750-a100/finetune_boundary_mri750_node_edge_focal_g05_tail250_seed364505"

bash cluster/jean_zay/submit_focal_h0_study.sh test
bash cluster/jean_zay/submit_focal_h0_study.sh prepare
```

The environment must already contain Optuna (`requirements/optuna.txt`). Preparation
is CPU-only and may take a little time reading/copying the full checkpoint once.
It does not repeat dataset scans for every queued segment. References for later
endpoints are frozen from the existing control when those endpoints are promoted.

## Required real GPU smoke

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh smoke
```

This requests one four-A100 job, dev QoS, two-hour limit. It uses only 32 train /
16 validation patches in disposable `gpu-smoke/v2` directories. Both the validation
loss loader and the independent metric loader are capped; scientific trials still
use the full validation split. The versioned directory preserves earlier smoke
outputs without reusing their uncapped validation history. Control, H0-only,
and small-H1 configurations each run three epochs as a 1+2 split to exercise real
DDP training and strict restart. BN calibration and evaluation settings remain
those of the reference. It also prints a fixed two-graph, single-GPU gradient probe
with weighted H0/H1 losses and parameter gradient norms (diagnostic, not a reliable
population estimate). Training time, sampling counts and effective weighted losses
are stored in each run's `performance.jsonl`; H0-only smoke verifies zero H1 graphs.
The tiny control exists only inside the smoke, not as another scientific control.

Check the returned job ID with `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode`.
Inspect `gpu-smoke/gradient-probe.json` and `gpu-smoke/passed.json`. There must be
no exception, finite losses/gradients, and `passed=true`. A zero topology gradient
on one batch is not proof of a defect, but should be inspected before relying on
that weighting. No scientific search can be submitted until the smoke passes.
Repeat `smoke` after a timeout; completed smoke phases are reused.

Study training uses single-node static rendezvous with an available loopback port
and unique rendezvous ID per launch. Workers reuse the agent's TCP store, avoiding
the separate rank-zero store bind used by standalone/c10d on PyTorch 2.3. The CPU
test suite exercises two real two-process Gloo launches as well as fake training;
the GPU smoke is still required to validate NCCL and actual H0/H1 backward passes.
The CPU-only probe pins Gloo to loopback and allows 180 seconds per launch,
including imports from shared storage. On failure it prints launcher/worker stage
logs and terminates its entire child process group. These probe settings do not
change the scientific training launcher or its NCCL networking.

## Launch and monitor H0 screening

```bash
export GNBM_WALLTIME="06:00:00"
export GNBM_QOS="qos_gpu_a100-t3"
export GNBM_H0_SEGMENTS=6
bash cluster/jean_zay/submit_focal_h0_study.sh screen 2
```

This submits six chained two-task arrays. Each task uses 4 GPUs, so up to 8 A100s
are active concurrently. Segments have six-hour limits and resume on `afterany`,
including after a timeout. The JournalStorage studies and kernel trial locks are
shared across workers; active trials cannot be recovered by another worker.
No `recover` operation or failure-to-new-trial conversion is needed.

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh status
bash cluster/jean_zay/submit_focal_h0_study.sh summary
sacct -u "$USER" --starttime 2026-10-20 --name=gnbm-focal-h0 \
  --format=JobID,State,Elapsed,ExitCode
```

Slurm logs: `$WORK/logs/graph-native-betti-matching/a100/gnbm-focal-h0-ARRAY_TASK.{out,err}`.
CSV reports: `$GNBM_OUTPUT_DIR/summaries/h0/epoch_0525.csv`, then 0550 and 0600.
Each has parameters, Pareto membership, control values, deltas, and diagnostics.
Latest validation epochs, checkpoints, resume counts and failures are printed by
`status`. Performance timing measures training epochs only, excluding validation.

If the entire queued chain ends with unfinished trials, submit another chain:

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh resume 2
```

Do not submit a second chain while an earlier one is pending unless you set
`GNBM_H0_AFTER_JOB_ID` to the last array ID of that chain. Resume never resets a
trial's optimizer/scheduler. Completed work is reused. Epoch-boundary checkpointing
can repeat a partially completed epoch after termination.

## Manual promotion

After reviewing the Pareto trade-offs and failures, replace the example IDs below
with the selected 4 and later 2 trial numbers. `promote` freezes the selection;
rerunning it is idempotent, changing it is refused.

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh promote 550 0,2,5,7
bash cluster/jean_zay/submit_focal_h0_study.sh extend 550 2
# Once complete, inspect the separate epoch-550 report and choose two:
bash cluster/jean_zay/submit_focal_h0_study.sh promote 600 2,7
bash cluster/jean_zay/submit_focal_h0_study.sh extend 600 2
```

Later-stage workers use trial locks and existing checkpoints. Repeat the same
`extend` command after that stage's chain ends if more time is required. Do not
update the live checkout while a chain still needs it.

## Clean manual stopping of screening trials

Cancel only the selected campaign's queued continuation jobs first, then its
active Slurm tasks. Wait until they leave the queue before finalizing the trials;
ordinary interruption deliberately leaves Optuna trials RUNNING for resumption.
For the manually stopped H0 trials 6 and 7:

```bash
GNBM_H0_FAMILY=h0 bash cluster/jean_zay/submit_focal_h0_study.sh prune 6,7 \
  "Manual stop: H0-only connectivity/cycle trade-off; proceeding to the H1 ablation"
bash cluster/jean_zay/submit_focal_h0_study.sh status
```

The command does **not** cancel Slurm jobs itself. It requires an explicit reason,
checks every target worker lock before changing any trial, and marks stopped
screening trials PRUNED. Each trial retains its checkpoint/history and receives
`manual-stop.json` plus matching Optuna user attributes, including the last
validation epoch and exclusion from endpoint comparisons. No checkpoint is
loaded or deleted, and no fake training-complete marker is created. Partial
metrics are not assigned epoch-525 objective values. Existing completed/failed
trials remain unchanged; a valid endpoint result saved just before cancellation
is recovered as COMPLETE rather than discarded.

Repeated finalization is safe. A durable manual-stop record blocks resumption
even if finalization is interrupted before Optuna records PRUNED. With all eight
H0 trials allocated, terminal pruning does not allocate replacement trials.
Reports explicitly count PRUNED trials: six completed plus two manually pruned is
not an eight-completed-trial experiment. The focal control and H1 journal are
untouched. Slurm accounting and existing W&B offline histories are not rewritten
or erased; their cancellation/failure exit status is distinct from Optuna PRUNED.

## H1 ablation from a completed epoch-525 H0 result

The selected H0-only comparator is **trial 3, H0 weight 0.03**. It only needs to
have completed epoch 525; no epoch-550 continuation is required. Prepare the
ablation without changing `campaign.json`, the H0 journal, or existing runs:

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh h1-prepare 3 0.001,0.003,0.01,0.03
cat "$GNBM_OUTPUT_DIR/h1-plan.json"
# After the existing H0 workers finish, submit the new grid:
bash cluster/jean_zay/submit_focal_h0_study.sh h1-screen 2
bash cluster/jean_zay/submit_focal_h0_study.sh h1-summary
GNBM_H0_FAMILY=h1 bash cluster/jean_zay/submit_focal_h0_study.sh status
```

This creates **four fixed-grid arms**, not another adaptive search: H0=0.03 with
H1=0.001, 0.003, 0.01, 0.03. The H1-zero comparison is the existing H0 trial 3,
not an additional training job. Every new arm restarts from the original frozen
epoch-500 model/optimizer, **not** from trial 3's epoch-525 model. The learning-rate
restart, 10-epoch ramp, focal recipe, global batch 32, patience and validation
protocol are unchanged. H0 uses all eight eligible local graphs; H1 samples one
graph/rank during training. Validation is unsampled. H1 FP/FN weights stay at
0.25/1.0; no loss-weight compensation for sampling is added.

The separate `h1-plan.json` freezes this grid and an epoch-matched snapshot of the
H0-only reference. Worker allocation and categorical choices use that plan, not
the older `h1_ablation_weights` field in the frozen H0 campaign settings. Repeated
preparation is idempotent, recovers partially enqueued grids, and refuses changed
weights or a different parent. Previously prepared legacy grids remain unchanged.

H1 files are isolated under `studies/h1`, `runs/h1`, `configs/h1`, `results/h1` and
`summaries/h1`. Epoch-525 reports use exactly 515/520/525 and include comparisons
against **both** the focal-only control and H0-only trial 3. CSV columns prefixed
`delta_` compare with focal-only; `delta_vs_h0_` compare with H0-only. Later-stage
reports never substitute the epoch-525 parent for a missing 550/600 reference.
Both references also have relative percentage-change columns.

Each worker requests four A100s. Two workers use up to eight A100s; the default
six chained six-hour segments strictly resume existing H1 arms after timeouts.
Repeat `h1-screen 2` only after its previous chain has ended. H1 jobs are named
`gnbm-focal-h1`, with logs `gnbm-focal-h1-ARRAY_TASK.{out,err}` in the usual A100
log directory. The successful existing GPU smoke already exercises H1 and strict
DDP resume; a new GPU smoke is not required for this fixed-grid setup change.

No H0-only or focal-control job is submitted by `h1-prepare` or `h1-screen`.
To promote H1 candidates later, use the same `promote`/`extend` commands with
`GNBM_H0_FAMILY=h1`, then unset that variable; no promotion is automatic.

W&B uses project `gnbm`, group `edge-focal-g05-h0-study`, one stable ID per trial
across segments and promotions. Each curve includes the original focal metrics
through epoch 500. Metrics rather than all minibatch losses are bootstrapped.
Sync completed offline transactions with the existing sync helper, explicitly
setting `GNBM_WANDB_SYNC_PROJECT=gnbm`; do not upload an actively written file.

## Two stronger H1 probes (no adaptive proposals yet)

After the four H1 arms, prepare **only** these two additional configurations:

| New family / trial | H0 weight | H1 weight |
|---|---:|---:|
| `h1-probes / 0000` | 0.03 | 0.06 |
| `h1-probes / 0001` | 0.03 | 0.10 |

Both restart the frozen original epoch-500 checkpoint, retaining the optimizer
and applying the existing initial LR-tail restart, then stop at epoch 525.
They do **not** continue from a trained H0/H1 candidate. FP/FN remain 0.25/1.0,
the ramp remains ten epochs, H0 uses eight graphs/rank, H1 one graph/rank, and
validation is full. The existing GPU smoke is reused; the topology implementation
has not changed. Local tests additionally exercise real finite H1 backward at
both stronger weights. This is a fixed two-arm probe, not adaptive TPE.

The separate `h1-probes-plan.json` and `studies/h1-probes` journal do not change
`campaign.json`, `h1-plan.json`, or either earlier journal. Runs, configurations,
results, summaries and stable W&B names also use `h1-probes`, e.g.
`runs/h1-probes/trial_0000` and `g05-h1-probes-000`. Project `gnbm`, the existing
group and the original metric bootstrap through epoch 500 remain unchanged.
Preparation validates the H0-only parent has weight 0.03; it is idempotent and
rejects any other probe grid. Workers repair an interrupted enqueue under the
allocation lock and never allocate more than these two arms.

From the dedicated `Graph-Native-Betti-Matching-h0-g05` checkout and with the
existing study's `GNBM_OUTPUT_DIR` exported:

```bash
bash cluster/jean_zay/submit_focal_h0_study.sh test
bash cluster/jean_zay/submit_focal_h0_study.sh h1-probes-prepare 3
cat "$GNBM_OUTPUT_DIR/h1-probes-plan.json"

# Two workers, four A100s each, three chained six-hour segments per worker.
# Unset a stale dependency so the probes do not wait on an unrelated old job.
unset GNBM_H0_AFTER_JOB_ID
GNBM_H0_SEGMENTS=3 GNBM_WALLTIME=06:00:00 \
  bash cluster/jean_zay/submit_focal_h0_study.sh h1-probes-screen 2

GNBM_H0_FAMILY=h1-probes \
  bash cluster/jean_zay/submit_focal_h0_study.sh status
bash cluster/jean_zay/submit_focal_h0_study.sh h1-probes-summary
```

Jobs are named `gnbm-focal-h1-probes`; logs use
`gnbm-focal-h1-probes-ARRAY_TASK.{out,err}` in the usual A100 log directory.
Chained segments strictly resume optimizer, scheduler, patience and runtime state.
If the chain expires before completion, repeat `h1-probes-screen 2` with the same
segment/time overrides, **only after the prior probe chain has ended**.
Completed trials are not retrained; extra queued segments do not create new
configurations. Reports compare the exact 515/520/525 tail with focal-only and
H0-only trial 3 and save deltas/percentages under `summaries/h1-probes`.

At screening, a generic `training-complete` marker may be absent because the
overall horizon is 750; use the endpoint result and Optuna COMPLETE state to
confirm epoch-525 completion. No stronger-weight promotion, adaptive study, new
focal control, or old-grid rerun is submitted by these commands.
