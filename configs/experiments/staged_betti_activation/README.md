# Staged Betti activation

This campaign is independent of `node_edge_betti_optuna` Study A.  It trains
one no-Betti MRI trajectory, snapshots its complete state at epochs 200, 300,
400, and 500, then screens branch-relative Betti ramps for 50 epochs from each
snapshot.  Cross-branch objectives are deltas from the matching no-Betti
trajectory at epochs 250, 350, and 450.

The fixed base recipe is node focal classification, edge cross-entropy,
Hungarian matching, and node-edge filtration alpha 0.5.  The search varies only
H0 weight, H1 weight, H1 false-positive weight, and ramp length.

`matched_mean` in the existing H1 loss averages matched-cycle terms but sums
unmatched false-cycle terms.  Therefore the H1 magnitude can increase with the
number of false cycles.  This campaign deliberately preserves that behavior;
normalizing it differently would define a separate loss variant.

The pre-campaign gradient audit also found that the former missed-target-cycle
term was constant and therefore could not create a missing cycle.  This branch
fixes that mechanics bug by applying the missed-cycle penalty at the target
class's deterministic critical predicted edge.  This is explicit and tested;
the normalization policy above is unchanged.

The trainer currently runs full precision and has no AMP scaler.  Checkpoints
record `scaler: null`, plus model, optimizer, scheduler, completed epoch,
global iteration, resolved training configuration, trainer state, and per-rank
Python/NumPy/PyTorch/CUDA/data-loader RNG state.

## Required real-model smoke and interruption check

Use a fresh smoke directory and the tiny 2/4/6-epoch protocol:

```bash
export GNBM_STAGED_CONFIG_OVERRIDE="configs/experiments/staged_betti_activation/smoke.yaml"
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-betti-staged-activation-smoke-a100"

bash cluster/jean_zay/submit_staged_betti_activation.sh prefix
bash cluster/jean_zay/submit_staged_betti_activation.sh prepare
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 2 2
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 4 2
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 6 2
bash cluster/jean_zay/submit_staged_betti_activation.sh final 2 0
bash cluster/jean_zay/submit_staged_betti_activation.sh summarize
```

For the resume check, cancel one screening array only after `status` shows a
trial checkpoint, confirm its Optuna state remains `RUNNING`, and run
`resume BRANCH 2`. The same trial number and parameters must finish with a
positive resume count. Unset `GNBM_STAGED_CONFIG_OVERRIDE` before any full
campaign command. `summarize` writes a passing `smoke-validation.json` only
after all six trials, at least one resumed trial, and one final continuation
are complete.

Use a fresh directory. Production submissions are dependency chains: by
default `prefix` submits 12 sequential 20-hour jobs, each screening submission
uses 4 sequential arrays, and each selected final continuation uses 8
sequential jobs. A successor starts with `afterany` only after its predecessor
has stopped, restores the latest full-state checkpoint, and exits quickly if
the stage is already complete. Override a chain length when needed with, for
example, `GNBM_STAGED_CHAIN_SEGMENTS=16`.

Every compute job forces W&B offline mode. All resumed segments retain the
same run ID and can be appended into one cloud run from a login node. The
shared-prefix control records the complete no-Betti curve. Each selected final
candidate additionally bootstraps its W&B metric history with the shared
node-focal trajectory through its activation epoch and its own screening
history, then logs its continuation through epoch 500. Consequently
`metrics/node_mAP`, `metrics/node_mAR`, `metrics/edge_mAP`, and
`metrics/edge_mAR` form a single activation-aware curve. The numerical
`metrics/betti_active` series marks the transition, and the effective H0/H1
weights are logged during Betti-supervised epochs.

Production execution defaults to two A100s with DDP, matching the earlier
node-focal fine-tuning campaign. The global training batch remains 32, split
as 16 samples per GPU. `GNBM_STAGED_GPUS` may be set to 1, 2, or 4 before a
new campaign, but a trajectory must retain the same world size when resumed.
Changing GPU count therefore requires a fresh output directory.

Use a fresh directory:

```bash
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-betti-staged-activation-a100"
export GNBM_STAGED_SMOKE_OUTPUT="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-betti-staged-activation-smoke-a100"
export GNBM_INITIAL_WEIGHTS="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/pretrain_boundary_mixed_node_focal_seed364505/models/best_metric_checkpoint.pt"

bash cluster/jean_zay/submit_staged_betti_activation.sh prefix
# After the prefix is complete:
bash cluster/jean_zay/submit_staged_betti_activation.sh prepare
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 200 4
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 300 4
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 400 4
bash cluster/jean_zay/submit_staged_betti_activation.sh summarize
```

The dependency chain automatically covers ordinary wall-time exits. If every
reserved segment is exhausted before completion, submit `prefix` again; it
resumes `shared-prefix/models/latest_checkpoint.pt` and never starts a second
trajectory in the marked output directory.

Once a branch checkpoint and its matched 50-epoch control endpoint exist, that
branch may be prepared and screened while the shared prefix continues.  The
branch-scoped command freezes only that control reference and does not require
the epoch-500 completion marker:

```bash
bash cluster/jean_zay/submit_staged_betti_activation.sh prepare 200
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 200 2
```

Branch 300 becomes eligible after validation epoch 350, and branch 400 after
validation epoch 450.  Calling `prepare` without a branch retains the stricter
final audit: it requires the completed prefix and validates every branch.
Each screening run bootstraps only the shared metric history through its own
activation epoch, so later prefix writes cannot change its W&B history.

Inspect `summaries/screening-pareto.csv`, duplicate-noise labels, and the
descriptive candidate suggestions. Then submit zero, one, or at most two
reviewed candidates per branch with `final BRANCH TRIAL`. Run `summarize`
again after those continuations finish.

Do not use the test split.  Inspect the three Pareto fronts before explicitly
starting any final continuation.

After jobs finish, upload all offline segments from a login node. The legacy
append path tolerates the truncated final transaction commonly left by a
wall-time termination:

```bash
GNBM_WANDB_SYNC_PROJECT=gnbm \
  bash cluster/jean_zay/sync_wandb_offline.sh "$GNBM_OUTPUT_DIR"
```

After selecting exactly one final model, the guarded command below performs
the campaign's only test evaluation. It records the selection before loading
test data and refuses to evaluate a different model afterward:

```bash
bash cluster/jean_zay/submit_staged_betti_activation.sh test 300 TRIAL_NUMBER
```

## Subsampled epoch-300 search

`subsampled_epoch300.yaml` is the feasible follow-up to the full-batch timing
diagnosis. During screening it computes H0 on all 16 local graphs and H1 on
two uniformly sampled graphs per rank. It averages only the selected H1
losses; it does not divide by the full local batch. Validation remains
full-batch. The search uses the frozen epoch-300 prefix checkpoint, eight
trials, and a 15-epoch endpoint.

It can reuse an existing staged campaign whose epoch-300 study has not yet
been created:

```bash
export GNBM_STAGED_CONFIG_OVERRIDE="configs/experiments/staged_betti_activation/subsampled_epoch300.yaml"
bash cluster/jean_zay/submit_staged_betti_activation.sh prepare 300
bash cluster/jean_zay/submit_staged_betti_activation.sh screen 300 2
```

Keep the existing `GNBM_OUTPUT_DIR`, `GNBM_INITIAL_WEIGHTS`, GPU count, and
global batch variables. Do not run this concurrently with old branch-200
screening jobs. Inspect `topology_scored_graphs` and the per-loss eligible,
selected, and sampling-fraction training metrics before extending the search.

Before launching the search, profile the epoch-300 checkpoint with:

```bash
bash cluster/jean_zay/submit_betti_sampling_profile.sh
```

The one-A100 diagnostic uses identical train streams for baseline, H0-only,
H1-only, full H0+H1, and H0+H1 with two or four graphs per rank. It writes
`summary.json` and `timings.csv` under
`$GNBM_OUTPUT_DIR/profiles/betti-sampling-epoch300`. The JSON also compares
sampled topology gradients against the full-batch topology gradient. This is
a diagnostic only: it does not create or mutate an Optuna study.
