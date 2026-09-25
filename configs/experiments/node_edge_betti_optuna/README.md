# Node/edge Betti Pareto studies

This framework performs validation-only, four-objective selection around the
existing `train.py`. It does not alter the Betti mathematics or duplicate the
training loop. No test sample is loaded during search, Pareto construction,
candidate selection, continuation, or paired validation.

## Why Pareto optimization

The earlier scalar score used arbitrary graph-metric degradation thresholds and
penalties. Those have been removed: a sufficiently large topology improvement may
justify some predictive degradation, and that scientific trade-off should remain
visible. Every trial now has exactly four primary objectives, in fixed order:

1. maximize node mAP;
2. maximize edge mAP;
3. minimize beta0 absolute error;
4. minimize beta1 absolute error.

F1, precision, recall, SMD, graph sizes, and predicted Betti numbers remain in the
reports as diagnostics. Making all of them objectives would produce a weakly
selective high-dimensional front. There is no unique mathematically best Pareto
configuration. Reports contain the complete front, four single-objective anchors,
and an explicitly conventional equal-weight ideal-distance representative. A
supervisor may choose a different Pareto point.

Control and trials use the same fixed `tail_mean` aggregation. Study A averages
the final three validation observations; the smoke averages the final two. No
metric and no trial gets a separately selected best epoch. The control is a
reference point for deltas and paired-integrity checks, not a pass/fail constraint.
Negative control deltas favor trials for beta errors; positive deltas favor trials
for mAP/F1.

NSGA-II is deterministic with seed 364505 and population 12. Native scalar
`trial.report` pruning was removed because a hidden scalar proxy would bias the
Pareto trade-off. Only invalid configurations, non-finite/missing outputs, failed
processes, and interruption end a trial early.

FGW is excluded. It changes assignment and would confound attribution between
matching, edge loss, and topology supervision; previous three-seed evidence did
not show a consistent advantage. `topology.complex.alpha` in Study B is the hybrid
node-edge **filtration coefficient**, not FGW `structure_weight`. The matcher is
asserted to remain Hungarian.

## Campaign stages

- **Smoke:** four tiny trials on 256/128 patches for five epochs. Infrastructure
  only; no scientific interpretation.
- **Study A:** 48 Betti-only trials, 4,000/500 deterministic patches, 100 epochs,
  validation every five epochs, `tail_mean_3`, edge cross-entropy, node focal
  gamma 2, Hungarian matching.
- **Study B:** a controlled refinement over edge CE/focal choice, focal gamma,
  filtration alpha, and Betti values extracted from selected Study A Pareto
  candidates. It is not “optimize everything,” and it must not include FGW.
- **Multi-fidelity continuation:** at most eight diverse Pareto candidates,
  suggested 20,000 training patches, full validation if affordable, 200 epochs,
  fixed `tail_mean_3`; no test data.
- **Paired confirmation:** predictive, balanced, and topology representatives,
  paired with matching no-Betti controls for seeds 364505–364507. A focal
  candidate must be compared with a no-Betti control using that same focal loss.
- **Final full-data experiment:** one validation-selected protocol, complete
  corrected train/validation splits, paired no-Betti/Betti arms, preferably three
  seeds, 500 epochs, and the same fixed epoch-500 checkpoint rule. Only after the
  protocol is frozen is the untouched test split evaluated once.

## Storage and resumption

SQLite is for one local/controller process and is protected by
`.controller.lock`. Never run concurrent SQLite workers. JournalStorage uses
Optuna's compatible file backend for at most four bounded Jean Zay workers. Trial
allocation is serialized briefly so the global requested count is not exceeded;
each globally allocated trial number owns `runs/trial_NNNN/` and is never
overwritten. Journal file storage is appropriate for this bounded shared-filesystem
mode, not unlimited concurrency.

SIGTERM/SIGINT terminates the child and marks the trial failed, never complete.
Hard-killed workers can leave RUNNING records. Only when **all workers are stopped**,
run the explicit `recover` command; it marks stale records failed and retains all
artifacts. Resubmitting workers continues the named study. Completed trials are
immutable. Extending Study A to 60 or 72 means changing `n_trials` in the same
config and reusing the same output/study name; inspect front expansion first and
never extend automatically.

## Dependency and local commands

```bash
python -m pip install -r requirements/optuna.txt
python -m unittest tests.test_node_edge_betti_optuna
python -m unittest tests.test_betti_node_edge_config \
  tests.test_betti_node_edge_filtration tests.test_graph_losses \
  tests.test_experiment_configs
```

The summary writes `summary.json`, `trials.csv`, `pareto-front.csv`,
`pareto-front.json`, the frozen `control-reference.json`, and representative JSON
and Markdown. CSV columns support mAP-versus-beta plots, predictive-versus-topology
comparisons, and parallel coordinates.

Generate a Study B proposal after reviewing Study A:

```bash
python scripts/propose_node_edge_betti_study_b.py \
  --study-a-summary "$STUDY_A_OUTPUT/summary.json"
```

By default it uses the unique representative trials. Pass `--trials ...` to use
other inspected Pareto candidates. Inspect `study_b_proposed.yaml`; it is never
launched automatically.

## Jean Zay smoke and monitoring

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching"
git switch features-gnbm-nodes-edges
git pull --ff-only origin features-gnbm-nodes-edges
source cluster/jean_zay/env_a100.sh
python -m pip install -r requirements/optuna.txt

export SYNTHETIC_MRI_DATASET="/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary"
export GNBM_INITIAL_WEIGHTS="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/pretrain_boundary_mixed_node_focal_seed364505/models/best_metric_checkpoint.pt"
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-node-edge-betti-pareto-smoke-a100"

bash cluster/jean_zay/submit_node_edge_betti_optuna.sh control smoke
# After the control succeeds:
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh worker smoke
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh summarize smoke
```

Resume the single worker with `resume smoke`; four existing
trials are not repeated. Optional two-worker storage smoke uses a fresh output:

```bash
export GNBM_OUTPUT_DIR="${GNBM_OUTPUT_DIR%-a100}-journal-a100"
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh control smoke
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh array smoke 2
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh summarize smoke 2
```

Resume that JournalStorage study with `resume smoke 2`. Run `recover smoke 2`
only after confirming no array worker remains active.

Monitor jobs (replace `JOBID`):

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -f "$WORK/logs/graph-native-betti-matching/a100/gnbm-betti-optuna-JOBID_0.out"
tail -f "$WORK/logs/graph-native-betti-matching/a100/gnbm-betti-optuna-JOBID_0.err"
find "$GNBM_OUTPUT_DIR" -maxdepth 3 -type f | sort
```

## Study A launch gate

Do not run these until local tests, single-worker smoke, resume, and Pareto-summary
inspection pass. The optional two-worker storage smoke should precede parallel use.

```bash
export GNBM_OUTPUT_DIR="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-node-edge-betti-pareto-study-a-a100"
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh control study-a
# Only after the control succeeds and after explicit user confirmation:
bash cluster/jean_zay/submit_node_edge_betti_optuna.sh array study-a 4
```

The preflight prints checkpoint, dataset, caps, epochs, objectives, sampler, trial
and worker counts, estimated GPU jobs, test exclusion, Hungarian matcher, and edge
CE. Forty-eight 100-epoch trials plus one control equal 4,900 training epochs over
4,000 patches. Four workers reduce wall-clock latency but not total GPU-hours; jobs
may require repeated 20-hour resumptions, and actual cost must be estimated from
the smoke/pilot throughput rather than claimed in advance.

## Preparing the final pair

After paired validation—not merely a Pareto front—materialize three-seed full-data
configs without launching them:

```bash
python scripts/prepare_node_edge_betti_final_pair.py \
  --selected-config /path/to/frozen-selected-resolved-config.yaml \
  --output configs/experiments/node_edge_betti_final_pair
```

Inspect `experiment-plan.json`. Both arms use complete train/validation data and
the real `models/checkpoint_epoch=500.pt`. Use segmented `submit_train_a100.sh`
jobs with `GNBM_AUTO_RESUME=1` and offline W&B. Test evaluation remains prohibited
until the full-data protocol and checkpoint rule are frozen.
