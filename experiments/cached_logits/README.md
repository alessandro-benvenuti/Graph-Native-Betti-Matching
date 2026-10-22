# Disposable cached-logit Betti diagnostic

This experiment optimizes patch-specific logits on training patches from a single
frozen epoch-500 checkpoint. It is supervised optimization, not model training,
an inference-time correction method, or evidence of generalization. It does not
run Optuna or change production loss code, campaign configurations, journals or jobs.

Base verified locally and on origin: `665e2f0de67a7dd5b9b72c608185a49788325984`.
Branch: `codex/cached-logit-betti-test`. Independent local clone:
`/Users/alessandrobenvenuti/Desktop/Università/Thesis/Graph-Native-Betti-Matching-cached-logit-test`.

## Local checks

From the disposable clone:

```bash
MPLCONFIGDIR=/tmp/gnbm-cached-logit-mpl python3 -m unittest \
  tests.test_cached_logit_betti tests.test_cached_logit_cluster_env tests.test_betti_h0 tests.test_betti_h1 \
  tests.test_betti_node_edge_filtration tests.test_graph_losses \
  tests.test_compare_loop_patch_predictions -v
```

The experiment tests cover probability/orientation fidelity, production helper
values and gradients, explicit zero births in A, clamping and detachment in B,
classification-only recovery of five graph examples, a false-cycle critical edge
that is GT-correct, duplicate candidates, empty/isolated graphs, identical arm
starts, frozen source data, missing support, zero denominators, output protection,
step-zero reproduction, runtime interruption and disk save/resume (absolute logit
tolerance `1e-7`, zero relative tolerance on CPU).

A complete synthetic CLI example (use fresh directories):

```bash
python3 -m experiments.cached_logits synthetic-cache --output artifacts/example-cache
MPLCONFIGDIR=/tmp/gnbm-cached-logit-mpl python3 -m experiments.cached_logits optimize \
  --cache artifacts/example-cache/predictions.pt --output artifacts/example-opt \
  --steps 200 --threads 1 --max-seconds 300
python3 -m experiments.cached_logits verify-smoke \
  --cache artifacts/example-cache/predictions.pt --run artifacts/example-opt
```

Synthetic verification writes `SYNTHETIC_VERIFIED.json` and cannot open the
cluster main-run gate.

## Copy-paste on Jean Zay: independent clone and smoke

Run this on the login node. The bootstrap refuses a pre-existing directory;
subsequent updates use the guarded preparation script shown below. It never
switches an existing campaign checkout.

```bash
set -e
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
export CACHED_LOGIT_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
if [ -e "$GNBM_REPO_DIR" ]; then
  echo "Directory already exists; inspect it before using the guarded update command below."
else
  mkdir -p "$WORK/projects"
  git clone --single-branch --branch codex/cached-logit-betti-test \
    https://github.com/alessandro-benvenuti/Graph-Native-Betti-Matching.git "$GNBM_REPO_DIR"
  printf '%s\n' codex/cached-logit-betti-test > "$GNBM_REPO_DIR/.git/cached-logit-disposable"
fi
cd "$GNBM_REPO_DIR"
bash experiments/cached_logits/cluster/prepare_checkout.sh
mkdir -p "$CACHED_LOGIT_ROOT/logs"
bash experiments/cached_logits/cluster/submit.sh smoke
```

`prepare_checkout.sh` checks the origin URL, branch, clean working tree,
independent `.git` directory and disposable marker before fetching and fast-forwarding.
It fails on unrelated work. Do not create that marker in a pre-existing unrelated clone.
For later pulls, run only:

```bash
bash "$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test/experiments/cached_logits/cluster/prepare_checkout.sh"
```

Smoke requests **one A100, one process, eight CPUs, one hour**, no DDP. It sets
`GNBM_REPO_DIR` before sourcing `env_a100.sh`, verifies `$PWD`, checks imports/CUDA,
verifies the login-node code snapshot without Git, selects exactly two training
IDs, caches them, closes the network stage, runs a
20-step CPU optimization, and verifies all arm starts against cache step-zero
metrics. No validation/test samples or training epochs run. A timeout/failure
cannot write the final success marker. Each output directory must be new.

The source defaults are exactly:

- Checkpoint: `/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g05_seed364505/models/latest_checkpoint.pt`
- Config: that run's `resolved-config.yaml`.
- Dataset: `/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary`, strict `train` split.

Caching rejects an epoch other than 500, incompatible model state, nonfinite
model/input/output tensors, unexpected focal gammas (node 2, edge .5), unmatched
classification or ratio balancing. It records checkpoint SHA256 (checked again
at cache completion), original/resolved configs and hashes, exact clean code
commit, Hungarian assignments, both directed logits, fixed support, label masks,
selection provenance and unchanged model buffers. No images/tokens are needed
after caching. All artifacts under `/lustre` must remain inside the new root.

## After smoke completion: inspect, benchmark, then main

Do not submit main while the smoke job is pending or failed. After its successful
completion, run the following on the login node. This reads small artifacts;
the network never runs on the login node.

```bash
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
export CACHED_LOGIT_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
cd "$GNBM_REPO_DIR"
module purge
module load pytorch-gpu/py3/2.3.0
source experiments/cached_logits/cluster/git_env.sh
cached_logit_require_git
source "${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/activate"
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cat "$CACHED_LOGIT_ROOT/smoke-opt/report.md"
python -m experiments.cached_logits next-commands --root "$CACHED_LOGIT_ROOT"
```

`next-commands` only provides the next submission command once the preceding
stage succeeds. It first checks a real two-patch smoke marker from the same code
commit, then provides the main cache command, then a CPU benchmark command, and
finally the bounded optimization command if its runtime estimate fits. Nothing
is submitted automatically. Run it again after each stage completes.

The main cache uses a **bounded deterministic pool of 64 training patches**,
then freezes up to **32 distinct IDs** with the existing category selector:
broken H0, false loops, genuine loops with missing edges, false-positive nodes,
and correct controls. It reports availability, overlapping category counts,
quotas, fallback choices and shortfalls rather than scanning further. Discovery
lists split filenames using existing helpers; only manifest patches undergo
image loading and network inference. Existing pools can instead be supplied to
`select --pool PATH --count 32`: they must contain `rows` compatible with
`select_topology_overfit_patches` and explicit
`provenance: {split: train, dataset_root: EXACT_ROOT}`. IDs are checked against
the training split. Selection categories use fixed-assignment errors on initial
B support; they do not claim spatial detection correctness.

Optimization defaults: **CPU, one bounded numerical thread**, Adam, learning
rate .03, 200 updates, fresh optimizer for each of the six mode/arm combinations.
CPU jobs request **four CPUs, zero GPUs, one hour**. The three-graph benchmark
samples smallest/median/largest cached supports for five updates, at most 300
seconds. It estimates `max(seconds/update) × 32 × 6 × 200 × 1.5`; main is gated
at 3,300 seconds, leaving scheduler headroom. Main saves/resumes and stops cleanly
at that runtime bound. Actual real-patch runtime is pending the cluster smoke.
A local two-patch **synthetic** 200-update CLI run took about 4.3 seconds; this
is not a runtime estimate for large cached graphs. The tiny 20-update CPU
benchmark measured .031/.032 seconds for 3/4 vertices. Large cycle spaces can
be much more expensive: matching itself uses detached CPU/Python GF2 work, so
a GPU is not assumed to resolve a slow CPU benchmark. If that gate fails, inspect
the measured bottleneck before requesting a different bounded resource plan;
no samples, steps, GPUs or walltime are increased automatically.

For direct CLI usage, `select`, `cache`, `benchmark`, `optimize`, `verify-smoke`,
`next-commands`, and `report` all expose `--help`. There is no study or sweep.

## Modes, objectives and evaluation

A freezes all node logits/coordinates and optimizes both directions of M–M
relations. Its vertices are all Hungarian-matched nodes, even low-confidence
ones, with explicit births at zero. Its GT is the corresponding induced graph.
This deliberate zero-birth adapter passes all-one node probabilities to the
production H0 helper. Excluded GT vertices and cycle rank are reported.

B freezes coordinates, assignments and initially selected topology queries.
It optimizes all node logits and cached relations, retaining production
matched-node confidence clamping, aggregation, normalization and the source
config's detachment of unmatched-incident raw edge probabilities. Support never
changes. All-query node classification metrics are reported separately from
node/edge graph metrics on selected support. Active predicted queries outside
support are counted; no unavailable vertices or edges are injected. A and B
have different supports and cannot be compared as though only one term changed.

Both modes use identical initial tensors for these paired arms:

|Arm|H0|H1|
|---|---:|---:|
|classification|0|0|
|h0|.03|0|
|h0_h1|.03|.01|

H1 FP/FN weights are .25/1. All other classification coefficients, class weights,
normalization, aggregation and detach rules come from the actual resolved config.
Topology is evaluated irrespective of the source enable flags. Weights are
constant: there are no training epochs/warmup/campaign ramps. Classification
curricula are evaluated at the completed source horizon. Frozen coordinate
losses and nondifferentiable argmax cardinality are omitted.

Topology uses the mean of directed positive probabilities. Edge classification
uses the equal-weight mean of two directed losses: the expected random
orientation objective, differing from a particular sampled training orientation.
Classification supervises M–M candidates only; capped negative selection is
seeded and frozen. Unmatched-incident labels are `-1` and always masked. Graph
thresholding preserves existing inference's **softmax of mean directed logits**,
with source thresholds (`null` means argmax). Thus topology and hard graph
scoring deliberately follow their respective production conventions.

Observations use steps 0,1,5,10,20,50,100,200 and the actual endpoint. No arm gets
its own best-step selection. Primary graph PR/F1 and Betti errors use frozen
assignment and are patch means, not detection edge mAP. Full-GT and induced
support errors, exact GF2 shared/false/missed cycle ranks, cycle gain/loss
turnover, category/loop/support cohorts, paired wins/ties/losses, and spatial
cycle audits are reported separately. Spatial audits reuse
`compare_loop_patch_predictions.spatial_cycle_result` at steps 0/endpoint with
an explicit default distance .1 (recorded in config; overridable while caching).
Rank equality alone never establishes genuine loop recovery.

At steps 0,20,200, separate classification/H0/H1 gradients are measured with
respect to optimized logits. A negative gradient along positive-vs-negative
logit contrast promotes the edge; positive suppresses it. Artifacts include
both directed and unique-pair counts, absolute logit-gradient mass, M–M versus
unmatched-incident attribution, node attribution, conflicts and critical-edge
GT membership. These are local diagnostics, not explanations of every final
change. All correctness statements are relative to GT annotation quality.

## Resumption and reporting

For an interrupted **stopped** main job, use
`bash experiments/cached_logits/cluster/submit.sh resume` on the login node. States include Adam moments, update number,
logits and past observations. Resume rejects changed cache, configuration,
optimizer settings, threads or code. The per-invocation runtime bound can change.
If a scheduler kill leaves `main-opt/RUNNING.lock`, first verify that its job is
no longer running; only then remove that lock. Never remove a live job's lock.
Completed runs cannot resume or overwrite. Cache files are hashed and never
modified by optimization. A failed stage has no success marker for that stage;
a completed cache remains valid if later optimization fails.

To regenerate reports in a **new** directory after a completed main run:

```bash
python -m experiments.cached_logits report \
  --cache "$CACHED_LOGIT_ROOT/main-cache/predictions.pt" \
  --run "$CACHED_LOGIT_ROOT/main-opt" --output "$CACHED_LOGIT_ROOT/main-report-copy"
```

Artifacts in `report.md`, `summary.json`, `per-patch.{json,csv}`,
`trajectories.{json,csv}`, `gradients.json`, `gradient-edges.csv` and two PNGs
answer whether Betti improves correctness beyond classification in each mode.
A result remains pending until real cached graphs have completed. Successful
patch-wise optimization cannot demonstrate benefit for network training or
by itself locate a training limitation.

## Created directories and retention

Tracked experiment code is only in `experiments/` and
`tests/test_cached_logit_betti.py` and `tests/test_cached_logit_cluster_env.py`. Local generated artifacts are under the
ignored `artifacts/cached-logit-local-smoke/`: cache, run, report-v2, benchmark,
verification JSON and all per-arm states. A clean-commit replay is retained in `final-cache/` and `final-run/`. See `LOCAL_VALIDATION.md` for the
local artifact inventory. Temporary tests use auto-cleaned system temp folders.

Expected Jean Zay directories, created only by the commands above:

- `$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test` (independent Git clone).
- New checkpoint root `gnbm-cached-logit-betti-test-a100/` with `logs/`,
  `matplotlib/`, `smoke-cache/`, `smoke-opt/`, `main-cache/`,
  `cpu-benchmark/`, `main-opt/`, and optional `main-report-copy/`.
- Root manifest files: `smoke-manifest.json`, `main-pool-manifest.json`.
- Immutable submission snapshots in the disposable clone's `.git/cached-logit-provenance/`.
- Existing environment setup may ensure shared tool/extension/temp caches exist;
  it does not alter campaign outputs. W&B is disabled.

Each cache has `predictions.pt`, `selection.json`, `step-zero.json`,
`provenance.json`, `resolved-config.yaml`, `CACHE_COMPLETE.json`. Each run has
`run.json`, twelve-to-192 `<sample>-<mode>-<arm>.pt` state files (six per patch),
reports/plots listed above, and `COMPLETE.json` only on successful completion.
Incomplete runs have `INCOMPLETE.json`; a running process owns `RUNNING.lock`.
Real smoke adds `SMOKE_VERIFIED.json`. The benchmark writes `benchmark.json`.

No automatic merge or deletion. Retain caches/results/provenance before removing
the disposable clone or branch. Never merge into `features-gnbm-nodes-edges` as
part of this diagnostic.

## Restart after a startup failure

Both reported jobs stopped before manifest/cache creation. Update the disposable
clone and submit through the login-node helper:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
bash experiments/cached_logits/cluster/prepare_checkout.sh
bash experiments/cached_logits/cluster/submit.sh smoke
```

Use `submit.sh` rather than calling `sbatch` directly. The login-node helper
checks the exact clean branch, independent clone, origin and commit with Git.
It captures SHA256 for all tracked files in an immutable JSON snapshot under
`.git/cached-logit-provenance/` and exports its path to the job. It uses only
Python's standard library before submission, without loading the model or data.
The batch job verifies every recorded file and rejects added Python sources;
its Python provenance calls use this snapshot rather than Git. No Git commands
or Git modules are used on compute nodes. Snapshot verification also runs before
CUDA/model imports and prints the verified commit. Code changed while queued
causes a clear failure; snapshots and results are never silently overwritten.

Git is still needed on the login node for `prepare_checkout.sh`, `submit.sh`,
and unsnapshotted local CLI commands. If absent there, the login-only helper
can load `${CACHED_LOGIT_GIT_MODULE:-git}`. Login-side Git failures now include
the exit code and original stderr. Production environment scripts and resource
requests are unchanged. Preserve previous logs; no output cleanup is needed
for the two failures reported so far.
