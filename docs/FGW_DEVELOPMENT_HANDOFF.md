# FGW matcher development handoff

Last updated: 2026-09-07

## 1. Objective and current decision

This project predicts 3D vascular graphs with RelationFormer. The model emits a
fixed set of node queries, their coordinates and object logits, plus relation
features used to score candidate edges.

The baseline uses Hungarian matching between predicted queries and ground-truth
nodes. Its cost uses node coordinates and node/object classification, but it
does not consider whether the matched queries reproduce the relations of the
ground-truth graph.

The current research question is whether Fused Gromov-Wasserstein (FGW) can
provide more stable, graph-aware node matches. The agreed design is:

```text
predicted coordinates and object probabilities ----> feature cost M
predicted relation probabilities ------------------> predicted structure Cpred
GT adjacency --------------------------------------> target structure Cgt

M, Cpred, Cgt --> partial FGW --> transport T --> hard one-to-one assignment
                                                        |
                                                        v
                                             existing node/edge/Betti losses
```

FGW is currently a matcher, not an additional differentiable loss. The
transport and hardening run without gradients, just as Hungarian matching does.
Gradients still flow through the selected node and edge predictions when the
existing losses are evaluated.

Hard matches were retained deliberately. Soft supervision could allow several
queries to receive positive supervision associated with the same target and
could fail to suppress duplicate predictions. The intended training target is
one predicted query per ground-truth node.

## 2. FGW quantities used in this implementation

For a graph with $n$ ground-truth nodes and a pool of candidate predicted
queries:

- $M$ is the target-by-candidate feature-cost matrix. It uses the same weighted
  coordinate L1 and object-classification terms as the Hungarian matcher. The
  classification term is `-p(object)`, not `-log(p(object))`.
- $C_{GT}$ is the symmetric, loop-free binary ground-truth adjacency matrix.
- $C_{pred}$ is the symmetric matrix of predicted edge probabilities. Both
  query orders are evaluated and averaged.
- $\alpha$ is `structure_weight`. POT optimizes a feature term weighted by
  $1-\alpha$ and a Gromov-Wasserstein structural term weighted by $\alpha$.
- The candidate pool always contains every unary Hungarian match, then adds the
  highest-confidence remaining queries up to `candidate_count` (normally 32).
- The final hard assignment is a global `linear_sum_assignment(-T)` projection,
  never independent row-wise argmax.

The structures are similarities used directly: binary GT adjacency and
predicted edge probability, both symmetric with zero diagonal. The solver is
POT's non-entropic partial FGW conditional-gradient routine; its linearized
subproblems use network-simplex EMD. Uniform masses and capacities are
deliberate and are not confidence weighted.

Partial capacities do not guarantee that a feasible transport is integral.
Row-argmax collisions can occur in a valid fractional plan and are therefore a
concentration diagnostic, not a feasibility failure. Global hardening does
guarantee one distinct query per target, but may change (and increase) the FGW
objective.

Complementing two complete structural matrices preserves every squared
difference because `(1-x)-(1-y)=y-x`. Complementing only off-diagonal entries
and then resetting diagonals is not generally equivalent for fractional
transport: diagonal/off-diagonal terms can couple through mass split across
different nodes. The implementation therefore documents and evaluates its
actual direct adjacency/probability matrices.

The predicted node probabilities are part of $M$, and edge probabilities are
part of $C_{pred}$. They are not currently used as nonuniform OT marginals.

## 3. Why semi-relaxed FGW was rejected

The first implementation used POT's
`semirelaxed_fused_gromov_wasserstein`. Ground-truth nodes were fixed-mass rows,
but prediction-column masses were unrestricted:

$$T\mathbf 1=p.$$

This permits multiple GT rows to select the same prediction. Hardening such a
sparse plan can force some targets onto entries with zero transport mass, so
the projection may be arbitrary and may destroy the soft FGW solution.

This was demonstrated by a five-sample validation diagnostic using the saved
Node Focal + Edge Focal matched-only checkpoint. For
$\alpha\in\{0.1,0.2,0.4\}$, every alpha produced the same result:

| Metric | Hungarian | Semi-relaxed FGW | FGW minus Hungarian |
|---|---:|---:|---:|
| Coordinate L1 | 0.021280 | 0.072051 | +0.050771 |
| Structural MSE | 0.017641 | 0.035040 | +0.017399 |
| Edge/non-edge separation | 0.732599 | 0.631633 | -0.100966 |
| Changed target fraction | 0 | 0.149077 | +0.149077 |

FGW changed at least one match in four of five graphs. Its row entropy was zero,
meaning the plans were sharp, but there were about 2.2 row-argmax collisions per
graph. The issue was concentrated many-to-one transport, not diffuse rows.

A second sanity sweep used
$\alpha\in\{0,0.01,0.05,0.1,0.4,0.8\}$. Critically, even $\alpha=0$ did not
reproduce Hungarian: it retained a changed-target fraction of 0.149077 and
about 2.4 collisions per graph. This is expected from the semi-relaxed
constraints. With $\alpha=0$, every target independently selects its cheapest
query; there is still no column capacity enforcing an injection.

At $\alpha=0.8$, the five-sample aggregate happened to be close to and slightly
more structural than Hungarian (coordinate L1 0.021721, structural MSE
0.017433, separation 0.733982), but the plan still had about 1.4 collisions per
graph. This small result was not considered reliable because the hard
projection was still resolving an unsuitable relaxation.

## 4. Why partial FGW is being tested

POT's `partial_fused_gromov_wasserstein` supports row and column capacity
constraints while allowing unused prediction mass. The implementation uses:

$$p_i=\frac{1}{n},\qquad q_j=\frac{1}{n},\qquad
s=\sum_i p_i\approx1.$$

The intended constraints are:

$$T\mathbf 1\le p,\qquad T^\top\mathbf 1\le q,\qquad
\sum_{ij}T_{ij}=s.$$

Because the transported mass equals all available GT mass, every GT row must be
saturated. Because every prediction column has capacity $1/n$, a prediction
cannot consume the mass of two complete GT nodes. Surplus candidates can have
zero transported mass. This represents an injection from GT nodes into a
larger candidate pool.

At $\alpha=0$, partial FGW must attain the same optimal unary cost as Hungarian,
provided the candidate pool contains the Hungarian queries. The mapping itself
must agree when the optimum is unique; tied optima may legitimately return a
different mapping. Diagnostics classify these cases separately.

## 5. Implementation locations

- `models/matcher.py`
  - `HungarianMatcher`
  - `FusedGromovWassersteinMatcher`
  - candidate selection
  - partial-FGW solve and transport validation
  - global hard projection
  - shared candidate-relation scoring
- `training/losses/criterion.py`
  - constructs predicted structures when the configured matcher requires them
  - consumes the same hard assignment interface for all existing losses
- `data/loaders/mixed.py`
  - permits deterministic, augmentation-free `train`, `val`, and `test`
    evaluation loaders for offline diagnostics
- `scripts/diagnose_fgw_matching.py`
  - compares Hungarian with several hardened-FGW alpha values on one frozen
    checkpoint and one model forward per batch
- `configs/matchers/fgw.yaml`
  - reusable FGW matcher overlay
- `configs/finetune_synthetic_mri_focal_fgw.yaml`
  - ready FGW training configuration
- `configs/smoke_synthetic_mri_focal_fgw.yaml`
  - bounded synthetic smoke configuration
- `requirements/jean-zay.txt`
  - contains `POT==0.9.7`
- `tests/test_matcher.py`
  - matcher contracts, capacity checks, real POT solve, and alpha-zero invariant
- `tests/test_fgw_diagnostic.py`
  - diagnostic metric and aggregation contracts

## 6. Current partial-FGW failures and fixes

### 6.1 Network-simplex iteration limit

The first partial-FGW run failed with:

```text
UserWarning: numItermax reached before optimality
ValueError: Error in the EMD resolution: try to increase the number of dummy points
```

The dummy-point message was misleading in this case. POT 0.9.7 shares
`numItermax` between the outer conditional-gradient loop and every internal
network-simplex EMD subproblem. The previous value of 100 was too small. The
matcher configuration, class default, and diagnostic default were changed to
10000, which is POT's default.

### 6.2 Floating-point transported mass

The next run failed before optimization with:

```text
ValueError: Problem infeasible. Parameter m should lower or equal than
min(|a|_1, |b|_1).
```

The code constructed `p` with `np.full(n, 1.0 / n)` but passed the literal
`m=1.0`. For counts such as $n=15$, floating-point summation can produce:

```text
sum(p) = 0.9999999999999999
m      = 1.0
```

POT therefore correctly rejected `m > sum(p)` by one unit in the last place.
The local fix now passes:

```python
transported_mass = float(target_mass.sum())
```

A regression test solves a 15-target/32-candidate partial problem and checks
row masses, prediction-column capacities, and represented total mass. The
expanded relevant local suite now passes 77 tests.

The original uncommitted state contained this fix in:

```text
models/matcher.py
tests/test_matcher.py
```

Those user changes were preserved during the later correctness pass. Any future
commit/push and development-checkout update remains an explicit user action.

## 7. Git and checkout separation

There are two independent Jean Zay checkouts because scheduled campaign jobs
load Python and environment files from their checkout when they actually start.
A pending job would therefore see later edits made in the same directory.

```text
$WORK/projects/Graph-Native-Betti-Matching
    frozen campaign checkout used by running and pending jobs

$WORK/projects/Graph-Native-Betti-Matching-next
    FGW development checkout on branch fgw-development
```

Do not edit, pull, merge, switch branches, or check out files in the frozen
campaign checkout until its campaign has finished. Running jobs have generally
already imported their modules, but pending jobs have not.

The desktop repository is also on `fgw-development`. Its remote is:

```text
https://github.com/alessandro-benvenuti/Graph-Native-Betti-Matching.git
```

The relevant recent history before the floating-mass fix is:

```text
ffca488 fixed fgw solver iteration limit bug
6af5881 partial fgw implementation
846f29c fgw test
61ad749 added fgw smoke config
ddde9aa timeout fix (main at the start of FGW development)
```

Normal desktop workflow:

```bash
git branch --show-current             # must print fgw-development
git status --short
git add models/matcher.py tests/test_matcher.py
git commit -m "Fix partial FGW represented transport mass"
git push origin fgw-development
```

Update only the development checkout on Jean Zay:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching-next"
git branch --show-current             # must print fgw-development
git status --short                    # should be clean before pulling
git pull --ff-only origin fgw-development
```

If the development checkout ever needs to be recreated, use the raw repository
URL, not Markdown link syntax:

```bash
git clone \
  --branch fgw-development \
  https://github.com/alessandro-benvenuti/Graph-Native-Betti-Matching.git \
  "$WORK/projects/Graph-Native-Betti-Matching-next"
```

## 8. Separate A100 FGW environment

The FGW development checkout uses its own venv and CUDA-extension cache:

```text
$WORK/venvs/vascular-graph-extraction-a100-torch230-fgw
$WORK/.cache/torch-extensions/gnbm-a100-torch230-fgw-sm80
```

The frozen campaign continues to use its original environment and cache. The
development environment inherits Jean Zay's A100 PyTorch module:

```text
arch/a100
pytorch-gpu/py3/2.3.0
PyTorch 2.3.0
CUDA runtime 12.2
compute capability 8.0 (sm_80)
```

Set the development overrides before sourcing `env_a100.sh`:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching-next"

export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-next"
export GNBM_A100_VENV="$WORK/venvs/vascular-graph-extraction-a100-torch230-fgw"
export GNBM_A100_EXTENSIONS_DIR="$WORK/.cache/torch-extensions/gnbm-a100-torch230-fgw-sm80"

export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/new_patches_boundary"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm-fgw-development"
export WANDB_MODE=disabled

source cluster/jean_zay/env_a100.sh
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `WORK` | Jean Zay-provided work filesystem root; repositories, venvs, caches and logs. |
| `SCRATCH` | Jean Zay-provided project filesystem root; datasets and experiment outputs. |
| `GNBM_REPO_DIR` | Forces scripts and jobs to use the `-next` checkout. |
| `GNBM_A100_VENV` | Selects the isolated FGW A100 venv before `env_a100.sh` is sourced. |
| `GNBM_A100_EXTENSIONS_DIR` | Selects the isolated sm_80 extension cache. |
| `SYNTHETIC_MRI_DATASET` | Root containing materialized `train`, `val`, and `test` splits. |
| `GNBM_OUTPUT_DIR` | Parent directory for training outputs. Diagnostics use their explicit `--output-dir`. |
| `WANDB_MODE` | Use `disabled` for diagnostics and smoke tests; production normally uses the campaign setting. |
| `GNBM_INITIAL_WEIGHTS` | Model-only initialization checkpoint for a new training run. |
| `GNBM_RESUME_CHECKPOINT` | Full training-state checkpoint for continuation; mutually exclusive with initial weights. |
| `GNBM_AUTO_RESUME` | Set to `1` to reuse a run's `latest_checkpoint.pt`. |
| `GNBM_GPUS` | Number of A100s for the training launcher. |
| `GNBM_BATCH_SIZE` | Per-GPU batch-size override. |
| `GNBM_WORKERS` | DataLoader workers per process. |
| `GNBM_QOS` | `qos_gpu_a100-dev` or `qos_gpu_a100-t3`. |
| `GNBM_WALLTIME` | Slurm wall time in `HH:MM:SS`. |

The custom deformable-attention extension was built successfully in the FGW
venv. A direct extension import may fail with `libc10.so` if PyTorch has not
first loaded its shared libraries. Validate it with PyTorch imported first:

```bash
python -c 'import torch; import MultiScaleDeformableAttention3D; print("extension OK")'
```

Rebuild only inside an A100 allocation if required:

```bash
bash scripts/build_deformable_attention.sh
```

The `uv` message saying reflinks are unsupported is harmless; it falls back to
a normal copy.

## 9. Checkpoint and configurations

Checkpoint used by the offline matcher diagnostic:

```text
/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/finetune_full_mri_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt
```

Matching training configuration for that checkpoint:

```text
configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml
```

This checkpoint was trained with Hungarian matching. The offline diagnostic
does not retrain it; it asks how the frozen predictions would be associated by
Hungarian versus FGW. Therefore, a good diagnostic is necessary but does not by
itself prove that FGW training will improve final test metrics.

## 10. Tests

Run the matcher and related CPU tests from an environment containing the project
dependencies and POT:

```bash
python -m unittest \
  tests.test_matcher \
  tests.test_fgw_diagnostic \
  tests.test_graph_losses \
  tests.test_data_loaders \
  tests.test_training \
  tests.test_experiment_configs
```

Current expected result after the matcher/diagnostic/config correctness pass:

```text
Ran 77 tests
OK
```

Run the complete suite when the full dataset/medical-imaging dependencies and
distributed runtime are available:

```bash
python -m unittest discover -s tests
```

Some desktop environments may lack `nibabel`, and sandboxed local Gloo tests may
fail to resolve host networking. Those environment-specific failures do not
replace the Jean Zay model smoke test.

The full one-A100 model/runtime smoke test can be submitted from the development
checkout after exporting `GNBM_MRI_CHECKPOINT` and a dedicated output root:

```bash
export GNBM_MRI_CHECKPOINT=/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/finetune_full_mri_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm-fgw-debug"
bash cluster/jean_zay/submit_debug_a100.sh
```

A real one-A100 FGW training smoke run has already completed successfully using
`configs/smoke_synthetic_mri_focal_fgw.yaml` and run name `fgw-smoke-2`:

```text
epoch 1/1 train total: 1.390816
validation total:      2.498050
train time:            37.050 seconds
peak allocated memory: 2.160 GiB
training-complete marker created
```

That smoke proved the matcher was wired into training and produced finite
losses, but it used the earlier semi-relaxed implementation. The partial solver
still requires the offline invariant check and then another training smoke.

## 11. Offline diagnostic command

After committing, pushing, pulling and sourcing the isolated A100 environment,
run the corrected five-sample partial-FGW sanity check on one development A100:

```bash
srun \
  --account=vnc@a100 \
  --partition=gpu_p5 \
  --constraint=a100 \
  --qos=qos_gpu_a100-dev \
  --time=00:20:00 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --hint=nomultithread \
  python -u scripts/diagnose_fgw_matching.py \
    --config configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml \
    --checkpoint /lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/finetune_full_mri_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt \
    --output-dir /lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm-fgw-diagnostics/validation-partial-alpha-sanity-5-mass-fix \
    --split val \
    --max-samples 5 \
    --batch-size 1 \
    --workers 0 \
    --alphas 0.0 0.01 0.05 0.1 0.4 0.8 \
    --candidate-count 32 \
    --max-iter 10000
```

The Paramiko Blowfish deprecation and non-power-of-two attention-head warnings
seen at startup are unrelated to FGW.

The diagnostic writes:

```text
summary.json
per-sample.csv
per-sample.json
metadata.json
resolved-config.yaml
```

The most important alpha-zero invariants are:

```text
alpha_zero_status = same_mapping or tied_optimum
alpha_zero_unary_cost_delta approximately 0
max_prediction_capacity_ratio <= 1 (within numerical tolerance)
transport_total_mass approximately 1
invariant_failure = 0
```

For a deliberately unique unary optimum, `same_mapping` is required. A tied
optimum may change geometry/structure metrics even though its unary objective is
equal. `soft_argmax_collisions` need not be zero for a feasible fractional plan;
the column-capacity and marginal checks determine feasibility.

For $\alpha>0$, desirable tendencies are:

- `structural_mse_delta_vs_hungarian < 0`;
- `edge_nonedge_separation_delta_vs_hungarian > 0`;
- a small, controlled coordinate-error increase;
- nonzero but not wholesale assignment changes;
- no capacity violation.

The first alpha call may include several seconds of one-time POT import and
initialization. Later `matcher_seconds` values are more representative.

## 12. Next experiment sequence

1. Commit and push the floating-mass fix.
2. Pull it only into `Graph-Native-Betti-Matching-next`.
3. Rerun the five-sample validation alpha-zero sanity check above.
4. If alpha zero has a unary-cost regression (or changes a deliberately unique
   optimum), stop and debug the partial constraints or projection before any
   larger experiment. A classified tied optimum is not a regression.
5. If it passes, inspect the higher-alpha results and per-sample CSV.
6. Run an augmentation-free 50-sample validation diagnostic using the useful
   alpha range and a distinct output directory.
7. Run the same offline comparison on an augmentation-free subset of the
   training split with `--split train`. This determines whether the matcher
   changes associations on predictions the checkpoint was optimized on.
8. Select one conservative alpha and rerun the one-epoch FGW training smoke
   using the partial matcher.
9. Only after all invariants and smoke tests pass, launch a controlled FGW
   training comparison against the Hungarian baseline.

For the 50-sample validation run, increase the allocation conservatively, for
example to one hour, and change only:

```text
--time=01:00:00
--max-samples 50
--output-dir .../validation-partial-50
```

For the later training-set diagnostic, use a new directory and:

```text
--split train
```

The diagnostic loader explicitly disables train augmentation so both matchers
see deterministic examples.

## 13. Matcher and diagnostic status after the local correctness pass

Candidate selection now returns exactly
`min(query_count, max(target_count, candidate_count))`: it retains every unary
Hungarian query and fills only the remaining slots in descending object
confidence order. The boundary case where the pool was already full is covered
by a regression test.

The optional matcher diagnostic path asks POT for its log and records the loss
history, number of conditional-gradient updates, last absolute and relative
objective changes, final POT objective, final EMD result/warning, captured
warnings, and solver time. POT does not expose an authoritative general-purpose
`converged` flag here, so none is invented. `termination_evidence` says only
what the available trace supports: tolerance met, iteration limit reached, or
not exposed by POT. Normal training calls retain the original assignment-only
interface.

For a single fixed candidate pool, each alpha now evaluates the actual
squared-loss objective for the unary Hungarian initialization, final soft plan,
and hardened plan. The implementation uses row/column masses and matrix
products rather than allocating an `(n,n,q,q)` tensor, and a tiny brute-force
test checks the formula. Reports include soft/hard changes from initialization
and `F(T_hard)-F(T_soft)`; there is intentionally no silent Hungarian fallback.

Offline output retains the five existing files and adds edge/nonedge squared
errors, undefined-metric counts for edgeless/complete/tiny graphs, summaries by
GT node count, candidate/query-subset information, separate relation-scoring
and solver timing, invariant failures, alpha-zero tie classification, package
versions, seed, solver settings, and Git revision/local-change state.

The matcher-only full comparison is:

```text
configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm_fgw_controlled.yaml
```

It inherits the exact checkpoint experiment and overrides only its experiment
name and matcher block. Its bounded one-epoch derivative is:

```text
configs/experiments/full_dataset_comparison/smoke_nodefocal_edgefocal_mm_fgw_controlled.yaml
```

`random_state` remains accepted for old configurations but is unused because
this path has a deterministic Hungarian initialization and no stochastic solver
step. New configurations omit it. Betti H0/H1 losses remain independently
configured and are neither enabled nor retuned by the FGW matcher overlay.

Improved assignment stability and improved final model quality remain
hypotheses. A small deterministic mechanism test only establishes that a
suitable alpha can prefer a clearly better structural correspondence; it is not
evidence of training improvement.

## 14. Exact next commands (development checkout only; do not run in campaign checkout)

Set the isolated environment and immutable starting checkpoint once:

```bash
cd "$WORK/projects/Graph-Native-Betti-Matching-next"
export GNBM_A100_VENV="$WORK/venvs/vascular-graph-extraction-a100-torch230-fgw"
export GNBM_INITIAL_WEIGHTS=/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/finetune_full_mri_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm-fgw-controlled"
```

### A. Five-sample frozen-checkpoint validation diagnostic

```bash
srun --account=vnc@a100 --partition=gpu_p5 --constraint=a100 \
  --qos=qos_gpu_a100-dev --time=00:20:00 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread \
  "$GNBM_A100_VENV/bin/python" -u scripts/diagnose_fgw_matching.py \
  --config configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml \
  --checkpoint "$GNBM_INITIAL_WEIGHTS" \
  --output-dir "$SCRATCH/experiments/gnbm-fgw-diagnostics/validation-partial-objectives-5" \
  --split val --max-samples 5 --batch-size 1 --workers 0 \
  --alphas 0 0.01 0.05 0.1 0.4 0.8 --candidate-count 32 \
  --max-iter 10000 --tolerance 1e-7
```

### B. Larger validation and augmentation-free training-subset diagnostics

Run these only if A has no invariant failures and alpha zero is either
`same_mapping` or `tied_optimum` with unary-cost delta within tolerance:

```bash
srun --account=vnc@a100 --partition=gpu_p5 --constraint=a100 \
  --qos=qos_gpu_a100-dev --time=01:00:00 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread \
  "$GNBM_A100_VENV/bin/python" -u scripts/diagnose_fgw_matching.py \
  --config configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml \
  --checkpoint "$GNBM_INITIAL_WEIGHTS" \
  --output-dir "$SCRATCH/experiments/gnbm-fgw-diagnostics/validation-partial-objectives-50" \
  --split val --max-samples 50 --batch-size 1 --workers 0 \
  --alphas 0 0.01 0.05 0.1 0.4 0.8 --candidate-count 32 \
  --max-iter 10000 --tolerance 1e-7

srun --account=vnc@a100 --partition=gpu_p5 --constraint=a100 \
  --qos=qos_gpu_a100-dev --time=01:00:00 --nodes=1 --ntasks=1 \
  --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread \
  "$GNBM_A100_VENV/bin/python" -u scripts/diagnose_fgw_matching.py \
  --config configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml \
  --checkpoint "$GNBM_INITIAL_WEIGHTS" \
  --output-dir "$SCRATCH/experiments/gnbm-fgw-diagnostics/train-noaug-partial-objectives-50" \
  --split train --max-samples 50 --batch-size 1 --workers 0 \
  --alphas 0 0.01 0.05 0.1 0.4 0.8 --candidate-count 32 \
  --max-iter 10000 --tolerance 1e-7
```

### C. One-epoch controlled partial-FGW GPU smoke

```bash
export GNBM_QOS=qos_gpu_a100-dev
export GNBM_WALLTIME=00:30:00
export GNBM_GPUS=1
export GNBM_BATCH_SIZE=1
bash cluster/jean_zay/submit_train_a100.sh \
  configs/experiments/full_dataset_comparison/smoke_nodefocal_edgefocal_mm_fgw_controlled.yaml \
  partial-fgw-controlled-smoke-1ep
```

### D. Matched Hungarian-versus-FGW training comparison

Both jobs below use the same `GNBM_INITIAL_WEIGHTS`, dataset, seed, global batch,
optimizer, schedule, losses, and evaluation protocol. Use distinct run names;
the common output root is safe because run name selects the subdirectory.

```bash
export GNBM_QOS=qos_gpu_a100-t3
export GNBM_WALLTIME=20:00:00
export GNBM_GPUS=4
export GNBM_BATCH_SIZE=8
bash cluster/jean_zay/submit_train_a100.sh \
  configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm.yaml \
  controlled-hungarian-seed364505
bash cluster/jean_zay/submit_train_a100.sh \
  configs/experiments/full_dataset_comparison/finetune_nodefocal_edgefocal_mm_fgw_controlled.yaml \
  controlled-partial-fgw-seed364505
```

Proceed to D only after: all mass/capacity/uniqueness invariants pass; alpha zero
has equal unary cost (mapping equality only when unique); POT traces show no
unexplained warnings/failures or systematic iteration-limit termination; soft
improvements usually survive hardening without unacceptable projection gaps;
true-edge and nonedge errors, geometry, and behavior by graph size are
acceptable; and relation/solver runtime plus the one-epoch smoke are practical.
Do not choose alpha from aggregate structural MSE alone. Final inference metrics
from the matched training runs decide whether graph-aware matching improves the
model.
