# Local validation, 22 October 2026

**Real Jean Zay checkpoint/data execution remains pending.** The configured SSH
identity was rejected; the user chose to run the supplied commands on Jean Zay.
No cluster jobs were submitted, changed, cancelled or restarted.

The original checkout remains on `features-gnbm-nodes-edges` at
`665e2f0de67a7dd5b9b72c608185a49788325984`, with its pre-existing untracked
`output/`. The remote source branch was checked before cloning and matched that
commit. All code changes are confined to the independent disposable clone and
`codex/cached-logit-betti-test`.

Validation command in README: **54 tests passed**, including 15 new experiment
tests plus existing production H0, H1, filtration, graph-loss and spatial-cycle
audit tests. Python compilation and Bash syntax checks passed. The cache stage
was exercised with a tiny stand-in network using strict checkpoint loading,
eval/no-grad assertions, frozen BN buffers, actual Hungarian matching and both
directed relation heads. This does not validate the unavailable A100 extension
or real checkpoint; the cluster smoke checks those before main execution.

The local synthetic two-patch CLI ran all three arms in both modes for 200
updates, wrote all reports/plots/gradient states, and reproduced step-zero
metrics. Both classification and Betti arms reached edge F1 1 and zero Betti
errors on these two tiny fixtures. There was no endpoint benefit beyond
classification in this toy smoke. Unit tests separately include a broken
true loop, missing connection and already-correct graph. These are diagnostic
correctness checks, not scientific results about the source network.

One-CPU-thread timings (local PyTorch 2.9.1; cluster uses its existing environment):

|Case|Updates|Seconds|
|---|---:|---:|
|Complete two-patch, six-arm/mode CLI including reports|200 each|4.3|
|3-vertex Mode B H0+H1 benchmark|20|0.031|
|4-vertex Mode B H0+H1 benchmark|20|0.032|
|12 vertices / 66 pairs, synthetic Mode B H0+H1|3|0.024|
|24 vertices / 276 pairs, synthetic Mode B H0+H1|3|0.139|
|48 vertices / 1,128 pairs, synthetic Mode B H0+H1|3|1.965|

These timings show why the actual cached graphs must be benchmarked before a
main resource estimate is accepted. There is no measured real-data runtime yet.
The job scripts request one A100/eight CPUs/one hour for caching, then zero
GPUs/four CPUs/one hour for bounded CPU optimization. They do not increase
resources automatically if the CPU projection exceeds the bound.

Local generated root:
`/Users/alessandrobenvenuti/Desktop/Università/Thesis/Graph-Native-Betti-Matching-cached-logit-test/artifacts/cached-logit-local-smoke/`.

Created artifact subdirectories: `cache/`, `run/`, `benchmark/`, `report-v2/`;
final clean-commit replay uses `final-cache/` and `final-run/`. Root files include
`verification.json`, `scaling-benchmark.json` and `inventory.json`. Earlier runs
are retained without overwriting. Each run's twelve `.pt` files retain Adam
state; JSON/CSV reports, trajectory PNGs and gradient audits are retained.
`inventory.json` records the final absolute directory/file inventory with SHA256.
The synthetic cache is explicitly tagged and its verification cannot authorize
the cluster main-run gate.

New tracked directories: `experiments/`, `experiments/cached_logits/`,
`experiments/cached_logits/cluster/`. The full independent clone has its own
`.git`; Python creates ignored `__pycache__` directories there. Matplotlib's
local cache is `/private/tmp/gnbm-cached-logit-mpl`. Unit-test temporary
directories are automatically removed. No automatic cleanup or merge is done.

Scientific question still pending: whether Betti improves real graph correctness
beyond classification, and whether the descriptive result changes in B. The
real report will compare the fixed endpoint and trajectories, separate GT/support
errors and exact cycle-space recovery, and expose correct-edge damage and cycle
turnover. Successful free-logit optimization alone cannot establish a benefit
for network training.

## Batch Git follow-up

The user-provided log for job 1440813 showed `git: command not found` in the
initial clone guard, before any prediction-cache work. The experimental setup
now loads Git when absent, rechecks after module purges, and fails explicitly
on Git-status errors. A directory-identity check supports filesystem aliases.
Four shell regression tests emulate missing Git, environment purge, failed
module loading and failed status. The complete focused suite now passes
**58 tests**; all five experimental Bash files pass syntax checks. Real cluster
verification remains pending resubmission by the user. Production environment
scripts and resource requests were not changed.


## Login-only Git provenance follow-up

The user-provided log for job 1442109 showed that compute-node `git status`
still failed. The new submission helper captures branch, clean status, commit
and all tracked file hashes on the login node. Compute jobs perform hash
verification with the Python standard library and never invoke Git. They
reject changed files, added Python sources and missing/altered snapshots.
Both actual batch scripts were exercised locally with Git absent from PATH;
only expensive model/data commands were stubbed in that startup test. This is
not a claim of successful cluster execution. Ten provenance/startup tests pass; the complete focused suite passes 64 tests.
Submit using `bash experiments/cached_logits/cluster/submit.sh smoke`; direct
`sbatch` calls without a captured snapshot now fail before model/data work.


## Historical configuration follow-up

Job 1444630 passed isolated-path checks, snapshot verification and A100/PyTorch
preflight, then failed loading a source config without the current topology
sampling mapping. Local regression reproduces that exact production-loader
error. The experiment-only adapter supplies and audits only missing modern
topology schema fields while preserving original coefficients and source bytes.
The cache integration test now loads a historical-style YAML through the real
adapter and validator; configuration loading is no longer mocked. Frozen-ID
reuse and preservation of failed cache directories have regression coverage.
No successful real cache or optimization has been reported yet.
The full focused suite now passes **70 tests**, including six schema/retry
regressions. Bash syntax, Python compilation and diff checks pass. The existing
failed cache and frozen two-sample manifest are preserved by the retry path.


## Measured main-cache CPU budget follow-up

The user reported CPU benchmark job 1447879 completed successfully: 8/26/37
vertices, 28/325/666 pairs, and .278/.624/2.166 seconds for five updates each.
Its conservative maximum-support projection is 24,955.139 seconds. An explicit
`main-long`/`resume-long` route now offers an eight-hour CPU allocation with a
27,000-second runtime cap. Normal one-hour submissions remain gated at 3,300
seconds. Source scientific settings, core objective and cached data are unchanged.
The scheduler is preflighted with `sbatch --test-only` before a longer submission.
Local tests validate both actual submission routes using a mocked scheduler,
exercise actual batch startup/main/resume scripts without Git, and confirm
identical short-vs-long-budget synthetic endpoint metrics. No long main run
has been submitted by this agent; scientific conclusions remain pending.
The full focused suite passes **75 tests** after the runtime-control change.
The tested submission helper checks the eight-hour request with Slurm test-only;
its ordinary stages retain one-hour requests. Core losses, cache construction,
selection, metrics and configuration adapters have no changes in this update.
