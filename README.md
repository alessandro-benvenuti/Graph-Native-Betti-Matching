# Graph-Native Betti Matching

Clean research implementation of a 3D graph-extraction pipeline with graph-native topology objectives.

This repository is being reconstructed from an experimental codebase. It currently contains the reviewed augmentation implementation, configuration schema, and supported dataset loaders; the remaining modules and interfaces will be migrated incrementally.

## Intended scope

The first supported workflow will focus on:

- 3D training and checkpoint evaluation;
- mixed plants and synthetic-MRI pretraining;
- synthetic-MRI finetuning;
- RelationFormer-style node and relation prediction;
- EMA-assisted hard-negative mining;
- focal relation objectives;
- graph-native Betti H0/H1 objectives;
- BatchNorm-calibrated evaluation.

Diameter-smoothing losses, metrics, profiling tools, and experiment configurations are intentionally out of scope.

Road, OCTA, MoCo, 2D pretraining, and general-transformer adapter paths will not be migrated unless they are explicitly restored as supported requirements.

## Planned structure

```text
.
├── configs/                 Reproducible training and evaluation configurations
├── data/
│   ├── loaders/             Dataset loading and batching
│   └── preprocessing/       Dataset creation, splitting, and normalization
├── models/
│   ├── domain_adaptation/   Domain classifiers and gradient reversal
│   └── ops/                 Custom deformable-attention extension
│       ├── functions/       Python autograd bindings
│       ├── modules/         PyTorch modules
│       └── src/
│           ├── cpu/         C++ CPU implementation
│           └── cuda/        CUDA implementation
├── training/
│   ├── losses/              Detection, relation, domain, and topology losses
│   └── evaluation/          Inference, metrics, and BN calibration
├── metrics/                 Reusable evaluation metrics
├── boxes/                   Box and non-maximum-suppression operations
├── utils/                   Small shared utilities
├── tests/                   Unit and integration tests
├── scripts/                 Reproducible workflow entry points
└── docs/                    Training, configuration, and evaluation documentation
```

## CUDA extension policy

The custom CUDA/C++ source code required by deformable attention belongs in `models/ops/src/` and will be version-controlled.

Precompiled shared libraries, object files, and `build/` directories will not be committed. They are tied to the operating system, CPU architecture, Python version, PyTorch version, CUDA toolkit, compiler, and C++ ABI of the machine that built them. The extension must therefore be compiled in the Jean Zay environment used for training.

The future build documentation will record the tested Jean Zay modules and provide a reproducible build or installation command. Build output should be generated locally on the cluster and can be cached outside Git when useful.

## Status

The augmentation and data-loading contracts, merged/validated configuration loading, reproducible batching, and optional real-dataset checks are in place. No training, model, loss, evaluation, or CUDA implementation has been migrated yet.
