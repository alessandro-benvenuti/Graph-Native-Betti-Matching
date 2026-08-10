# Configuration contract

Configuration is split into one complete baseline and small experiment overlays.
`configs.load_config` recursively merges the files listed in `defaults`, from
left to right, expands environment variables, and validates the resolved
configuration before a dataset is constructed.

## Files

- `base.yaml`: shared model, optimizer, loss, augmentation, evaluation, and
  topology defaults.
- `pretrain_mixed.yaml`: Plants + syntheticMRI mixed pretraining.
- `finetune_synthetic_mri.yaml`: syntheticMRI-only finetuning with
  cross-entropy edge classification.
- `finetune_synthetic_mri_focal.yaml`: the focal-edge variant. It differs from
  the finetuning configuration only where the experiment is intentionally
  different.

Dataset and output paths are environment-variable references, not machine paths.
The loader reports an unset variable together with its configuration location.

```python
from configs import load_config
from data.loaders import build_data_loaders

config = load_config("configs/pretrain_mixed.yaml")
train_loader, validation_loader = build_data_loaders(config)
```

## Edge candidate and balancing semantics

`loss.edge.candidates.max_per_graph` is an upper bound on the number of matched
edge candidates generated for one graph. It is not an upsampling target. With
120 object queries, at most `120 * 119 / 2 = 7140` undirected pairs exist, so
the compatibility value `9999` currently means "retain all pairs".

`loss.edge.balancing` is a later, independent operation on the loss examples:

- `ratio_upsample` duplicates minority-class examples until the configured
  positive-to-negative ratio is within tolerance;
- `none` performs no duplication, downsampling, or truncation.

The old `EDGE_UPSAMPLING` boolean is intentionally absent because it was unused.
There must be only one source of truth: `loss.edge.balancing.mode`.

The previous loss also truncated positive edges to 40 when sampling mode was
`none`. That behavior is a bug and must not be ported. `positive_cap: null`
records the intended invariant explicitly. Tests for the future loss module must
verify that focal mode retains more than 40 positive edges.

Validation rules for loss construction:

1. `loss.edge.classification.name: focal` requires
   `loss.edge.balancing.mode: none`.
2. `ratio_upsample` requires a positive ratio and a non-negative tolerance.
3. `max_per_graph` must be positive or `null` (`null` means all candidates).
4. `positive_cap` must remain `null` unless a separately reviewed experiment
   introduces an explicit positive cap.
5. Validation and test never perform edge-class balancing.

## Compatibility decisions

The baseline numerical values come from the established experiment configurations.
Configuration names are reorganized to describe behavior instead of preserving
the old layout. Diameter smoothing is intentionally omitted. Inactive road,
OCTA, MoCo, 2D, and adapter experiments are also omitted.

CLI overrides should be restricted to operational values such as paths, batch
size, worker count, resume checkpoint, and run name. Scientific settings such as
the loss type or balancing mode belong in version-controlled YAML files so the
resolved configuration fully describes an experiment.
