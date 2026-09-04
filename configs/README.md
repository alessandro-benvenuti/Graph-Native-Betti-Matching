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
- `finetune_synthetic_mri_focal_fgw.yaml`: a ready-to-run focal finetuning
  configuration using the FGW matcher.
- `experiments/finetune_mri/`: controlled baseline/focal/Betti/focal+Betti
  finetuning matrix. See `docs/EXPERIMENTS.md` before launching it.
- `experiments/focal_matrix_600/`: paired mixed-pretraining and 600-epoch MRI
  specialization configurations for the seven unweighted-focal recipes.
- `losses/`: reusable focal and Betti overlays containing only loss changes;
  they never modify datasets, model architecture, or optimization settings.
- `matchers/fgw.yaml`: semi-relaxed Fused Gromov-Wasserstein matching followed
  by a hard one-to-one projection.
- `smoke_mixed_focal_betti.yaml`: one-epoch, four-sample integration check for
  the complete focal + topology training path.
- `overfit_synthetic_mri_focal_betti.yaml`: ten-epoch fixed eight-sample MRI
  overfit check with augmentation disabled and one best checkpoint.

The inference thresholds and complete metric protocol are defined under
`evaluation` in `base.yaml`. See `docs/EVALUATION.md`; changing those values
changes the scientific evaluation protocol and belongs in a version-controlled
configuration overlay.

Dataset and output paths are environment-variable references, not machine paths.
The loader reports an unset variable together with its configuration location.
W&B defaults live under `tracking`; credentials and deployment-specific
project, entity, group, and mode may use standard `WANDB_*` environment
variables.

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

Checkpoint policies are `none`, `best_only`, `interval`, and
`interval_and_best`. Storage-constrained pilots use `best_only`, which
overwrites `best_checkpoint.pt` and never writes epoch checkpoints.

## Compatibility decisions

The baseline numerical values come from the established experiment configurations.
Configuration names are reorganized to describe behavior instead of preserving
the old layout. Diameter smoothing is intentionally omitted. Inactive road,
OCTA, MoCo, 2D, and adapter experiments are also omitted.

Adversarial domain adaptation is also absent from the active schema. The
published baseline constructed and logged its domain loss but did not include it
in the optimized total. Source/target sampling is independently configured by
`data.mixed_sampling.balance_source_target`, and target graph supervision by
`loss.supervise_target_graphs`. The archived implementation and reactivation
contract are documented in `docs/experiments/domain_adaptation.md`.

CLI overrides should be restricted to operational values such as paths, batch
size, worker count, resume checkpoint, and run name. Scientific settings such as
the loss type or balancing mode belong in version-controlled YAML files so the
resolved configuration fully describes an experiment.

## FGW matcher

The FGW overlay replaces only the training-time matcher. Ground-truth nodes are
the fixed-mass source measure, prediction queries are the relaxed target
measure, and the returned transport is projected globally to one-to-one hard
matches before the existing node, edge, and topology losses run.

Scoring the complete graph of 120 queries would require 7,140 undirected query
pairs, or 14,280 ordered relation evaluations after symmetrization, per sample.
`candidate_count` bounds this work. The candidate pool always contains every
coordinate/class Hungarian match and is filled with the highest-confidence
remaining queries. Set it to 120 for exhaustive diagnostic runs.
`pair_chunk_size` bounds temporary relation-feature memory and does not change
the selected candidates or objective.

Use the overlay after a complete training configuration, for example:

```yaml
defaults:
  - finetune_synthetic_mri_focal.yaml
  - matchers/fgw.yaml
```

## Model compatibility values

`model.decoder.encoder_layers: 6` and `model.encoder.strides: [1,2,2,2]`
describe the effective legacy architecture. The old YAML displayed 4 encoder
layers and `[2,2,2,2]`, but its builder silently used the transformer default of
6 and ignored the first stage stride. These corrected fields are now consumed by
the model builder. See `docs/MODEL.md` for the complete compatibility contract.

See `docs/LOSSES.md` for the baseline, Betti, and focal/HNM loss contract.
