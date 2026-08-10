# Archived domain-adaptation experiment

## Status

Adversarial domain adaptation is not part of the active Graph-Native Betti
Matching baseline. Its configuration, discriminator modules, gradient-reversal
path, optimizer group, loss calculation, and TensorBoard scalar must remain
absent unless a new controlled experiment explicitly reintroduces them.

Source and target labels remain in the model-facing batch. They describe dataset
roles and may be used for sampling, filtering, and diagnostics; their presence
does not mean that adversarial domain adaptation is enabled.

## Why it was removed

The pre-focal, pre-Betti reference repository contained image-level and
instance-level domain discriminators and calculated a domain loss. However, its
committed 3D configurations enabled only `boxes`, `class`, `cards`, `nodes`, and
`edges`. The criterion constructed `total` only from that list, and the trainer
called `backward()` only on `total`. Consequently, the domain loss was logged but
did not update either discriminator or the graph-extraction network.

Reference implementation:

- model discriminators: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/models/relationformer.py#L69-L71>
- domain-loss calculation: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L749-L777>
- optimized-total dispatch: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/losses.py#L850-L868>
- reference 3D loss list: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/configs/mixed_plants_synth_HNS_3D.yaml#L138-L145>
- backward call: <https://github.com/alexscavo/Vascular-Graph-Extraction/blob/main/3d/training/trainer.py#L318-L345>

This was therefore dead training logic in the effective baseline, despite the
adversarial flags and the `domain_loss` TensorBoard curve. Removing it preserves
the optimized objective while reducing computation, memory use, checkpoint
size, and configuration ambiguity. Git history remains the source for the old
implementation.

## Intended objective

The archived design followed the domain-adversarial pattern used by
Domain-Adaptive Faster R-CNN rather than the core RelationFormer objective. A
gradient-reversal layer was applied before each domain discriminator. The
intended loss was

```text
L_domain = w_image * NLL(image_domain_prediction, domain)
         + w_graph * NLL(graph_domain_prediction, domain)
         + w_consistency * MSE(mean_image_prediction, graph_prediction)

L_total = L_graph + w_domain * L_domain
```

During backpropagation, gradient reversal multiplies the feature gradient from
`L_domain` by `-alpha`, encouraging domain-invariant features while the
discriminators learn to classify source versus target.

## Historical configurations

The effective GitHub baseline had adversarial flags set but omitted `domain`
from `TRAIN.LOSSES`, so `W_DOMAIN` had no effect.

A separate cluster run did activate the experiment with the following relevant
legacy settings:

```yaml
TRAIN:
  DOMAIN_LR: 0.00007
  DOMAIN_WEIGHTING: [0.05, 0.95]
  IMAGE_ADVERSARIAL: true
  GRAPH_ADVERSARIAL: true
  CONSISTENCY_REGULARIZATION: true
  ALPHA_COEFF: 1.0
  LOSSES: [boxes, class, cards, nodes, edges, domain]
  W_DOMAIN: 1.0
```

That run is an experimental variant, not evidence that domain adaptation was
part of the best committed baseline.

The old fields `UPSAMPLE_TARGET_DOMAIN` and `COMPUTE_TARGET_GRAPH_LOSS` were not
intrinsically adversarial. Their active replacements are:

```yaml
data:
  mixed_sampling:
    balance_source_target: true

loss:
  supervise_target_graphs: true
```

## Reactivation contract

If domain adaptation is tested again, it should be introduced as one explicit,
self-contained configuration block:

```yaml
domain_adaptation:
  enabled: true
  image_weight: 1.0
  graph_weight: 1.0
  consistency_weight: 1.0
  loss_weight: 1.0
  gradient_reversal_alpha: 1.0
  learning_rate: 0.00007
  class_weights: [0.05, 0.95]
```

`enabled: true` must have all of the following observable effects:

1. construct only the requested discriminator modules;
2. add their parameters to an explicit optimizer group;
3. calculate `L_domain`;
4. include `loss_weight * L_domain` in the tensor passed to `backward()`;
5. log both raw and weighted domain losses;
6. fail configuration validation if there is no source/target mixture.

`enabled: false` must construct no discriminators, perform no discriminator
forward passes, add no optimizer group, and emit no domain-loss metric. The loss
must never be enabled indirectly through an independent list of names.

Before accepting a reimplementation, add tests that demonstrate non-zero
gradients in both a discriminator and an encoder parameter, zero domain-related
state when disabled, and checkpoint loading behavior with and without legacy
discriminator keys. Compare at least image-only, graph-only, combined, and
combined-with-consistency variants against the unchanged baseline.
