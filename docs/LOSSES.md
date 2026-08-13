# Graph loss stack

The implementation is split by responsibility under `training/losses/`:

- `criterion.py` owns matching, baseline supervision, edge candidates, and
  opt-in integration;
- `focal.py` contains focal classification and hard-negative selection;
- `betti_h0.py` and `betti_h1.py` contain graph-native topology losses.

`training.losses.build_criterion(config, model)` is the supported entry point.

## Baseline behavior

The default configuration preserves the established 3D objective:

1. Hungarian assignment combines object probability with L1 distance over the
   first three predicted node coordinates.
2. Node classification uses weighted cross-entropy with `[0.25, 0.75]` and
   class `1` as the object class.
3. Node regression is summed L1 error divided by target-node count.
4. The box term uses the final three node-head values as predicted sizes and a
   fixed target size of `0.2` before 3D generalized IoU.
5. Relation supervision uses matched nodes, one random orientation per
   undirected pair, cross-entropy, and legacy ratio upsampling.
6. Cardinality is normalized L1 error in predicted object count.

Disabled extensions do not score candidates, consume random values, or enter
the optimized objective.

## Betti extensions

`topology.betti_h0` and `topology.betti_h1` are independent switches. Both use
the complete graph over Hungarian-matched nodes. Every undirected pair is
scored in both endpoint orders and averaged. Discrete persistence matching uses
detached probabilities, while selected filtration values retain gradients.
Warmup, ramp, weight, normalization, and `log_only` are YAML-controlled.

## Focal loss and hard-negative mining

Node and edge classification can independently select `focal`. Hard-negative
mining belongs to the focal edge candidates rather than being a separate loss:

```yaml
loss:
  edge:
    classification:
      name: focal
    candidates:
      include_unmatched: true
```

Active unmatched queries form candidates with matched and other active
unmatched queries. Each candidate is evaluated in both endpoint orders and the
two logit vectors are averaged, making hard-negative selection and focal
supervision explicitly undirected. Selection ranks detached edge probabilities
but indexes the original symmetric logits, preserving gradients through both
relation-head evaluations and their endpoint tokens. Caps, threshold, weight,
warmup, and ramp are configurable.

## Explicit exclusions

Diameter smoothing, degree matching, domain-adaptation loss, and their logging
and profiling code are not part of this stack. There is no dormant diameter
path or YAML option.
