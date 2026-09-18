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

`topology.betti_h0` and `topology.betti_h1` are independent switches. Every
undirected pair is scored in both endpoint orders and averaged. Discrete
persistence matching uses detached probabilities, while selected filtration
values retain gradients. Warmup, ramp, weight, normalization, and `log_only`
are YAML-controlled.

`topology.complex.mode: matched_only` preserves the original complete graph
over matched queries. `node_aware` appends a capped top-confidence set of
unmatched queries. Matched vertices use existence confidence `1`; selected
unmatched vertices use their live object probability `q_i`. Edge confidence is
one of

```text
min:     min(q_i, q_j, p_ij)
product: q_i q_j p_ij
hybrid:  alpha min(q_i, q_j, p_ij) + (1-alpha) q_i q_j p_ij
```

and the node/edge filtrations are `1-q_i` and `1-s_ij`. All three rules obey
the face condition, and all reduce to the original edge filtration when both
endpoints are matched. Top-k selection is detached; selected node and relation
probabilities remain differentiable. H0 uses the single matched target
component as the omitted reduced-H0 class, so node-aware H0 is skipped for an
empty target. Coordinates and the discrete query selection receive no topology
gradient.

Selected unmatched vertices also receive an explicit filtration mismatch:
their predicted vertex filtration is `1-q_i`, while target absence is placed
at filtration `1`, producing the penalty `q_i^2`. With `normalization:
matched_mean`, genuine correspondences are averaged but false, missed, and
absent-vertex penalties are summed. Adding another error therefore cannot
dilute the existing loss through a larger feature-count denominator.

When `detach_unmatched_edge_probabilities` is enabled, edges incident to an
unmatched topology vertex still participate in the forward filtration, but
their relation probabilities are detached. Gradients continue through the
unmatched node probability; the shared relation head is supervised only by
edges whose two endpoints correspond to target vertices.

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
