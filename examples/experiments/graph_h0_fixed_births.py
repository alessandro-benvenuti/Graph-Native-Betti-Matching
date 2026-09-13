"""Fixed-vertex H0 Betti matching: executable research/demo implementation.

Scope
-----
Hungarian-aligned vertices are assumed to exist.  They have filtration zero;
edges have f_p(e)=T*(1-p_e), with binary target edge probabilities.  Prediction,
target and comparison share a connected candidate graph (complete by default).
The comparison is the simplex-wise minimum, hence K_u(t)=K_p(t) union K_g(t).
No triangles are filled; this module supervises H0 only.

Ordinary finite pairs are (0,d).  Zero-length pairs participate in the forest
but are omitted from the loss.  The one essential pair (0,infinity) is recorded
for display and omitted from optimization: its birth is fixed and its cap,
if used, would be the same fixed T in both graphs.  We retain finite deaths at
T: they encode distinct target components, including isolated target vertices.

Matching
--------
This is the graph specialization of the refined/extended H0 construction in
Stucki et al., ICML 2023, Appendix C:
https://proceedings.mlr.press/v202/stucki23a/stucki23a.pdf
Ordinary pairs are indexed by their birth simplex.  A comparison forest sweep
computes image pairs (source birth vertex, comparison death edge).  Matching
image-pair left endpoints to source pairs and right endpoints to comparison
pairs gives two partial matchings.  Compose them through the SAME comparison
pair.  This is a spatially labelled refinement, not a diagram-distance search.
All vertex births tie; vertex IDs refine those ties, and edges tie by canonical
endpoint IDs.  These are explicit conventions, not a claim of a unique or
vertex-relabel-invariant barcode matching.  Input edge enumeration is immaterial.

Loss (N is the number of matched plus unmatched features)
-------------------------------------------------------
L = [sum_matched (d_p-d_g)^2
     + eta/2 * sum_unmatched_prediction d_p^2
     + eta/2 * sum_unmatched_target d_g^2] / max(1,N).
With connected GT, there are no positive finite target bars; minimizing the
prediction terms strengthens the selected spanning-tree edges.  An unmatched
prediction bar here means a connectivity delay, not necessarily a false vessel.
With disconnected GT, matched deaths are pushed toward T (suppress bridges).

The PyTorch adapter detaches only combinatorics and reindexes live probabilities.
It is piecewise differentiable; ties/feature-set changes need not be smooth.
This file is independent of the training package and does not modify its loss.
Run `python graph_h0_fixed_births.py` for mathematical and autograd checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Mapping

Edge = tuple[int, int]


def edge(u: int, v: int) -> Edge:
    if int(u) != u or int(v) != v or u == v:
        raise ValueError('Edges need distinct integer endpoints.')
    return tuple(sorted((int(u), int(v))))


def complete_edges(n: int) -> tuple[Edge, ...]:
    return tuple(combinations(range(n), 2))


@dataclass(frozen=True)
class Graph:
    n: int
    edges: tuple[Edge, ...]
    values: tuple[float, ...]
    births: tuple[float, ...]
    terminal: float = 1.0

    def __post_init__(self):
        if self.n < 1 or len(self.births) != self.n:
            raise ValueError('At least one vertex and one birth per vertex required.')
        if not isfinite(self.terminal) or self.terminal <= 0:
            raise ValueError('terminal must be finite and positive.')
        if len(self.edges) != len(self.values) or len(set(self.edges)) != len(self.edges):
            raise ValueError('Candidate edges must be unique and match the values.')
        if any(not isfinite(b) or not 0 <= b <= self.terminal for b in self.births):
            raise ValueError('Invalid vertex birth.')
        uf = UnionFind(self.births)
        for (u, v), f in zip(self.edges, self.values):
            if not (0 <= u < v < self.n):
                raise ValueError('Edges must be canonical and endpoints in range.')
            if not isfinite(f) or not max(self.births[u], self.births[v]) <= f <= self.terminal:
                raise ValueError('Invalid filtration or violated face condition.')
            uf.link((u, v))
        if len(uf.components()) != 1:
            raise ValueError('The candidate graph must be connected at T.')

    @property
    def order(self):
        return sorted(range(len(self.edges)), key=lambda i: (self.values[i], self.edges[i]))


class UnionFind:
    def __init__(self, births):
        self.parent = list(range(len(births)))
        self.births = births

    def find(self, v):
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def link(self, e):
        a, b = self.find(e[0]), self.find(e[1])
        if a == b:
            return None
        older, younger = sorted((a, b), key=lambda v: (self.births[v], v))
        self.parent[younger] = older
        return younger

    def components(self):
        groups = {}
        for v in range(len(self.parent)):
            groups.setdefault(self.find(v), []).append(v)
        return tuple(tuple(group) for _, group in sorted(groups.items()))


def from_probabilities(n, probabilities: Mapping[Edge, float], *, terminal=1., vertex_mode='fixed'):
    """Legacy incident-max births are available ONLY for the visual comparison."""
    canonical = {}
    for e, p in probabilities.items():
        e = edge(*e)
        if e in canonical:
            raise ValueError('Duplicate undirected edge.')
        if not isfinite(float(p)) or not 0 <= float(p) <= 1:
            raise ValueError('Probabilities must be finite and in [0,1].')
        if not 0 <= e[0] < e[1] < n:
            raise ValueError('Endpoint outside vertex set.')
        canonical[e] = float(p)
    edges = tuple(sorted(canonical))
    values = tuple(terminal * (1-canonical[e]) for e in edges)
    if vertex_mode == 'fixed':
        births = (0.,) * n
    elif vertex_mode == 'incident':
        births = tuple(min((f for e, f in zip(edges, values) if v in e), default=terminal) for v in range(n))
    else:
        raise ValueError("vertex_mode must be 'fixed' or 'incident'.")
    return Graph(n, edges, values, births, terminal)


def target_graph(prediction, true_edges, *, vertex_mode='fixed'):
    truth = {edge(*e) for e in true_edges}
    if not truth <= set(prediction.edges):
        raise ValueError('Every target edge must be a candidate.')
    return from_probabilities(prediction.n, {e: float(e in truth) for e in prediction.edges},
                              terminal=prediction.terminal, vertex_mode=vertex_mode)


def minimum_union(p, g):
    if (p.n, p.edges, p.terminal) != (g.n, g.edges, g.terminal):
        raise ValueError('Shared ordered candidate complex and terminal required.')
    return Graph(p.n, p.edges, tuple(map(min, p.values, g.values)),
                 tuple(map(min, p.births, g.births)), p.terminal)


@dataclass(frozen=True)
class Pair:
    vertex: int
    birth: float
    death: float
    edge_index: int
    death_edge: Edge

    @property
    def lifetime(self):
        return self.death - self.birth


@dataclass(frozen=True)
class Event:
    time: float
    edge: Edge
    before: tuple
    after: tuple
    killed: int | None


@dataclass(frozen=True)
class Persistence:
    pairs: tuple[Pair, ...]
    zero_pairs: tuple[Pair, ...]
    essential: tuple[int, float]
    events: tuple[Event, ...]


def persistence(graph):
    uf = UnionFind(graph.births)
    positive, zero, events = [], [], []
    for i in graph.order:
        e, t = graph.edges[i], graph.values[i]
        before = uf.components()
        killed = uf.link(e)
        events.append(Event(t, e, before, uf.components(), killed))
        if killed is not None:
            pair = Pair(killed, graph.births[killed], t, i, e)
            (positive if pair.lifetime > 0 else zero).append(pair)
    root = uf.find(0)
    return Persistence(tuple(positive), tuple(zero), (root, graph.births[root]), tuple(events))


def components_at(graph, t):
    """Actual sublevel components, also respecting births in the legacy display."""
    uf = UnionFind(graph.births)
    active = {v for v, b in enumerate(graph.births) if b <= t}
    for e, value in zip(graph.edges, graph.values):
        if value <= t:
            uf.link(e)
    return tuple(tuple(v for v in c if v in active) for c in uf.components() if any(v in active for v in c))


@dataclass(frozen=True)
class Match:
    prediction: Pair
    target: Pair
    comparison: Pair


@dataclass(frozen=True)
class Matching:
    prediction: Persistence
    target: Persistence
    comparison: Persistence
    matches: tuple[Match, ...]
    unmatched_prediction: tuple[Pair, ...]
    unmatched_target: tuple[Pair, ...]
    trace: tuple[dict, ...]

    @property
    def count(self):
        return len(self.matches) + len(self.unmatched_prediction) + len(self.unmatched_target)


def induced_matching(p, g):
    """Extended H0 correspondence, with every image-pair lookup exposed in trace.

    In the second sweep all three UF instances traverse COMPARISON forest
    edges; their partitions are NOT the ordinary source filtrations at time t.
    Source birth orders differ only in the optional legacy illustration.
    """
    u = minimum_union(p, g)
    pp, gp, up = persistence(p), persistence(g), persistence(u)
    p_by_birth = {x.vertex: x for x in pp.pairs}
    g_by_birth = {x.vertex: x for x in gp.pairs}
    u_by_edge = {x.death_edge: x for x in up.pairs}
    pu, gu, uu = UnionFind(p.births), UnionFind(g.births), UnionFind(u.births)
    matches, trace = [], []
    for i in u.order:
        e = u.edges[i]
        before = uu.components()
        ru = uu.link(e)
        if ru is None:
            continue
        rp, rg = pu.link(e), gu.link(e)
        a, b, c = p_by_birth.get(rp), g_by_birth.get(rg), u_by_edge.get(e)
        matched = a is not None and b is not None and c is not None
        if matched:
            matches.append(Match(a, b, c))
        trace.append(dict(edge=e, time=u.values[i], before=before, after=uu.components(),
                          rp=rp, rg=rg, ru=ru, prediction=a, target=b,
                          comparison=c, matched=matched))
    used_p = {m.prediction.vertex for m in matches}
    used_g = {m.target.vertex for m in matches}
    return Matching(pp, gp, up, tuple(matches),
                    tuple(x for x in pp.pairs if x.vertex not in used_p),
                    tuple(x for x in gp.pairs if x.vertex not in used_g), tuple(trace))


def loss_terms(m, *, unmatched_weight=1.):
    return dict(matched=sum((a.prediction.birth-a.target.birth)**2 +
                            (a.prediction.death-a.target.death)**2 for a in m.matches),
                unmatched_prediction=unmatched_weight*.5*sum(a.lifetime**2 for a in m.unmatched_prediction),
                unmatched_target=unmatched_weight*.5*sum(a.lifetime**2 for a in m.unmatched_target))


def scalar_loss(m, *, normalize=True, unmatched_weight=1.):
    return sum(loss_terms(m, unmatched_weight=unmatched_weight).values()) / (max(1, m.count) if normalize else 1)


def differentiable_loss(probabilities, candidate_edges, true_edges, *, num_vertices,
                        terminal=1., normalize=True, unmatched_weight=1.):
    """PyTorch fixed-birth loss. Edge tensor shape [E,2], probability shape [E].

    No epsilon clamps, no straight-through sorting, no differentiable matching.
    Sorting reorders live probabilities before indexing their critical edges.
    Numerator and feature-count normalization exactly match scalar_loss.
    """
    import torch
    if probabilities.ndim != 1 or not probabilities.is_floating_point():
        raise ValueError('probabilities must be a floating vector.')
    if candidate_edges.ndim != 2 or candidate_edges.shape != (probabilities.numel(), 2):
        raise ValueError('candidate_edges must have shape [E,2].')
    if not isfinite(unmatched_weight) or unmatched_weight < 0:
        raise ValueError('unmatched_weight must be finite and nonnegative.')
    es = [edge(*e) for e in candidate_edges.detach().cpu().tolist()]
    if len(set(es)) != len(es):
        raise ValueError('Duplicate candidate edge.')
    truth = true_edges.detach().cpu().tolist() if torch.is_tensor(true_edges) else true_edges
    pred = from_probabilities(num_vertices, dict(zip(es, probabilities.detach().cpu().tolist())), terminal=terminal)
    gt = target_graph(pred, truth)
    m = induced_matching(pred, gt)
    original_index = {e: i for i, e in enumerate(es)}
    order = torch.tensor([original_index[e] for e in pred.edges], device=probabilities.device)
    live_deaths = terminal * (1-probabilities[order.long()])
    total = probabilities.sum() * 0
    for pair in m.matches:
        total = total + (live_deaths[pair.prediction.edge_index]-pair.target.death).square()
    for pair in m.unmatched_prediction:
        total = total + .5 * unmatched_weight * live_deaths[pair.edge_index].square()
    total = total + probabilities.new_tensor(.5 * unmatched_weight * sum(x.death**2 for x in m.unmatched_target))
    return total / (max(1, m.count) if normalize else 1), m


EXAMPLE = {(0,1): .90, (0,2): .10, (0,3): .02, (1,2): .20, (1,3): .05, (2,3): .80}
TRUTH = {(0,1), (2,3)}


def self_check():
    """Independent boundary-column reduction, concrete losses and real autograd."""
    import random
    import torch
    p = from_probabilities(4, EXAMPLE)
    m = induced_matching(p, target_graph(p, TRUTH))
    assert [a.death_edge for a in m.prediction.pairs] == [(0,1), (2,3), (1,2)]
    assert len(m.matches) == 1 and m.matches[0].prediction.vertex == 2
    assert abs(scalar_loss(m, normalize=False)-.065) < 1e-12
    assert abs(scalar_loss(m)-.065/3) < 1e-12
    assert scalar_loss(induced_matching(target_graph(p, TRUTH), target_graph(p, TRUTH))) == 0
    # Uniform low confidences: old H0=0; fixed H0>0. An isolated GT node survives to T.
    for mode in ('fixed', 'incident'):
        low = from_probabilities(3, {e:.1 for e in complete_edges(3)}, vertex_mode=mode)
        mm = induced_matching(low, target_graph(low, {(0,1),(1,2)}, vertex_mode=mode))
        assert abs(scalar_loss(mm) - (.405 if mode == 'fixed' else 0)) < 1e-12
    isolated = target_graph(p, {(0,1)})
    assert len(persistence(isolated).pairs) == 2
    # Independent GF(2) boundary-column reducer: same pairs, including zero pairs.
    rng = random.Random(19)
    checks = 0
    for n in range(2,9):
        for _ in range(30):
            graph = from_probabilities(n, {e:rng.random() for e in complete_edges(n)})
            pivots, expected = {}, []
            for i in graph.order:
                u,v = graph.edges[i]
                col = (1 << u) ^ (1 << v)
                while col and col.bit_length()-1 in pivots:
                    col ^= pivots[col.bit_length()-1]
                if col:
                    root = col.bit_length()-1
                    pivots[root] = col
                    expected.append((root, graph.edges[i]))
            assert [(a.vertex,a.death_edge) for a in persistence(graph).pairs] == expected
            truth = {e for e in graph.edges if rng.random() < .3}
            mm = induced_matching(graph, target_graph(graph, truth))
            assert len({x.prediction.vertex for x in mm.matches}) == len(mm.matches)
            assert len({x.target.vertex for x in mm.matches}) == len(mm.matches)
            checks += 1
    es = torch.tensor(p.edges, dtype=torch.long)
    x = torch.tensor(list(EXAMPLE.values()), dtype=torch.float64, requires_grad=True)
    objective = lambda q: differentiable_loss(q, es, TRUTH, num_vertices=4)[0]
    assert torch.autograd.gradcheck(objective, (x,), eps=1e-6, atol=1e-6)
    value = objective(x)
    grad, = torch.autograd.grad(value, x)
    assert torch.allclose(grad, torch.tensor([-.1/3,0,0,.4/3,0,-.2/3], dtype=x.dtype))
    perm = torch.tensor([5,3,1,4,0,2])
    xp = x.detach()[perm].requires_grad_()
    lp, _ = differentiable_loss(xp, es[perm][:,[1,0]], TRUTH, num_vertices=4)
    gp, = torch.autograd.grad(lp, xp)
    assert torch.allclose(lp, value) and torch.allclose(gp, grad[perm])
    # Two-class relation head, both endpoint orders; topology alone must reach head/tokens.
    torch.manual_seed(8)
    tokens = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    head = torch.nn.Linear(6,2,dtype=torch.float64)
    q = .5*(head(torch.cat((tokens[es[:,0]],tokens[es[:,1]]),dim=1)).softmax(-1)[:,1] +
            head(torch.cat((tokens[es[:,1]],tokens[es[:,0]]),dim=1)).softmax(-1)[:,1])
    differentiable_loss(q, es, TRUTH, num_vertices=4)[0].backward()
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.norm() > 0
    assert torch.isfinite(tokens.grad).all() and tokens.grad.norm() > 0
    return dict(boundary_reduction_cases=checks, example_loss=float(value.detach()),
                gradcheck='passed', edge_permutation='passed', isolated_node='passed',
                topology_only_head_backward='passed')


if __name__ == '__main__':
    print(self_check())
