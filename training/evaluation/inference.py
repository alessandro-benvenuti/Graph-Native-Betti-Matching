"""Convert RelationFormer tokens into explicit undirected graphs."""

from __future__ import annotations

import torch


@torch.no_grad()
def infer_graphs(
    tokens,
    predictions,
    relation_embed,
    *,
    object_queries,
    relation_tokens,
    node_threshold=None,
    edge_threshold=None,
):
    object_features = tokens[..., :object_queries, :]
    shared_relations = tokens[
        ..., object_queries : object_queries + relation_tokens, :
    ]
    node_probabilities = predictions["pred_logits"].softmax(-1)[..., 1]
    if node_threshold is None:
        valid = predictions["pred_logits"].argmax(-1) == 1
    else:
        valid = node_probabilities > float(node_threshold)

    graphs = []
    for batch in range(tokens.shape[0]):
        query_ids = torch.nonzero(valid[batch], as_tuple=False).flatten()
        nodes = predictions["pred_nodes"][batch, query_ids, :3]
        scores = node_probabilities[batch, query_ids]
        if query_ids.numel() < 2:
            pairs = torch.empty((0, 2), dtype=torch.long, device=tokens.device)
            relation_scores = scores.new_empty((0,))
        else:
            local_pairs = torch.combinations(
                torch.arange(query_ids.numel(), device=tokens.device), r=2
            )
            query_pairs = query_ids[local_pairs]
            left = object_features[batch, query_pairs[:, 0]]
            right = object_features[batch, query_pairs[:, 1]]
            if relation_tokens:
                relation = shared_relations[batch].reshape(1, -1).expand(
                    local_pairs.shape[0], -1
                )
                forward = torch.cat((left, right, relation), dim=-1)
                reverse = torch.cat((right, left, relation), dim=-1)
            else:
                forward = torch.cat((left, right), dim=-1)
                reverse = torch.cat((right, left), dim=-1)
            relation_logits = 0.5 * (
                relation_embed(forward) + relation_embed(reverse)
            )
            probabilities = relation_logits.softmax(-1)[:, 1]
            if edge_threshold is None:
                keep = relation_logits.argmax(-1) == 1
            else:
                keep = probabilities > float(edge_threshold)
            pairs = local_pairs[keep]
            relation_scores = probabilities[keep]
        graphs.append(
            {
                "nodes": nodes.detach().cpu(),
                "node_scores": scores.detach().cpu(),
                "edges": pairs.detach().cpu(),
                "edge_scores": relation_scores.detach().cpu(),
                "query_ids": query_ids.detach().cpu(),
            }
        )
    return graphs


__all__ = ["infer_graphs"]
