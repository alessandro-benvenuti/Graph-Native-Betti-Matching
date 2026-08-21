"""Pure helpers for focal classification and unmatched relation candidates."""

import torch
import torch.nn.functional as F


def softmax_focal_loss(
    logits,
    targets,
    class_weights=None,
    gamma=2.0,
    reduction="mean",
):
    """Multi-class focal loss for logits with arbitrary leading dimensions."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "targets must match all non-class logits dimensions: "
            f"{targets.shape} vs {logits.shape}"
        )
    if gamma < 0:
        raise ValueError("focal gamma must be non-negative")

    log_prob = F.log_softmax(logits, dim=-1)
    target_log_prob = log_prob.gather(
        -1, targets.long().unsqueeze(-1)
    ).squeeze(-1)
    target_prob = target_log_prob.exp()
    # During a gamma curriculum, 0 < gamma < 1.  If a correct prediction has
    # saturated to target_prob == 1, pow(0, gamma) has an infinite derivative
    # and produces NaN gradients even though the scalar loss is zero.  Clamp
    # the modulating base away from zero; the clamp's flat saturated branch
    # preserves a zero loss/gradient for an exactly confident prediction.
    modulation_base = (1.0 - target_prob).clamp_min(
        torch.finfo(logits.dtype).eps
    )
    loss = -modulation_base.pow(float(gamma)) * target_log_prob

    alpha_t = None
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        if (
            class_weights.ndim != 1
            or class_weights.numel() != logits.shape[-1]
        ):
            raise ValueError(
                "class_weights must contain one value per logit class"
            )
        alpha_t = class_weights[targets.long()]
        loss = loss * alpha_t

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(f"unsupported reduction: {reduction}")
    if loss.numel() == 0:
        return logits.sum() * 0.0
    if alpha_t is None:
        return loss.mean()
    # Matches torch weighted cross-entropy at gamma=0.
    return loss.sum() / alpha_t.sum().clamp_min(
        torch.finfo(logits.dtype).eps
    )


@torch.no_grad()
def select_active_unmatched_queries(
    pred_logits,
    matched_src,
    object_threshold=0.5,
    max_active_unmatched=16,
):
    """Return matched/unmatched masks and capped active unmatched indices."""
    if pred_logits.ndim != 2 or pred_logits.shape[-1] != 2:
        raise ValueError("pred_logits must have shape [queries, 2]")
    if max_active_unmatched < 0:
        raise ValueError("max_active_unmatched must be non-negative")

    device = pred_logits.device
    matched_src = matched_src.to(device=device, dtype=torch.long)
    matched_mask = torch.zeros(
        pred_logits.shape[0], dtype=torch.bool, device=device
    )
    matched_mask[matched_src] = True
    unmatched_mask = ~matched_mask
    object_probability = pred_logits.softmax(-1)[:, 1].detach()
    active = torch.nonzero(
        unmatched_mask & (object_probability >= float(object_threshold)),
        as_tuple=False,
    ).flatten()
    if active.numel() > max_active_unmatched:
        keep = object_probability[active].topk(
            max_active_unmatched, largest=True, sorted=True
        ).indices
        active = active[keep]
    return matched_mask, unmatched_mask, active, object_probability


def build_unmatched_relation_pairs(matched_src, active_unmatched):
    """Build each UxM and U-combination pair once in original query space."""
    device = active_unmatched.device
    matched_src = matched_src.to(device=device, dtype=torch.long)
    active_unmatched = active_unmatched.to(device=device, dtype=torch.long)
    parts = []
    if active_unmatched.numel() and matched_src.numel():
        u = active_unmatched[:, None].expand(-1, matched_src.numel())
        m = matched_src[None, :].expand(active_unmatched.numel(), -1)
        parts.append(
            torch.stack((u.reshape(-1), m.reshape(-1)), dim=1)
        )
    if active_unmatched.numel() >= 2:
        parts.append(torch.combinations(active_unmatched, r=2))
    if not parts:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)


def select_hard_unmatched_relation_logits(
    relation_logits,
    max_pairs=0,
):
    """Keep the highest-P(edge) unmatched candidates without detaching loss.

    Selection is discrete and therefore uses detached probabilities.  The
    returned logits are indexed from the original tensor, so focal/CE loss on
    them still backpropagates through the relation head and object tokens.
    ``max_pairs=0`` preserves the legacy uncapped behavior.
    """
    if relation_logits.ndim != 2 or relation_logits.shape[-1] != 2:
        raise ValueError("relation_logits must have shape [pairs, 2]")
    max_pairs = int(max_pairs)
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")

    pair_count = int(relation_logits.shape[0])
    if max_pairs == 0 or pair_count <= max_pairs:
        indices = torch.arange(
            pair_count,
            dtype=torch.long,
            device=relation_logits.device,
        )
        return relation_logits, indices

    with torch.no_grad():
        edge_probability = relation_logits.softmax(-1)[:, 1]
        indices = edge_probability.topk(
            max_pairs,
            largest=True,
            sorted=True,
        ).indices
    return relation_logits[indices], indices


def scheduled_candidate_weight(epoch, target, warmup, ramp):
    """Warm up at zero, then optionally ramp to a per-candidate weight."""
    target = max(0.0, float(target))
    epoch = max(1, int(epoch))
    warmup = max(0, int(warmup))
    ramp = max(0, int(ramp))
    if epoch <= warmup:
        return 0.0
    if ramp == 0:
        return target
    progress = min(1.0, (epoch - warmup) / float(ramp))
    return target * progress


def linear_progress_schedule(
    progress_pct,
    target,
    start_pct,
    end_pct,
    start_value=0.0,
):
    """Linearly interpolate a value over a percentage of training progress.

    Percentages are deliberately expressed on a 0--100 scale so command-line
    values such as ``10`` and ``30`` are unambiguous.  Equal start/end
    percentages define a step schedule.
    """
    progress_pct = float(progress_pct)
    target = float(target)
    start_value = float(start_value)
    start_pct = float(start_pct)
    end_pct = float(end_pct)
    if not 0.0 <= start_pct <= end_pct <= 100.0:
        raise ValueError(
            "curriculum percentages must satisfy "
            "0 <= start_pct <= end_pct <= 100"
        )
    if progress_pct <= start_pct:
        return start_value
    if progress_pct >= end_pct or start_pct == end_pct:
        return target
    fraction = (progress_pct - start_pct) / (end_pct - start_pct)
    return start_value + fraction * (target - start_value)
