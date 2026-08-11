"""Supported model construction API."""


def build_model(config):
    """Build the only supported architecture: 3D RelationFormer."""

    from .relationformer import build_relationformer

    return build_relationformer(config)


__all__ = ["build_model"]
