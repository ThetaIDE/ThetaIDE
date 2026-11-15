"""Bounded, explicit sequential topology adapter on isolated scoring forwards."""
from __future__ import annotations

from torch import nn


POINTWISE = (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU, nn.LeakyReLU,
             nn.ELU, nn.Identity, nn.Dropout)


def sequential_modules(access, target):
    """Override ModelAccess.topology_modules for architecture-owned adapters.

    The default supports an explicitly selected Sequential, not arbitrary graphs
    inferred from module registration order. No edges are guessed across branches.
    """
    module = access.target(target)
    if not isinstance(module, nn.Sequential):
        raise ValueError("Network viewer requires a Sequential target or an architecture-owned topology_modules adapter")
    result = []

    def visit(parent, prefix):
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Sequential):
                visit(child, path)
            elif isinstance(child, (nn.Linear, nn.LayerNorm, *POINTWISE)):
                result.append((path, child))
            else:
                raise ValueError(f"Unsupported topology operation {path}: {type(child).__name__}; supply an architecture adapter")
    visit(module, "")
    if not result or not any(isinstance(m, nn.Linear) for _, m in result):
        raise ValueError("Network topology needs at least one Linear operation")
    return result
