from __future__ import annotations
import torch
from torch import Tensor, Generator
from typing import Callable


def generate_binary_context(size: int, seed: int | None = None) -> Tensor:
    """Binary context in {-1, +1}^size."""

    if seed is not None:
        g = Generator()
        g.manual_seed(seed)
        rand = torch.rand(size, generator=g)
    else:
        rand = torch.rand(size)

    return torch.where(rand < 0.5, -torch.ones_like(rand), torch.ones_like(rand))


def generate_complex_context(size: int, seed: int | None = None) -> Tensor:
    """Complex unit-phase vector e^{iθ} represented as 2D real tensor (cos, sin)."""
    if seed is not None:
        g = Generator()
        g.manual_seed(seed)
        theta = 2 * torch.pi * torch.rand(size, generator=g)
    else:
        theta = 2 * torch.pi * torch.rand(size)

    # Return a *real* representation: shape (size, 2) or flattened (2*size)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def generate_rotation_context(size: int, seed: int | None = None) -> Tensor:
    """Random orthogonal context vector (rotation embedding)."""
    if seed is not None:
        g = Generator()
        g.manual_seed(seed)
        vec = torch.randn(size, generator=g)
    else:
        vec = torch.randn(size)

    # Normalize to unit length
    return vec / vec.norm(p=2)


# -------------------------------------------------
# Factory: return a function that generates contexts
# -------------------------------------------------

def get_context_generator(context_type: str) -> Callable[[int, int | None], Tensor]:
    context_type = context_type.lower()

    if context_type == "binary":
        return generate_binary_context
    elif context_type == "complex":
        return generate_complex_context
    elif context_type == "rotation":
        return generate_rotation_context
    else:
        raise ValueError(
            f"Unknown context_type '{context_type}'. "
            f"Available: 'binary', 'complex', 'rotation'."
        )
