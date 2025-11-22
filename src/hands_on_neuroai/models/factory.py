from __future__ import annotations

import torch
from torch import device as torch_device

from hands_on_neuroai.models.mlp import MLP2Layer
from hands_on_neuroai.models.context import get_context_generator
from hands_on_neuroai.models.psp import PSPMLP2Layer


def build_model_for_perm_mnist(
    context_type: str,
    hidden_dim: int,
    num_tasks: int,
    base_seed: int,
    device: torch_device,
) -> torch.nn.Module:
    """Factory for models used in permuting-MNIST experiments.

    context_type:
        - "none"      -> baseline MLP2Layer
        - "binary"    -> PSP with binary contexts
        - "complex"   -> PSP with complex contexts
        - "rotation"  -> PSP with rotational contexts
    """
    context_type = context_type.lower()

    if context_type == "none":
        model: torch.nn.Module = MLP2Layer(hidden_dim=hidden_dim)
        return model.to(device)

    # PSP variants
    context_fn = get_context_generator(context_type)
    model = PSPMLP2Layer(
        hidden_dim=hidden_dim,
        num_tasks=num_tasks,
        context_fn=context_fn,
        seed=base_seed,
    )
    return model.to(device)
