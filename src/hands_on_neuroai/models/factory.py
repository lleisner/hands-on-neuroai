from __future__ import annotations

import torch
from torch import device as torch_device

from hands_on_neuroai.models.mlp import MLP2Layer
from hands_on_neuroai.models.context import get_context_generator
from hands_on_neuroai.models.psp import PSPMLP2Layer


def build_model_for_continual_learning(
    context_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_tasks: int,
    base_seed: int,
    device: torch_device,
) -> torch.nn.Module:
    """Factory for models used in continual learning experiments.

    Works with any dataset and any task transform (permutation, rotation, class splits, etc).

    Args:
        context_type: Type of context mechanism to use:
            - "none"      -> baseline MLP2Layer (no task awareness)
            - "binary"    -> PSP with binary contexts
            - "complex"   -> PSP with complex contexts
            - "rotation"  -> PSP with rotational contexts
        input_dim: Input feature dimension (e.g., 784 for MNIST, 3072 for CIFAR-10)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (number of classes, e.g., 10)
        num_tasks: Number of tasks
        base_seed: Random seed for context generation
        device: Device to place model on (cpu or cuda)

    Returns:
        A torch.nn.Module ready for training on continual learning tasks.
    """
    context_type = context_type.lower()

    if context_type == "none":
        model: torch.nn.Module = MLP2Layer(input_dim, hidden_dim, output_dim)
        return model.to(device)

    # PSP variants (task-aware)
    context_fn = get_context_generator(context_type)
    model = PSPMLP2Layer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_tasks=num_tasks,
        context_fn=context_fn,
        seed=base_seed,
    )
    return model.to(device)


# Backwards compatibility alias
def build_model_for_perm_mnist(
    context_type: str = "binary",
    hidden_dim: int = 256,
    num_tasks: int = 10,
    base_seed: int = 42,
    device: torch_device = "cpu",
) -> torch.nn.Module:
    """Backwards compatibility alias for MNIST permuted task experiments.

    Convenience wrapper that calls build_model_for_continual_learning() with
    MNIST-specific input_dim=784 and output_dim=10.

    Args:
        context_type: Type of context mechanism (see build_model_for_continual_learning)
        hidden_dim: Hidden layer dimension
        num_tasks: Number of tasks
        base_seed: Random seed for context generation
        device: Device to place model on (cpu or cuda)

    Returns:
        A torch.nn.Module configured for permuted MNIST tasks.
    """
    return build_model_for_continual_learning(
        context_type=context_type,
        input_dim=784,  # MNIST: 28x28 = 784
        hidden_dim=hidden_dim,
        output_dim=10,  # 10 digits
        num_tasks=num_tasks,
        base_seed=base_seed,
        device=device,
    )
