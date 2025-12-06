from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import torch
from torch import device as torch_device

from hands_on_neuroai.dataset_creation.base import DATASET_METADATA
from hands_on_neuroai.dataset_creation.clutter import (
    ClutteredCompositeDataset,
    ClutteredDatasetConfig,
)
from hands_on_neuroai.models.mlp import MLP2Layer
from hands_on_neuroai.models.context import get_context_generator
from hands_on_neuroai.models.psp import PSPMLP2Layer
from hands_on_neuroai.models.rcnn import RCNN, RCNNConfig


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


# --- RCNN for cluttered dataset experiments ---


def _infer_num_classes_from_bases(base_names: Sequence[str]) -> int:
    """Infer total classes by summing metadata of base datasets (no domain tags)."""
    total = 0
    for name in base_names:
        name = name.lower()
        if name not in DATASET_METADATA:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_METADATA.keys())}")
        total += DATASET_METADATA[name][3]
    return total


def build_rcnn_for_clutter(
    clutter_cfg: ClutteredDatasetConfig,
    device: torch_device,
    **rcnn_overrides: Any,
) -> RCNN:
    """
    Build an RCNN with config inferred from a ClutteredDatasetConfig.

    Args:
        clutter_cfg: Dataset config (provides image_size, base datasets, etc.).
        device: torch device.
        rcnn_overrides: Optional overrides for RCNNConfig fields (e.g., timesteps, interaction).

    Returns:
        RCNN on the requested device.
    """
    # Normalize base names
    base_names = (
        (clutter_cfg.base_datasets,) if isinstance(clutter_cfg.base_datasets, str) else clutter_cfg.base_datasets
    )

    # Infer in_channels by peeking at a tiny dataset instance
    probe_cfg = replace(clutter_cfg, num_samples=1, return_aux_labels=False)
    probe_ds = ClutteredCompositeDataset(probe_cfg)
    in_channels = rcnn_overrides.pop("in_channels", probe_ds.out_channels)

    # Infer num_classes by summing base dataset class counts (assumes disjoint labels)
    num_classes = rcnn_overrides.pop("num_classes", _infer_num_classes_from_bases(base_names))

    image_size = rcnn_overrides.pop("image_size", clutter_cfg.image_size)

    rcnn_cfg = RCNNConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        image_size=image_size,
        **rcnn_overrides,
    )
    return RCNN(rcnn_cfg).to(device)
