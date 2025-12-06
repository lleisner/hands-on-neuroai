from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hands_on_neuroai.dataset_creation.clutter import (
    ClutteredDatasetConfig,
    build_cluttered_composite_dataset,
    build_cluttered_dataloader,
)
from hands_on_neuroai.models.factory import build_rcnn_for_clutter
from hands_on_neuroai.training.metrics import evaluate_accuracy


@dataclass
class ClutterRCNNExperimentConfig:
    """Experiment configuration for the cluttered dataset RCNN training."""

    dataset_config: ClutteredDatasetConfig
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    eval_interval: int = 1  # evaluate every N epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    timesteps: int = 4
    interaction: str = "multiplicative"  # or "additive"


def train_rcnn_on_clutter(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    eval_interval: int = 1,
    verbose: int = 1,
) -> Tuple[list[int], list[float]]:
    """Simple training loop for the cluttered RCNN."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logged_epochs: list[int] = []
    logged_accs: list[float] = []

    for epoch in range(1, epochs + 1):
        if verbose >= 1:
            step_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        else:
            step_iter = train_loader

        model.train()
        for imgs, labels in step_iter:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits: Tensor = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        if eval_interval and (epoch % eval_interval == 0):
            acc = evaluate_accuracy(model, val_loader, device)
            logged_epochs.append(epoch)
            logged_accs.append(acc)
            if verbose >= 1:
                print(f"[Epoch {epoch}] val acc={acc:.4f}")

    return logged_epochs, logged_accs


def run_clutter_rcnn_experiment(
    cfg: ClutterRCNNExperimentConfig,
    val_split: float = 0.1,
    verbose: int = 1,
) -> Tuple[list[int], list[float]]:
    """
    Build datasets/loaders, create the RCNN, and train/evaluate it.

    Args:
        cfg: Experiment configuration.
        val_split: Fraction of train set to hold out for validation.
        verbose: Verbosity level.
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.dataset_config.seed)

    # Build datasets
    full_train = build_cluttered_composite_dataset(cfg.dataset_config)
    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.dataset_config.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataset_config.num_workers if hasattr(cfg.dataset_config, "num_workers") else 4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataset_config.num_workers if hasattr(cfg.dataset_config, "num_workers") else 4,
        pin_memory=True,
    )

    # Build model (infer channels/classes/image_size from dataset config)
    model = build_rcnn_for_clutter(
        clutter_cfg=cfg.dataset_config,
        device=device,
        timesteps=cfg.timesteps,
        interaction=cfg.interaction,
    )

    epochs, accs = train_rcnn_on_clutter(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
        eval_interval=cfg.eval_interval,
        verbose=verbose,
    )

    return epochs, accs
