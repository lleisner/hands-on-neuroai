from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy for a model on a given dataloader."""
    model.eval()
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits: Tensor = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)
