from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np

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
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits: Tensor = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(1, total)


def collect_hidden_activations(
    model: torch.nn.Module,
    hidden_module: torch.nn.Module,
    loaders: List[DataLoader],
    num_tasks: int,
    num_samples_per_task: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect hidden activations from `hidden_module` for each task.

    Args:
        model: The model to run (baseline or PSP). If it has `set_task`,
               this will be called with the current task id.
        hidden_module: The specific nn.Module whose output we want to record
                       (e.g. model.fc1).
        loaders: List of DataLoaders, one per task.
        num_tasks: Number of tasks to collect from (usually len(loaders)).
        num_samples_per_task: How many samples to collect per task (approx).
        device: torch.device to run the model on.

    Returns:
        acts:      np.ndarray of shape [N_total, hidden_dim]
        task_ids:  np.ndarray of shape [N_total], ints in [0, num_tasks-1]
    """
    model.eval()

    hidden_collector: List[torch.Tensor] = []

    def hook_fn(module, inp, out):
        # out: Tensor of shape [batch_size, hidden_dim, ...]
        # we keep it as is, flatten later if needed
        hidden_collector.append(out.detach().cpu())

    handle = hidden_module.register_forward_hook(hook_fn)

    all_acts: List[torch.Tensor] = []
    all_task_ids: List[int] = []

    with torch.no_grad():
        for task_id in range(num_tasks):
            # If this is a PSP model, switch context for this task
            if hasattr(model, "set_task"):
                model.set_task(task_id)  # type: ignore[attr-defined]

            collected = 0
            for imgs, labels in loaders[task_id]:
                imgs = imgs.to(device, non_blocking=True)

                # clear any stale activations from previous batch
                hidden_collector.clear()

                _ = model(imgs)  # triggers hook

                if not hidden_collector:
                    raise RuntimeError("Hook did not collect any activations.")
                batch_acts = hidden_collector[-1]  # last captured
                # flatten if needed: [B, D, ...] -> [B, D']
                batch_acts = batch_acts.view(batch_acts.size(0), -1)

                all_acts.append(batch_acts)
                all_task_ids.extend([task_id] * batch_acts.size(0))

                collected += batch_acts.size(0)
                if collected >= num_samples_per_task:
                    break

    handle.remove()

    acts_tensor = torch.cat(all_acts, dim=0)  # [N_total, hidden_dim]
    acts = acts_tensor.numpy()
    task_ids = np.array(all_task_ids, dtype=np.int64)

    return acts, task_ids
