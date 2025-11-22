from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from hands_on_neuroai.data.mnist import MNISTConfig, build_permuted_mnist_loaders
from hands_on_neuroai.models.factory import build_model_for_perm_mnist
from hands_on_neuroai.training.metrics import evaluate_accuracy


@dataclass
class PermMNISTExperimentConfig:
    """Configuration for a single permuting-MNIST experiment run."""

    mnist: MNISTConfig

    # core scientific knobs
    hidden_dim: int
    context_type: str  # "none", "binary", "complex", "rotation"
    num_tasks: int
    steps_per_task: int

    # other training knobs
    batch_size: int = 128
    eval_interval: int = 100
    lr: float = 1e-3
    base_seed: int = 0  # for contexts / permutations etc.


def run_perm_mnist_experiment(
    cfg: PermMNISTExperimentConfig,
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    """Run a single permuting-MNIST experiment.

    Returns:
        steps: list of global step indices where we evaluated
        accs:  list of accuracies on task 0 at those steps
    """
    torch.manual_seed(cfg.base_seed)

    train_loaders, test_loaders = build_permuted_mnist_loaders(
        config=cfg.mnist,
        num_tasks=cfg.num_tasks,
    )

    # override batch size, if it differs from mnist config
    if cfg.batch_size != cfg.mnist.batch_size:
        for i, loader in enumerate(train_loaders):
            train_loaders[i] = DataLoader(
                loader.dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.mnist.num_workers,
                pin_memory=cfg.mnist.pin_memory,
            )

    model = build_model_for_perm_mnist(
        context_type=cfg.context_type,
        hidden_dim=cfg.hidden_dim,
        num_tasks=cfg.num_tasks,
        base_seed=cfg.base_seed,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    total_steps = cfg.steps_per_task * cfg.num_tasks
    global_step = 0

    test_loader_task0 = test_loaders[0]

    logged_steps: List[int] = []
    logged_accs: List[float] = []

    has_task_context = hasattr(model, "set_task")

    for task_id in range(cfg.num_tasks):
        train_loader = train_loaders[task_id]

        if has_task_context:
            # type: ignore[attr-defined]
            model.set_task(task_id)

        data_iter = iter(train_loader)
        steps_in_task = 0

        while steps_in_task < cfg.steps_per_task:
            try:
                imgs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                imgs, labels = next(data_iter)

            imgs, labels = imgs.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            logits: Tensor = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            steps_in_task += 1

            if global_step % cfg.eval_interval == 0:
                if has_task_context:
                    # type: ignore[attr-defined]
                    model.set_task(0)
                acc = evaluate_accuracy(model, test_loader_task0, device)
                if has_task_context:
                    # type: ignore[attr-defined]
                    model.set_task(task_id)

                logged_steps.append(global_step)
                logged_accs.append(acc)

            if global_step >= total_steps:
                break

    return logged_steps, logged_accs
