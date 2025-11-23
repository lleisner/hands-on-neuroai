from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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



def train_model_on_perm_mnist(
    model: torch.nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_tasks: int,
    steps_per_task: int,
    lr: float,
    eval_interval: int,
    device: torch.device,
    verbose: int = 1,
) -> Tuple[List[int], List[float]]:
    """
    Core training loop for permuted-MNIST with well-defined verbosity levels.

    verbose:
        0 = silent (no bars, no prints)
        1 = task-level bar only
        2 = task-level + step-level bars, no prints
        3 = task-level + step-level bars + periodic loss prints
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0

    test_loader_task0 = test_loaders[0]
    has_task_context = hasattr(model, "set_task")

    logged_steps: List[int] = []
    logged_accs: List[float] = []

    # Outer loop over tasks
    if verbose >= 1:
        task_iter = tqdm(range(num_tasks), desc="Tasks", leave=True)
    else:
        task_iter = range(num_tasks)

    for task_id in task_iter:

        if has_task_context:
            model.set_task(task_id)

        train_loader = train_loaders[task_id]
        data_iter = iter(train_loader)

        # Step-level iteration
        if verbose >= 2:
            step_iter = tqdm(range(steps_per_task), desc=f"Task {task_id}", leave=False)
        else:
            step_iter = range(steps_per_task)

        for step_in_task in step_iter:

            try:
                imgs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                imgs, labels = next(data_iter)

            imgs, labels = imgs.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1

            # Eval during training
            if global_step % eval_interval == 0:
                if has_task_context:
                    model.set_task(0)
                acc = evaluate_accuracy(model, test_loader_task0, device)
                if has_task_context:
                    model.set_task(task_id)

                logged_steps.append(global_step)
                logged_accs.append(acc)

                # Only verbose=3 shows losses
                if verbose == 3:
                    print(
                        f"[Task {task_id} | Step {step_in_task+1}/{steps_per_task} | "
                        f"global step {global_step}] "
                        f"loss={loss.item():.4f}, acc(task0)={acc:.4f}"
                    )

    if verbose >= 1:
        print("Training completed.")

    return logged_steps, logged_accs




def run_perm_mnist_experiment(
    cfg: PermMNISTExperimentConfig,
    device: torch.device,
    verbose: int = 1,
) -> Tuple[List[int], List[float]]:
    """Run a single permuted-MNIST experiment."""
    torch.manual_seed(cfg.base_seed)

    train_loaders, test_loaders = build_permuted_mnist_loaders(
        config=cfg.mnist,
        num_tasks=cfg.num_tasks,
        batch_size=cfg.batch_size,
        base_seed=cfg.base_seed,
    )

    model = build_model_for_perm_mnist(
        context_type=cfg.context_type,
        hidden_dim=cfg.hidden_dim,
        num_tasks=cfg.num_tasks,
        base_seed=cfg.base_seed,
        device=device,
    )

    steps, accs = train_model_on_perm_mnist(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=cfg.num_tasks,
        steps_per_task=cfg.steps_per_task,
        lr=cfg.lr,
        eval_interval=cfg.eval_interval,
        device=device,
        verbose=verbose,
    )

    return steps, accs