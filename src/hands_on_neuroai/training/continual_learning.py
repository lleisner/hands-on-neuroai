"""General-purpose continual learning training loop.

Works with any dataset, task transform, and model configuration.
Supports arbitrary combinations of:
  - Datasets: MNIST, CIFAR-10, CIFAR-100, FashionMNIST, etc.
  - Task transforms: Permutation, Rotation, Class-Incremental, etc.
  - Models: Baseline MLPs, PSP with various contexts, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hands_on_neuroai.dataset_creation.datasets import DatasetConfig
from hands_on_neuroai.models.factory import build_model_for_continual_learning
from hands_on_neuroai.training.metrics import evaluate_accuracy


@dataclass
class ContinualLearningExperimentConfig:
    """Configuration for a general continual learning experiment run.
    
    This config works with any dataset, task transform, and model combination.
    """

    # Dataset configuration
    dataset_config: DatasetConfig
    dataset_name: str  # "mnist", "cifar10", "cifar100", "fashionmnist", etc.

    # Task configuration
    num_tasks: int
    steps_per_task: int

    # Model configuration
    hidden_dim: int
    context_type: str  # "none", "binary", "complex", "rotation"

    # Training configuration
    batch_size: int = 128
    eval_interval: int = 100
    lr: float = 1e-3
    base_seed: int = 0


def train_model_on_continual_learning_tasks(
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
    Core training loop for continual learning with arbitrary task distributions.

    Works with any model (baseline or task-aware) and any dataset/transform combination.

    Args:
        model: Trained model (baseline MLP or PSP).
        train_loaders: List of training DataLoaders, one per task.
        test_loaders: List of test DataLoaders, one per task.
        num_tasks: Number of tasks.
        steps_per_task: Training steps per task.
        lr: Learning rate.
        eval_interval: Evaluation interval (in steps).
        device: Device to run on.
        verbose: Verbosity level:
            0 = silent
            1 = task-level progress bar only
            2 = task + step-level bars
            3 = task + step-level bars + loss prints

    Returns:
        (logged_steps, logged_accs): Lists of global steps and accuracies on task 0.
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


def run_continual_learning_experiment(
    cfg: ContinualLearningExperimentConfig,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    verbose: int = 1,
) -> Tuple[List[int], List[float]]:
    """Run a single continual learning experiment.
    
    Args:
        cfg: Experiment configuration.
        train_loaders: Pre-built training DataLoaders (one per task).
        test_loaders: Pre-built test DataLoaders (one per task).
        device: Device to train on.
        verbose: Verbosity level.
    
    Returns:
        (logged_steps, logged_accs): Training metrics.
    """
    torch.manual_seed(cfg.base_seed)

    # Get dataset-specific dimensions
    from hands_on_neuroai.data.datasets import get_image_shape, get_num_classes
    input_dim, _ = get_image_shape(cfg.dataset_name)
    output_dim = get_num_classes(cfg.dataset_name)

    model = build_model_for_continual_learning(
        context_type=cfg.context_type,
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim,
        num_tasks=cfg.num_tasks,
        base_seed=cfg.base_seed,
        device=device,
    )

    steps, accs = train_model_on_continual_learning_tasks(
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
