from __future__ import annotations

import argparse
import itertools
from typing import List, Tuple

from hands_on_neuroai.data.mnist import MNISTConfig
from hands_on_neuroai.training.perm_mnist import PermMNISTExperimentConfig


def parse_perm_mnist_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run permuting-MNIST experiments (single run or sweeps) "
            "over hidden_dim, context_type, num_tasks, steps_per_task."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, nargs="+", required=True)
    parser.add_argument("--context-type", type=str, nargs="+", required=True)
    parser.add_argument("--num-tasks", type=int, nargs="+", required=True)
    parser.add_argument("--steps-per-task", type=int, nargs="+", required=True)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/perm_mnist")

    return parser.parse_args()


def build_perm_mnist_experiment_configs(
    args: argparse.Namespace,
) -> Tuple[List[PermMNISTExperimentConfig], MNISTConfig]:
    hidden_dims: List[int] = args.hidden_dim
    context_types: List[str] = [ct.lower() for ct in args.context_type]
    num_tasks_list: List[int] = args.num_tasks
    steps_per_task_list: List[int] = args.steps_per_task

    mnist_cfg = MNISTConfig(root=args.data_root)

    configs: List[PermMNISTExperimentConfig] = []
    for hidden_dim, context_type, num_tasks, steps_per_task in itertools.product(
        hidden_dims,
        context_types,
        num_tasks_list,
        steps_per_task_list,
    ):
        configs.append(
            PermMNISTExperimentConfig(
                mnist=mnist_cfg,
                hidden_dim=hidden_dim,
                context_type=context_type,
                num_tasks=num_tasks,
                steps_per_task=steps_per_task,
                batch_size=args.batch_size,
                eval_interval=args.eval_interval,
                lr=args.lr,
                base_seed=args.seed,
            ),
        )

    return configs, mnist_cfg
