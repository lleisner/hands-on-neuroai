from __future__ import annotations

import json
from pathlib import Path
from typing import List

from hands_on_neuroai.data.mnist import MNISTConfig
from hands_on_neuroai.training.perm_mnist import PermMNISTExperimentConfig


def get_perm_mnist_result_dir(
    base_dir: Path,
    cfg: PermMNISTExperimentConfig,
) -> Path:
    return (
        base_dir
        / f"context_{cfg.context_type}"
        / f"hidden_{cfg.hidden_dim}"
        / f"tasks_{cfg.num_tasks}"
        / f"steps_{cfg.steps_per_task}"
    )


def save_perm_mnist_results(
    base_dir: Path,
    cfg: PermMNISTExperimentConfig,
    mnist_cfg: MNISTConfig,
    steps: List[int],
    accs: List[float],
) -> Path:
    exp_dir = get_perm_mnist_result_dir(base_dir, cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)

    out_path = exp_dir / f"seed_{cfg.base_seed}.json"

    payload = {
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "context_type": cfg.context_type,
            "num_tasks": cfg.num_tasks,
            "steps_per_task": cfg.steps_per_task,
            "batch_size": cfg.batch_size,
            "eval_interval": cfg.eval_interval,
            "lr": cfg.lr,
            "base_seed": cfg.base_seed,
            "mnist": {
                "root": mnist_cfg.root,
                "num_workers": mnist_cfg.num_workers,
                "pin_memory": mnist_cfg.pin_memory,
            },
        },
        "steps": steps,
        "acc_task0": accs,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    return out_path
