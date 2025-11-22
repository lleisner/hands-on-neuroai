#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import torch

from hands_on_neuroai.utils.cli import (
    parse_perm_mnist_args,
    build_perm_mnist_experiment_configs,
)
from hands_on_neuroai.utils.io import save_perm_mnist_results
from hands_on_neuroai.training.perm_mnist import run_perm_mnist_experiment



def main() -> None:
    args = parse_perm_mnist_args()
    configs, mnist_cfg = build_perm_mnist_experiment_configs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir)

    print(f"Using device: {device}")
    print(f"Planned experiments: {len(configs)}")

    for i, cfg in enumerate(configs, start=1):
        print(
            f"\n=== Experiment {i}/{len(configs)} ===\n"
            f"hidden_dim={cfg.hidden_dim}, "
            f"context_type={cfg.context_type}, "
            f"num_tasks={cfg.num_tasks}, "
            f"steps_per_task={cfg.steps_per_task}",
        )
        steps, accs = run_perm_mnist_experiment(cfg, device=device)
        out_path = save_perm_mnist_results(out_root, cfg, mnist_cfg, steps, accs)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
