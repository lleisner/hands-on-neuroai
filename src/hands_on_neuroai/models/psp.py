from __future__ import annotations

from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Generator


class PSPLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tasks: int,
        context_fn: Callable[[int, int | None], Tensor],
        seed: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.num_tasks = num_tasks

        g = Generator().manual_seed(seed)
        self.contexts: Dict[int, Tensor] = {}

        for t in range(num_tasks):
            s = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item())
            self.contexts[t] = context_fn(in_features, s)

        self.register_buffer("active_context", torch.ones(in_features))
        if num_tasks > 0:
            self.set_task(0)

    def set_task(self, task_id: int) -> None:
        self.active_context = self.contexts[task_id].to(self.active_context.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x * self.active_context)


class PSPMLP2Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_tasks: int,
        context_fn: Callable[[int, int | None], Tensor],
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PSPLinear(784, hidden_dim, num_tasks, context_fn, seed)
        self.fc2 = PSPLinear(hidden_dim, hidden_dim, num_tasks, context_fn, seed + 1)
        self.fc3 = PSPLinear(hidden_dim, 10, num_tasks, context_fn, seed + 2)

        self.num_tasks = num_tasks
        self.set_task(0)

    def set_task(self, t: int) -> None:
        self.fc1.set_task(t)
        self.fc2.set_task(t)
        self.fc3.set_task(t)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
