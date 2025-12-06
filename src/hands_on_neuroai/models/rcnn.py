from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _conv_out_dim(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Compute output spatial dim for a conv layer."""
    return (size + 2 * padding - kernel) // stride + 1


def _pool_out_dim(size: int, kernel: int, stride: Optional[int] = None, padding: int = 0) -> int:
    """Compute output spatial dim for a pooling layer."""
    stride = stride or kernel
    return (size + 2 * padding - kernel) // stride + 1


@dataclass
class RCNNConfig:
    """Config for the small recurrent CNN (Thorat et al.-style)."""

    in_channels: int = 1
    num_classes: int = 20
    image_size: int = 100
    timesteps: int = 4
    interaction: Literal["multiplicative", "additive"] = "multiplicative"
    conv1_channels: int = 8
    conv2_channels: int = 16
    fc_dim: int = 128
    # Kernels/strides mimic the paper defaults
    conv_kernel: int = 5
    pool_kernel: int = 3
    pool_stride: int = 3
    # Transposed conv parameters for top-down connections
    td_conv2_to_input: Tuple[int, int, int] = (20, 10, 0)  # kernel, stride, padding
    td_conv1_to_input: Tuple[int, int, int] = (7, 3, 0)
    # Map conv2 (pooled 9x9) back to conv1 pooled map (32x32): (9-1)*3 + 8 = 32
    td_conv2_to_conv1: Tuple[int, int, int] = (8, 3, 0)


class RCNN(nn.Module):
    """
    Recurrent CNN with lateral and top-down connections, unrolled over time.

    Follows Thorat et al. 2022 in spirit: two conv layers + FC, recurrent
    lateral/top-down signals that multiplicatively modulate feedforward flow,
    and a classifier over concatenated FC activations across timesteps.
    """

    def __init__(self, config: RCNNConfig):
        super().__init__()
        self.cfg = config
        self.interaction = config.interaction
        self.timesteps = config.timesteps

        # Feedforward layers
        self.conv1 = nn.Conv2d(config.in_channels, config.conv1_channels, kernel_size=config.conv_kernel)
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=config.conv_kernel)
        self.pool = nn.MaxPool2d(kernel_size=config.pool_kernel, stride=config.pool_stride)

        # Compute spatial dims after conv/pool to size recurrent projections and FC
        h1 = _conv_out_dim(config.image_size, config.conv_kernel)
        h1 = _pool_out_dim(h1, config.pool_kernel, config.pool_stride)
        h2 = _conv_out_dim(h1, config.conv_kernel)
        h2 = _pool_out_dim(h2, config.pool_kernel, config.pool_stride)
        w1 = h1
        w2 = h2
        self.spatial1 = (h1, w1)
        self.spatial2 = (h2, w2)

        self.fc = nn.Linear(config.conv2_channels * h2 * w2, config.fc_dim)

        # Lateral recurrence
        self.lat_conv1 = nn.Conv2d(config.conv1_channels, config.conv1_channels, kernel_size=5, padding=2)
        self.lat_conv2 = nn.Conv2d(config.conv2_channels, config.conv2_channels, kernel_size=5, padding=2)
        self.lat_fc = nn.Linear(config.fc_dim, config.fc_dim)

        # Top-down recurrence
        k, s, p = config.td_conv2_to_input
        self.td_conv2_to_input = nn.ConvTranspose2d(config.conv2_channels, config.in_channels, kernel_size=k, stride=s, padding=p)
        k, s, p = config.td_conv1_to_input
        self.td_conv1_to_input = nn.ConvTranspose2d(config.conv1_channels, config.in_channels, kernel_size=k, stride=s, padding=p)
        k, s, p = config.td_conv2_to_conv1
        self.td_conv2_to_conv1 = nn.ConvTranspose2d(config.conv2_channels, config.conv1_channels, kernel_size=k, stride=s, padding=p)
        self.td_fc_to_conv2 = nn.Linear(config.fc_dim, config.conv2_channels * h2 * w2)
        self.td_fc_to_conv1 = nn.Linear(config.fc_dim, config.conv1_channels * h1 * w1)

        # Classifier over concatenated FC activations across time
        self.classifier = nn.Linear(config.fc_dim * config.timesteps, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _interact(self, ff: Tensor, rec: Tensor) -> Tensor:
        if self.interaction == "multiplicative":
            return torch.relu(ff * (1.0 + rec))
        return torch.relu(ff + rec)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the RCNN for cfg.timesteps and return logits from concatenated FC states.
        """
        batch_size = x.shape[0]
        device = x.device

        # Recurrent states initialized to zeros
        conv1_state = torch.zeros(batch_size, self.cfg.conv1_channels, *self.spatial1, device=device)
        conv2_state = torch.zeros(batch_size, self.cfg.conv2_channels, *self.spatial2, device=device)
        fc_state = torch.zeros(batch_size, self.cfg.fc_dim, device=device)

        fc_traces: List[Tensor] = []

        for t in range(self.timesteps):
            # Top-down to input
            if t == 0:
                rec_input = torch.zeros_like(x)
            else:
                td = self.td_conv2_to_input(conv2_state, output_size=x.shape[-2:])
                td = td + self.td_conv1_to_input(conv1_state, output_size=x.shape[-2:])
                rec_input = td
            x_mod = torch.clamp(x * (1.0 + rec_input), 0.0, 1.0)

            # Conv1
            ff1 = self.pool(torch.relu(self.conv1(x_mod)))
            if t == 0:
                rec1 = torch.zeros_like(ff1)
            else:
                td_c2_c1 = self.td_conv2_to_conv1(conv2_state, output_size=self.spatial1)
                td_fc_c1 = self.td_fc_to_conv1(fc_state).view(batch_size, self.cfg.conv1_channels, *self.spatial1)
                rec1 = self.lat_conv1(conv1_state) + td_c2_c1 + td_fc_c1
            conv1_state = self._interact(ff1, rec1)

            # Conv2
            ff2 = self.pool(torch.relu(self.conv2(conv1_state)))
            if t == 0:
                rec2 = torch.zeros_like(ff2)
            else:
                td_fc_c2 = self.td_fc_to_conv2(fc_state).view(batch_size, self.cfg.conv2_channels, *self.spatial2)
                rec2 = self.lat_conv2(conv2_state) + td_fc_c2
            conv2_state = self._interact(ff2, rec2)

            # FC
            ff_fc = torch.relu(self.fc(conv2_state.flatten(1)))
            rec_fc = torch.zeros_like(ff_fc) if t == 0 else self.lat_fc(fc_state)
            fc_state = self._interact(ff_fc, rec_fc)
            fc_traces.append(fc_state)

        logits = self.classifier(torch.cat(fc_traces, dim=1))
        return logits
