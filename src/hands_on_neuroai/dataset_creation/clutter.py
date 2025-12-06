from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .base import get_dataset


@dataclass
class ClutteredDatasetConfig:
    """
    Configuration for cluttered composite datasets.

    Allows mixing multiple base datasets (e.g., MNIST + FashionMNIST + CIFAR),
    and generates images with one intact target and scrambled distractors.
    """

    base_datasets: Sequence[str] = ("mnist", "fashion_mnist")
    root: str = "data"
    split: str = "train"  # "train" or "test"
    image_size: int = 100
    num_clutter_objects: int = 7  # scrambled objects (paper uses 7)
    block_grid: int = 3  # scramble into block_grid x block_grid tiles
    base_object_scale: float = 0.28  # relative object size vs canvas (28px on 100px canvas)
    scale_choices: tuple[float, float] = (0.9, 1.5)
    scale_jitter: float = 0.1
    rotation_deg: float = 30.0
    rotation_jitter: float = 5.0
    location_jitter: float = 2.5
    num_samples: Optional[int] = None  # if None, min(len(base_datasets))
    seed: int = 0
    return_aux_labels: bool = False  # include location/scale/orientation labels
    base_transform: Callable | None = None  # applied to base datasets (defaults to ToTensor)
    output_transform: Callable | None = None  # optional transform on final composite
    allow_channel_broadcast: bool = True  # repeat grayscale to RGB if needed
    allow_channel_reduce: bool = True  # collapse RGB to grayscale if needed
    download: bool = False


def _load_base_datasets(config: ClutteredDatasetConfig) -> list[Dataset]:
    """Load the base datasets specified in the config."""
    base_tform = config.base_transform or transforms.ToTensor()
    train_flag = config.split.lower() == "train"

    base_names: Sequence[str]
    if isinstance(config.base_datasets, str):
        base_names = (config.base_datasets,)
    else:
        base_names = config.base_datasets

    base_datasets = []
    for name in base_names:
        base = get_dataset(
            dataset_name=name,
            root=config.root,
            train=train_flag,
            transform=base_tform,
            download=config.download,
        )
        base_datasets.append(base)
    return base_datasets


def _infer_channels(dataset: Dataset) -> int:
    """Peek at the first sample to infer number of channels."""
    sample, _ = dataset[0]
    if isinstance(sample, Tensor):
        return sample.shape[0]
    # Fallback: convert with ToTensor
    return transforms.ToTensor()(sample).shape[0]


def _ensure_channels(
    img: Tensor,
    out_channels: int,
    allow_broadcast: bool,
    allow_reduce: bool,
) -> Tensor:
    """Ensure img has out_channels, repeating or reducing channels if permitted."""
    in_channels = img.shape[0]
    if in_channels == out_channels:
        return img
    if in_channels == 1 and out_channels == 3 and allow_broadcast:
        return img.repeat(3, 1, 1)
    if in_channels == 3 and out_channels == 1 and allow_reduce:
        return TF.rgb_to_grayscale(img)
    raise ValueError(
        f"Channel mismatch: got {in_channels} channels, expected {out_channels}. "
        "Set allow_channel_broadcast=True to repeat grayscale to RGB or "
        "allow_channel_reduce=True to downmix RGB to grayscale."
    )


def _random_choice(seq: Sequence[Any], generator: torch.Generator) -> Any:
    idx = torch.randint(len(seq), (1,), generator=generator).item()
    return seq[idx]


def _affine_transform(
    img: Tensor,
    generator: torch.Generator,
    scale_choices: tuple[float, float],
    scale_jitter: float,
    rotation_deg: float,
    rotation_jitter: float,
) -> tuple[Tensor, float, float]:
    """Sample scale and rotation sign/jitter, apply affine, and return labels.

    Returns:
        transformed image, base scale (before jitter), orientation sign (+1/-1 for CW/CCW)
    """
    base_scale = _random_choice(scale_choices, generator)
    scale_jitter_val = (torch.rand(1, generator=generator) * 2 * scale_jitter - scale_jitter).item()
    scale = float(base_scale + scale_jitter_val)
    sign = -1.0 if torch.rand(1, generator=generator) < 0.5 else 1.0
    rot_jitter_val = (torch.rand(1, generator=generator) * 2 * rotation_jitter - rotation_jitter).item()
    angle = float(sign * rotation_deg + rot_jitter_val)
    transformed = TF.affine(img, angle=angle, translate=(0.0, 0.0), scale=scale, shear=[0.0, 0.0])
    return transformed, base_scale, sign


def _scramble_blocks(img: Tensor, grid: int, generator: torch.Generator) -> Tensor:
    """Divide img into equal-sized grid x grid blocks (with padding), permute, then crop."""
    c, h, w = img.shape
    pad_h = (grid - h % grid) % grid
    pad_w = (grid - w % grid) % grid
    padded = F.pad(img, (0, pad_w, 0, pad_h))
    _, hp, wp = padded.shape
    block_h = hp // grid
    block_w = wp // grid

    # Extract blocks
    blocks = []
    for gy in range(grid):
        for gx in range(grid):
            blocks.append(
                padded[:, gy * block_h : (gy + 1) * block_h, gx * block_w : (gx + 1) * block_w]
            )

    perm = torch.randperm(len(blocks), generator=generator)
    permuted = [blocks[i] for i in perm]

    # Reassemble padded canvas then crop back to original size
    out = torch.zeros_like(padded)
    idx = 0
    for gy in range(grid):
        for gx in range(grid):
            out[:, gy * block_h : (gy + 1) * block_h, gx * block_w : (gx + 1) * block_w] = permuted[idx]
            idx += 1

    return out[:, :h, :w]


class ClutteredCompositeDataset(Dataset):
    """
    Synthetic cluttered dataset with one intact target and multiple scrambled distractors.

    Designed to emulate Thorat et al. 2022, but generalizes to arbitrary base datasets
    (MNIST/FashionMNIST by default, can also use CIFAR).
    """

    def __init__(self, config: ClutteredDatasetConfig):
        self.config = config
        self.base_datasets = _load_base_datasets(config)
        self.out_channels = _infer_channels(self.base_datasets[0])
        self.num_samples = (
            config.num_samples
            if config.num_samples is not None
            else min(len(ds) for ds in self.base_datasets)
        )

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def _sample_base(self, generator: torch.Generator) -> tuple[Tensor, int]:
        """Draw a random sample from pooled base datasets and harmonize channels.

        This lets you mix grayscale and RGB sources; grayscale can be broadcast to RGB
        or RGB downmixed to grayscale if allowed. Sampling is driven by the provided
        torch.Generator for determinism.
        """
        ds = _random_choice(self.base_datasets, generator)
        idx = torch.randint(len(ds), (1,), generator=generator).item()
        img, label = ds[idx]
        if not isinstance(img, Tensor):
            img = transforms.ToTensor()(img)
        img = _ensure_channels(
            img,
            self.out_channels,
            self.config.allow_channel_broadcast,
            self.config.allow_channel_reduce,
        )
        # Optionally resize base object to a fraction of the canvas size (keeps objects
        # visually comparable when changing canvas dimensions).
        if self.config.base_object_scale is not None:
            target_size = int(round(self.config.image_size * self.config.base_object_scale))
            if target_size > 0 and (img.shape[1] != target_size or img.shape[2] != target_size):
                img = TF.resize(img, [target_size, target_size], interpolation=InterpolationMode.BILINEAR)
        return img, int(label)

    def _place_on_canvas(self, canvas: Tensor, obj: Tensor, center_y: float, center_x: float) -> None:
        """Alpha-free paste: place object on canvas with clipping and max-overlap blending.

        The object is centered at (center_y, center_x); any parts outside the canvas are
        dropped. Overlaps keep the per-pixel max so later draws (e.g., target) appear on top.
        """
        c, h, w = obj.shape
        top = int(round(center_y - h / 2.0))
        left = int(round(center_x - w / 2.0))

        y0 = max(top, 0)
        x0 = max(left, 0)
        y1 = min(top + h, canvas.shape[1])
        x1 = min(left + w, canvas.shape[2])

        if y0 >= y1 or x0 >= x1:
            return

        obj_y0 = y0 - top
        obj_x0 = x0 - left
        obj_y1 = obj_y0 + (y1 - y0)
        obj_x1 = obj_x0 + (x1 - x0)

        region = canvas[:, y0:y1, x0:x1]
        canvas[:, y0:y1, x0:x1] = torch.max(region, obj[:, obj_y0:obj_y1, obj_x0:obj_x1])

    def __getitem__(self, idx: int) -> tuple[Tensor, Any]:  # type: ignore[override]
        """Generate one composite: scramble clutter, place in quadrants, then overlay target.

        Steps (seeded by idx):
        1) Choose a target quadrant and build a canvas of zeros.
        2) Place scrambled clutter. If `num_clutter_objects` >= 7 (paper default), we mirror
           the paper: 2 scrambled items in each non-target quadrant and 1 in the target
           quadrant (total 7). Otherwise we ensure at least one per quadrant, distributing any
           remaining scrambled items randomly. Scrambling permutes blocks after random
           scale/rotation.
        3) Sample the intact target, apply random scale/rotation, and place it in its quadrant
           with jitter so it occludes underlying clutter.
        4) Optionally apply an output transform and/or return auxiliary labels.
        """
        g = torch.Generator()
        g.manual_seed(self.config.seed + idx)

        canvas = torch.zeros(self.out_channels, self.config.image_size, self.config.image_size)
        centers = [
            (self.config.image_size * 0.25, self.config.image_size * 0.25),
            (self.config.image_size * 0.25, self.config.image_size * 0.75),
            (self.config.image_size * 0.75, self.config.image_size * 0.25),
            (self.config.image_size * 0.75, self.config.image_size * 0.75),
        ]
        target_quadrant = torch.randint(4, (1,), generator=g).item()

        target_label: int = 0
        chosen_scale = self.config.scale_choices[0]
        orientation_sign = 1.0

        num_scrambled = max(self.config.num_clutter_objects, 0)
        quadrants_for_scramble: list[int] = []
        remaining = 0

        if num_scrambled >= 7:
            # Paper default: 2 scrambled per non-target quadrant, 1 scrambled in the target quadrant.
            for q in range(4):
                repeats = 1 if q == target_quadrant else 2
                quadrants_for_scramble.extend([q] * repeats)
            remaining = num_scrambled - 7
        else:
            # Ensure at least one scrambled per quadrant, then distribute the rest.
            quadrants_for_scramble.extend([0, 1, 2, 3])
            remaining = max(num_scrambled - 4, 0)

        if remaining > 0:
            extra_quads = torch.randint(4, (remaining,), generator=g).tolist()
            quadrants_for_scramble.extend(extra_quads)

        for quad_idx in quadrants_for_scramble:
            cy, cx = centers[quad_idx]
            scr_img, _ = self._sample_base(g)
            scr_img, _, _ = _affine_transform(
                scr_img,
                generator=g,
                scale_choices=self.config.scale_choices,
                scale_jitter=self.config.scale_jitter,
                rotation_deg=self.config.rotation_deg,
                rotation_jitter=self.config.rotation_jitter,
            )
            scr_img = _scramble_blocks(scr_img, self.config.block_grid, g)
            jitter_y = float((torch.rand(1, generator=g) * 2 * self.config.location_jitter - self.config.location_jitter).item())
            jitter_x = float((torch.rand(1, generator=g) * 2 * self.config.location_jitter - self.config.location_jitter).item())
            self._place_on_canvas(canvas, scr_img, cy + jitter_y, cx + jitter_x)

        # Target object last (drawn on top).
        target_img, target_label = self._sample_base(g)
        target_img, chosen_scale, orientation_sign = _affine_transform(
            target_img,
            generator=g,
            scale_choices=self.config.scale_choices,
            scale_jitter=self.config.scale_jitter,
            rotation_deg=self.config.rotation_deg,
            rotation_jitter=self.config.rotation_jitter,
        )
        cy, cx = centers[target_quadrant]
        jitter_y = float((torch.rand(1, generator=g) * 2 * self.config.location_jitter - self.config.location_jitter).item())
        jitter_x = float((torch.rand(1, generator=g) * 2 * self.config.location_jitter - self.config.location_jitter).item())
        self._place_on_canvas(canvas, target_img, cy + jitter_y, cx + jitter_x)

        if self.config.output_transform is not None:
            canvas = self.config.output_transform(canvas)

        if self.config.return_aux_labels:
            aux = {
                "location": target_quadrant,
                "scale_choice": float(chosen_scale),
                "orientation_sign": float(orientation_sign),
            }
            return canvas, target_label, aux

        return canvas, target_label


def build_cluttered_composite_dataset(config: ClutteredDatasetConfig) -> ClutteredCompositeDataset:
    """Factory to create a cluttered composite dataset instance."""
    return ClutteredCompositeDataset(config)


def build_cluttered_dataloader(
    config: ClutteredDatasetConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Convenience wrapper to create a DataLoader for the cluttered composite dataset.

    Args:
        config: ClutteredDatasetConfig defining sources and synthesis settings.
        batch_size: Batch size for the loader.
        shuffle: Whether to shuffle samples.
        num_workers: DataLoader workers.
        pin_memory: Whether to pin memory (if using CUDA).
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0).
        prefetch_factor: Number of batches to prefetch per worker.
    """
    dataset = build_cluttered_composite_dataset(config)
    use_persistent = persistent_workers and num_workers > 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
