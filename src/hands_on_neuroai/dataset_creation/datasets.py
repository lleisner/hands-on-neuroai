from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    root: str = "data"
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = False


# Dataset metadata: name -> (torchvision_class, height, width, num_classes)
DATASET_METADATA: dict[str, tuple[type, int, int, int]] = {
    "mnist": (datasets.MNIST, 28, 28, 10),
    "cifar10": (datasets.CIFAR10, 32, 32, 10),
    "cifar100": (datasets.CIFAR100, 32, 32, 100),
    "fashion_mnist": (datasets.FashionMNIST, 28, 28, 10),
}


def get_dataset(
    dataset_name: str,
    root: str,
    train: bool,
    transform: Callable | None = None,
    download: bool = False,
) -> Dataset:
    """Load a torchvision dataset by name."""
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_METADATA:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_METADATA.keys())}"
        )

    dataset_class, _, _, _ = DATASET_METADATA[dataset_name]
    return dataset_class(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def get_image_shape(dataset_name: str) -> tuple[int, int]:
    """Get the (height, width) for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    _, h, w, _ = DATASET_METADATA[dataset_name]
    return h, w


def get_num_classes(dataset_name: str) -> int:
    """Get the number of classes for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    _, _, _, num_classes = DATASET_METADATA[dataset_name]
    return num_classes


def get_default_transforms(dataset_name: str) -> tuple[Callable, Callable]:
    """Return simple default (train_transform, test_transform) for common datasets.

    These are minimal and intended as convenience; users should override with
    dataset-specific augmentations/normalization as needed.
    """
    name = dataset_name.lower()
    if name.startswith("cifar"):
        # CIFAR: normalize with standard mean/std
        train_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
        return train_t, test_t

    # defaults: just ToTensor
    t = transforms.ToTensor()
    return t, t


def get_datasets(
    dataset_name: str,
    config: DatasetConfig,
    transform: Callable | None = None,
    train_transform: Callable | None = None,
    test_transform: Callable | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load train/test datasets.

    Args:
        dataset_name: Name of the dataset.
        config: DatasetConfig with root and other settings.
        transform: Optional transform to apply to all splits (backwards-compatible).
        train_transform/test_transform: optional per-split transforms.

    Returns:
        (train, test) tuple of datasets.
    """
    # Backwards-compatible behavior: if `transform` is provided and per-split
    # transforms are not, use `transform` for all splits.
    if train_transform is None and test_transform is None and transform is not None:
        train_transform = test_transform = transform

    # If still not provided, use defaults
    if train_transform is None or test_transform is None:
        default_train_t, default_test_t = get_default_transforms(dataset_name)
        train_transform = train_transform or default_train_t
        test_transform = test_transform or default_test_t

    # Load base datasets with the appropriate transforms
    train_dataset = get_dataset(
        dataset_name, config.root, train=True, transform=train_transform, download=config.download
    )
    test_dataset = get_dataset(
        dataset_name, config.root, train=False, transform=test_transform, download=config.download
    )

    return train_dataset, test_dataset


def generate_pixel_permutation(
    seed: int,
    num_pixels: int,
) -> Tensor:
    """Generate a reproducible random pixel permutation."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_pixels, generator=generator)
    return permutation


def filter_dataset_by_classes(dataset: Dataset, allowed_classes: Sequence[int]) -> torch.utils.data.Subset:
    """Filter a dataset to only include samples from allowed classes.
    
    Args:
        dataset: The base dataset to filter.
        allowed_classes: List of class indices to keep.
    
    Returns:
        A Subset containing only samples from allowed classes.
    """
    import numpy as np
    allowed_set = set(allowed_classes)
    # Get all labels from the dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    # Find indices matching allowed classes
    keep_idx = np.where(np.isin(labels, list(allowed_set)))[0]
    return torch.utils.data.Subset(dataset, keep_idx)


# --- Modular TaskTransform system ---


class TaskTransform:
    """Abstract base class for deterministic per-task transformations."""

    def __call__(self, img: Tensor | Any) -> Tensor | Any:
        """Apply the task transform to an image (Tensor or PIL)."""
        raise NotImplementedError


class PermutePixels(TaskTransform):
    """Fixed pixel permutation (scrambles spatial structure)."""

    def __init__(self, seed: int):
        self.seed = seed
        self.permutation: Tensor | None = None

    def __call__(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = transforms.ToTensor()(img)

        if self.permutation is None:
            flat_size = img.numel()
            self.permutation = generate_pixel_permutation(self.seed, flat_size)

        flat_img = img.flatten()
        return flat_img[self.permutation].reshape(img.shape)


class Rotate(TaskTransform):
    """Fixed rotation angle."""

    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, img: Any) -> Any:
        return transforms.functional.rotate(img, self.degrees)


class ComposeTaskTransforms(TaskTransform):
    """Compose multiple TaskTransforms sequentially."""

    def __init__(self, transforms_list: Sequence[TaskTransform]):
        self.transforms_list = transforms_list

    def __call__(self, img: Tensor | Any) -> Tensor | Any:
        for t in self.transforms_list:
            img = t(img)
        return img


class TaskDataset(Dataset):
    """Wrapper that applies a TaskTransform to each sample."""

    def __init__(self, base_dataset: Dataset, task_transform: TaskTransform):
        self.base_dataset = base_dataset
        self.task_transform = task_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor | Any, int]:
        img, label = self.base_dataset[idx]
        img = self.task_transform(img)
        return img, label


# Convenience factories for common task transforms
def make_permuted_task(seed: int) -> TaskDataset | None:
    """Factory for a permuted task."""
    return PermutePixels(seed)


def make_rotated_task(degrees: float) -> Rotate:
    """Factory for a rotated task."""
    return Rotate(degrees)


# --- Dataset builders ---


def build_task_datasets(
    dataset_name: str,
    config: DatasetConfig,
    task_transforms: Sequence[TaskTransform],
    transform: Callable | None = None,
) -> tuple[list[Dataset], list[Dataset]]:
    """
    Create train/test datasets for multiple tasks.

    Args:
        dataset_name: Name of the base dataset.
        config: DatasetConfig.
        task_transforms: Sequence of TaskTransforms to apply.
        transform: Optional global transform (applied before TaskTransforms).

    Returns:
        (train_datasets, test_datasets), each a list of TaskDatasets (one per task).
    """
    base_train, base_test = get_datasets(dataset_name, config, transform)

    train_datasets = [TaskDataset(base_train, t) for t in task_transforms]
    test_datasets = [TaskDataset(base_test, t) for t in task_transforms]

    return train_datasets, test_datasets


def build_class_incremental_tasks(
    dataset_name: str,
    config: DatasetConfig,
    class_splits: Sequence[Sequence[int]],
    transform: Callable | None = None,
) -> tuple[list[Dataset], list[Dataset]]:
    """
    Create train/test datasets for class-incremental continual learning.
    Each task contains only samples from specific classes.

    Args:
        dataset_name: Name of the base dataset.
        config: DatasetConfig.
        class_splits: List of class lists, one per task. E.g., [[0,1,2], [3,4,5], [6,7,8], [9]]
        transform: Optional global transform.

    Returns:
        (train_datasets, test_datasets), each a list of Subsets (one per task, filtered by class).
    """
    base_train, base_test = get_datasets(dataset_name, config, transform)

    train_datasets = [filter_dataset_by_classes(base_train, classes) for classes in class_splits]
    test_datasets = [filter_dataset_by_classes(base_test, classes) for classes in class_splits]

    return train_datasets, test_datasets


def build_task_splits(
    dataset_name: str,
    config: DatasetConfig,
    task_transforms: Sequence[TaskTransform],
    val_ratio: float = 0.2,
    transform: Callable | None = None,
    seed: int = 42,
) -> tuple[list[Dataset], list[Dataset], list[Dataset]]:
    """
    Create train/val/test splits for each task with TaskTransforms applied.

    Args:
        dataset_name: Name of the base dataset.
        config: DatasetConfig.
        task_transforms: Sequence of TaskTransforms to apply to each split.
        val_ratio: Fraction of training data to use for validation.
        transform: Optional global transform.
        seed: Random seed for reproducible splits.

    Returns:
        (train_datasets, val_datasets, test_datasets), each a list of TaskDatasets.
        If val_ratio=0, val_datasets will be empty datasets (length 0).
    """
    base_train, base_test = get_datasets(dataset_name, config, transform)

    train_datasets, val_datasets, test_datasets = [], [], []

    total_size = len(base_train)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    for tform in task_transforms:
        if val_size > 0:
            train_split, val_split = random_split(
                base_train,
                [train_size, val_size],
                generator=generator,
            )
        else:
            train_split = base_train
            val_split = torch.utils.data.Subset(base_train, [])

        train_datasets.append(TaskDataset(train_split, tform))
        val_datasets.append(TaskDataset(val_split, tform))
        test_datasets.append(TaskDataset(base_test, tform))

    return train_datasets, val_datasets, test_datasets


def build_dataloaders(
    datasets_list: Sequence[Dataset],
    config: DatasetConfig,
    batch_size: int = 32,
    shuffle: bool = True,
) -> list[DataLoader]:
    """Create DataLoaders from a list of datasets."""
    dataloaders = []
    for dataset in datasets_list:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        dataloaders.append(dl)
    return dataloaders


# --- Cluttered composite dataset (Thor at et al.-style) ---


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
) -> DataLoader:
    """
    Convenience wrapper to create a DataLoader for the cluttered composite dataset.

    Args:
        config: ClutteredDatasetConfig defining sources and synthesis settings.
        batch_size: Batch size for the loader.
        shuffle: Whether to shuffle samples.
        num_workers: DataLoader workers.
        pin_memory: Whether to pin memory (if using CUDA).
    """
    dataset = build_cluttered_composite_dataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
