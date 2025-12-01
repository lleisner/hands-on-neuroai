from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


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
