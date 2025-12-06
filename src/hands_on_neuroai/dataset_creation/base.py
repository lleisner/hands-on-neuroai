from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    root: str = "data"
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = False
    persistent_workers: bool = True  # Keep workers alive between epochs (requires num_workers > 0)
    prefetch_factor: int = 2  # Number of batches to prefetch per worker


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

    # If still not provided, use sensible defaults (ToTensor + optional normalization)
    if train_transform is None or test_transform is None:
        name = dataset_name.lower()
        if name.startswith("cifar"):
            # CIFAR: normalize with standard mean/std
            default_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            ])
        else:
            # defaults: just ToTensor
            default_t = transforms.ToTensor()
        
        train_transform = train_transform or default_t
        test_transform = test_transform or default_t

    # Load base datasets with the appropriate transforms
    train_dataset = get_dataset(
        dataset_name, config.root, train=True, transform=train_transform, download=config.download
    )
    test_dataset = get_dataset(
        dataset_name, config.root, train=False, transform=test_transform, download=config.download
    )

    return train_dataset, test_dataset


def build_dataloaders(
    datasets_list: list[Dataset],
    config: DatasetConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[DataLoader]:
    """Create DataLoaders from a list of datasets."""
    dataloaders = []
    for dataset in datasets_list:
        # persistent_workers requires num_workers > 0
        use_persistent = config.persistent_workers and config.num_workers > 0
        
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            drop_last=drop_last,
        )
        dataloaders.append(dl)
    return dataloaders
