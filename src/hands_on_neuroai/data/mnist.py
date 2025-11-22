from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class MNISTConfig:
    root: str = "data"
    num_workers: int = 4
    pin_memory: bool = True


def get_mnist_datasets(config: MNISTConfig) -> Tuple[Dataset, Dataset]:
    """Return base MNIST train/test datasets with ToTensor transform."""
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=config.root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=config.root,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def generate_pixel_permutation(
    seed: int,
    num_pixels: int = 28 * 28,
) -> Tensor:
    """Generate a reproducible random pixel permutation."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_pixels, generator=generator)
    return permutation


class PermutedMNIST(Dataset):
    """MNIST dataset with a fixed pixel permutation applied to each image."""

    def __init__(self, base_dataset: Dataset, permutation: Tensor) -> None:
        if permutation.dim() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if permutation.numel() != 28 * 28:
            raise ValueError("Permutation must have length 28*28.")

        self.base_dataset = base_dataset
        self.permutation = permutation.long()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:  # type: ignore[override]
        img, target = self.base_dataset[index]
        flat = img.view(-1)
        perm_flat = flat[self.permutation]
        perm_img = perm_flat.view(1, 28, 28)
        return perm_img, target


def build_permuted_mnist_tasks(
    config: MNISTConfig,
    num_tasks: int,
    base_seed: int = 1000,
) -> Tuple[List[Dataset], List[Dataset]]:
    """Create train/test datasets for multiple permuted-MNIST tasks."""
    base_train, base_test = get_mnist_datasets(config)

    train_datasets: List[Dataset] = []
    test_datasets: List[Dataset] = []

    for task_id in range(num_tasks):
        perm = generate_pixel_permutation(seed=base_seed + task_id)
        train_datasets.append(PermutedMNIST(base_train, perm))
        test_datasets.append(PermutedMNIST(base_test, perm))

    return train_datasets, test_datasets


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_permuted_mnist_loaders(
    config: MNISTConfig,
    num_tasks: int,
    batch_size: int,
    base_seed: int = 1000,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Return lists of train/test loaders for num_tasks permuted-MNIST tasks."""
    train_datasets, test_datasets = build_permuted_mnist_tasks(
        config=config,
        num_tasks=num_tasks,
        base_seed=base_seed,
    )

    train_loaders = [
        make_dataloader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for ds in train_datasets
    ]

    test_loaders = [
        make_dataloader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for ds in test_datasets
    ]

    return train_loaders, test_loaders
