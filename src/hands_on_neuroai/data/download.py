# src/hands_on_neuroai/data/download.py
from __future__ import annotations

import os
from typing import Optional
from torchvision import datasets, transforms


def download_mnist(root: str = "data") -> None:
    """Download MNIST into the given root folder.

    This uses torchvision's built-in MNIST downloader
    and ensures the directory exists.
    """
    os.makedirs(root, exist_ok=True)
    transform = transforms.ToTensor()

    print(f"📥 Downloading MNIST into: {root}/MNIST")

    datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    print("✅ MNIST download complete.")
