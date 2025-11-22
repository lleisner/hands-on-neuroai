#!/usr/bin/env python3
# scripts/download_data.py

from __future__ import annotations

import argparse

from hands_on_neuroai.data.download import download_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for the project.")
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Folder where MNIST will be stored (default: data/).",
    )

    args = parser.parse_args()
    download_mnist(root=args.root)


if __name__ == "__main__":
    main()
