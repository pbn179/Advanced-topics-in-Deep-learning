"""Reproducible MNIST loading with a fixed 55,000/5,000 training split."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


# Both mirrors distribute the original gzip files. Torchvision checks the
# published MNIST MD5 for every downloaded resource before extracting it.
DEFAULT_MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch; use deterministic supported operations.

    Repeatability is expected on the same software/device. Bitwise equality
    across CPU, CUDA, MPS, operating systems or PyTorch versions is not promised.
    """
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def split_indices(
    length: int = 60_000,
    validation_size: int = 5_000,
    split_seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Make a fixed random partition independent of model initialization seeds."""
    if not 0 < validation_size < length:
        raise ValueError("validation_size must be positive and smaller than length")
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[validation_size:], indices[:validation_size]


def _seed_worker(worker_id: int) -> None:
    """Top-level callback so opt-in multiprocessing also works on Windows."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _mnist_dataset(root: Path, train: bool, download: bool, mirrors: Sequence[str]):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    # Keep the MNIST class name (used for the cache directory) while restricting
    # the mirror override to this object rather than changing global class data.
    dataset = MNIST.__new__(MNIST)
    dataset.mirrors = list(mirrors)
    MNIST.__init__(dataset, root=str(root), train=train, transform=ToTensor(), download=download)
    return dataset


def _limited(indices: list[int], limit: int | None, label: str) -> list[int]:
    if limit is None:
        return indices
    if not isinstance(limit, int) or not 0 < limit <= len(indices):
        raise ValueError(f"{label} must be between 1 and {len(indices)}, or None")
    return indices[:limit]


def make_loaders(
    data_dir: str | Path,
    batch_size: int = 60,
    split_seed: int = 42,
    seed: int = 0,
    num_workers: int = 0,
    download: bool = True,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    mirrors: Sequence[str] | None = None,
    preload_tensors: bool = False,
) -> dict[str, DataLoader]:
    """Load real MNIST, never replacing unavailable data with synthetic examples.

    Pixels use ToTensor's [0, 1] scaling, with no augmentation or additional
    normalization. The fixed split is reused for every architecture/model seed.
    Optional limits are explicitly for quick debugging, not scientific runs.
    num_workers=0 is the notebook-safe default on Windows and macOS.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    selected_mirrors = DEFAULT_MNIST_MIRRORS if mirrors is None else tuple(mirrors)
    if not selected_mirrors or any(not url.startswith("https://") for url in selected_mirrors):
        raise ValueError("Provide at least one HTTPS MNIST mirror")
    root = Path(data_dir).expanduser()
    train_dataset = _mnist_dataset(root, train=True, download=download, mirrors=selected_mirrors)
    test_dataset = _mnist_dataset(root, train=False, download=download, mirrors=selected_mirrors)
    if preload_tensors:
        # Exactly ToTensor's uint8 -> float32 / 255 operation, once per image.
        # Keeps the entire dataset/split, avoiding repeated PIL conversions.
        train_dataset = tensorize_mnist(train_dataset)
        test_dataset = tensorize_mnist(test_dataset)
    train_indices, val_indices = split_indices(len(train_dataset), 5_000, split_seed)
    datasets = {
        "train": Subset(train_dataset, _limited(train_indices, train_limit, "train_limit")),
        "validation": Subset(train_dataset, _limited(val_indices, val_limit, "val_limit")),
        "test": Subset(test_dataset, _limited(list(range(len(test_dataset))), test_limit, "test_limit")),
    }
    loaders = {}
    for offset, (name, dataset) in enumerate(datasets.items()):
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size if name == "train" else max(batch_size, 256),
            shuffle=name == "train",
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
            pin_memory=False,
        )
    return loaders


def tensorize_mnist(dataset):
    """Materialize the identical [0,1] MNIST tensors (~220 MB for train+test)."""
    return TensorDataset(dataset.data.unsqueeze(1).to(torch.float32).div(255), dataset.targets)
