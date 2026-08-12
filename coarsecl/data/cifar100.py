"""CIFAR-100 with its native fine->coarse (subclass->superclass) hierarchy.

torchvision's CIFAR100 yields fine labels in 0..99 (alphabetical class order). The
canonical mapping below sends each fine index to its superclass index in 0..19.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

# Canonical CIFAR-100 fine(0..99) -> coarse(0..19) map, in torchvision label order.
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def fine_to_coarse_tensor(device=None) -> torch.Tensor:
    """LongTensor[100] mapping fine label -> coarse label, on `device`."""
    return torch.tensor(FINE_TO_COARSE, dtype=torch.long, device=device)


def coarse_of(fine_labels: torch.Tensor) -> torch.Tensor:
    """Map a batch of fine labels to coarse labels (same device)."""
    table = fine_to_coarse_tensor(fine_labels.device)
    return table[fine_labels]


def _transforms(train: bool):
    norm = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.ToTensor(), norm])


def get_datasets(data_root: str = "data") -> Tuple[Dataset, Dataset]:
    """Return (train, test) CIFAR-100 datasets; each item is (image, fine_label)."""
    train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=_transforms(True)
    )
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=_transforms(False)
    )
    return train, test
