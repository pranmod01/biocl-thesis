"""Small shared helpers: seeding and device resolution."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seen_mask_from_classes(seen_classes, num_fine: int, device) -> torch.Tensor:
    """Bool[num_fine] mask, True for classes seen so far."""
    mask = torch.zeros(num_fine, dtype=torch.bool, device=device)
    mask[torch.tensor(seen_classes, dtype=torch.long, device=device)] = True
    return mask
