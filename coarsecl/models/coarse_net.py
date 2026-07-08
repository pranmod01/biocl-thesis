"""Independent small CNN that predicts the 20 superclasses (condition d).

Deliberately NOT the fine backbone — it is its own network, trained once upfront on
coarse labels (see coarsecl/coarse/train_coarse.py) and loaded frozen at run time.
Its softmax over the 20 superclasses is the coarse signal for the fine classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_COARSE


class CoarseNet(nn.Module):
    def __init__(self, num_coarse: int = NUM_COARSE) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                              # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                             # 4x4
        )
        self.head = nn.Linear(128 * 4 * 4, num_coarse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.head(z)  # logits over 20 superclasses


def build_coarse_net(num_coarse: int = NUM_COARSE) -> CoarseNet:
    return CoarseNet(num_coarse)
