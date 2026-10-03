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


class ResNetCoarseNet(nn.Module):
    """The fine backbone's architecture (CIFAR ResNet-18) with a 20-way head: a
    stronger coarse net (arch "resnet18"). Still a separate, independently trained
    network; it only shares the architecture, never weights."""

    def __init__(self, num_coarse: int = NUM_COARSE) -> None:
        super().__init__()
        from .backbone import ResNetBackbone  # local import avoids a cycle
        self.backbone = ResNetBackbone(feature_dim=512)
        self.head = nn.Linear(512, num_coarse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


COARSE_ARCHS = {"small": CoarseNet, "resnet18": ResNetCoarseNet}


def build_coarse_net(num_coarse: int = NUM_COARSE, arch: str = "small") -> nn.Module:
    if arch not in COARSE_ARCHS:
        raise ValueError(f"unknown coarse arch {arch!r}; one of {list(COARSE_ARCHS)}")
    return COARSE_ARCHS[arch](num_coarse)


def load_coarse_net(ckpt_path: str, device) -> nn.Module:
    """Rebuild the net named by the checkpoint's `arch` (checkpoints written before the
    field existed are the small CNN) and load its weights."""
    state = torch.load(ckpt_path, map_location=device)
    arch = state.get("arch", "small") if "model" in state else "small"
    net = build_coarse_net(arch=arch)
    net.load_state_dict(state["model"] if "model" in state else state)
    return net.to(device)
