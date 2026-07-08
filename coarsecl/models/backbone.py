"""Small CIFAR-style ResNet-18 feature extractor (no classification head).

Produces a `feature_dim`-d penultimate vector. Identical across every condition and
conditioning mechanism, so the only thing that varies between runs is the coarse
signal and how it is injected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetBackbone(nn.Module):
    """CIFAR ResNet-18: 3x3 stem, no max-pool, global-avg-pooled features."""

    def __init__(self, num_blocks=(2, 2, 2, 2), feature_dim: int = 512) -> None:
        super().__init__()
        assert feature_dim == 512, "resnet18_small produces 512-d features"
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.feature_dim = 512

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out  # [B, 512]


class FrozenCoarseBackbone(nn.Module):
    """Frozen mid-stack features from a coarse-trained CoarseNet (default: block 2, the
    tap that best preserves fine information per the layer-wise probe). A stable
    'cortical' substrate — features never drift, so continual forgetting is head-only.

    Loaded frozen: params require no grad, BN stays in eval mode regardless of the
    parent's train/eval state, and the forward runs under no_grad."""

    def __init__(self, ckpt_path: str, tap_upto: int = 13) -> None:
        super().__init__()
        from .coarse_net import build_coarse_net  # local import avoids a cycle
        cnet = build_coarse_net()
        state = torch.load(ckpt_path, map_location="cpu")
        cnet.load_state_dict(state["model"])
        # keep only the conv trunk up to (and including) the tap layer
        self.trunk = nn.Sequential(*list(cnet.features)[: tap_upto + 1])
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.trunk.eval()
        with torch.no_grad():
            self.feature_dim = self.trunk(torch.zeros(1, 3, 32, 32)).flatten(1).shape[1]

    def train(self, mode: bool = True) -> "FrozenCoarseBackbone":
        # frozen: never enter train mode, so BN running stats stay fixed
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.trunk(x).flatten(1)


def build_backbone(name: str, feature_dim: int, coarse_ckpt: str = None):
    if name == "resnet18_small":
        return ResNetBackbone(feature_dim=feature_dim)
    if name == "frozen_coarse_block2":
        if coarse_ckpt is None:
            raise ValueError("frozen_coarse_block2 needs the coarse-net checkpoint path")
        return FrozenCoarseBackbone(coarse_ckpt, tap_upto=13)  # block 2 (2nd maxpool)
    raise ValueError(f"unknown backbone {name!r}")
