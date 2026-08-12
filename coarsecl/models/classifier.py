"""ConditionedClassifier = shared backbone + a conditioning strategy.

The factory encodes the one coupling rule: when there is no coarse source, the
conditioning is forced to `none` (the mechanism field is meaningless without a
signal). Otherwise the configured mechanism is used.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import Config
from .backbone import build_backbone
from .conditioning import (ConditioningStrategy, build_conditioning,
                           masked_fine_logits)


class ConditionedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, strategy: ConditioningStrategy,
                 rotation_head: Optional[nn.Module] = None,
                 rotation_tap: str = "layer2") -> None:
        super().__init__()
        self.backbone = backbone
        self.strategy = strategy
        self.rotation_head = rotation_head  # None unless the rotation aux is enabled
        self.rotation_tap = rotation_tap

    def forward(self, images: torch.Tensor,
                coarse_dist: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        return self.strategy(features, coarse_dist)

    def compute_loss(self, images, fine_targets, coarse_dist, seen_mask) -> torch.Tensor:
        outputs = self.forward(images, coarse_dist)
        return self.strategy.compute_loss(outputs, fine_targets, coarse_dist, seen_mask)

    def rotation_logits(self, rot_images: torch.Tensor) -> torch.Tensor:
        """4-way rotation logits from the intermediate trunk tap (aux task only)."""
        feat_map = self.backbone.forward_tap(rot_images, self.rotation_tap)
        return self.rotation_head(feat_map)

    @torch.no_grad()
    def predict(self, images, coarse_dist, seen_mask) -> torch.Tensor:
        """Class-IL prediction: argmax over the seen-class set only."""
        outputs = self.forward(images, coarse_dist)
        logits = self.strategy.eval_logits(outputs, coarse_dist)
        return masked_fine_logits(logits, seen_mask).argmax(dim=1)


def conditioning_for(cfg: Config) -> str:
    """Resolve which conditioning mechanism this run uses (the coupling rule)."""
    return "none" if cfg.coarse.source == "none" else cfg.conditioning.mechanism


def build_classifier(cfg: Config) -> ConditionedClassifier:
    backbone = build_backbone(cfg.model.backbone, cfg.model.feature_dim,
                              coarse_ckpt=cfg.coarse.trained.ckpt)
    strategy = build_conditioning(
        conditioning_for(cfg), backbone.feature_dim, cfg.conditioning
    )
    rotation_head = None
    if cfg.rotation.enabled:
        from .rotation import RotationHead
        rotation_head = RotationHead(
            backbone.TAP_CHANNELS[cfg.rotation.tap], hidden=cfg.rotation.hidden
        )
    return ConditionedClassifier(backbone, strategy, rotation_head, cfg.rotation.tap)
