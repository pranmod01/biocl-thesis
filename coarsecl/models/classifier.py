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
    def __init__(self, backbone: nn.Module, strategy: ConditioningStrategy) -> None:
        super().__init__()
        self.backbone = backbone
        self.strategy = strategy

    def forward(self, images: torch.Tensor,
                coarse_dist: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        return self.strategy(features, coarse_dist)

    def compute_loss(self, images, fine_targets, coarse_dist, seen_mask) -> torch.Tensor:
        outputs = self.forward(images, coarse_dist)
        return self.strategy.compute_loss(outputs, fine_targets, coarse_dist, seen_mask)

    @torch.no_grad()
    def predict(self, images, coarse_dist, seen_mask) -> torch.Tensor:
        """Class-IL prediction: argmax over the seen-class set only."""
        logits = self.forward(images, coarse_dist)["fine_logits"]
        return masked_fine_logits(logits, seen_mask).argmax(dim=1)


def conditioning_for(cfg: Config) -> str:
    """Resolve which conditioning mechanism this run uses (the coupling rule)."""
    return "none" if cfg.coarse.source == "none" else cfg.conditioning.mechanism


def build_classifier(cfg: Config) -> ConditionedClassifier:
    backbone = build_backbone(cfg.model.backbone, cfg.model.feature_dim)
    strategy = build_conditioning(
        conditioning_for(cfg), backbone.feature_dim, cfg.conditioning
    )
    return ConditionedClassifier(backbone, strategy)
