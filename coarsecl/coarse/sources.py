"""Coarse signal sources: the four-condition axis.

Every source returns a distribution over the 20 superclasses, FloatTensor[B, 20]
(or None for baseline). This uniform return type is what lets the conditioning
mechanism stay agnostic to which condition produced the signal. A new condition is
a new @COARSE_SOURCES.register class + one config value.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..config import NUM_COARSE, Config
from ..data.cifar100 import coarse_of
from ..registry import Registry

COARSE_SOURCES: "Registry[CoarseSource]" = Registry("coarse_source")


class CoarseSource:
    """Maps a batch to a coarse distribution (or None). Stateless unless noted."""

    num_coarse = NUM_COARSE

    def get_coarse_dist(self, images: torch.Tensor,
                        fine_targets: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def _one_hot(self, fine_targets: torch.Tensor) -> torch.Tensor:
        coarse = coarse_of(fine_targets)
        return F.one_hot(coarse, self.num_coarse).float()


@COARSE_SOURCES.register("none")
class NoneSource(CoarseSource):
    def get_coarse_dist(self, images, fine_targets):
        return None


@COARSE_SOURCES.register("oracle")
class OracleSource(CoarseSource):
    """Ground-truth superclass as a one-hot distribution."""

    def get_coarse_dist(self, images, fine_targets):
        return self._one_hot(fine_targets)


@COARSE_SOURCES.register("soft")
class SoftSource(CoarseSource):
    """dist = (1 - smooth) * one_hot(true) + smooth * uniform.
    smooth=0 -> oracle; smooth=1 -> uniform (no information)."""

    def __init__(self, smooth: float = 0.3):
        self.smooth = float(smooth)

    def get_coarse_dist(self, images, fine_targets):
        one_hot = self._one_hot(fine_targets)
        uniform = torch.full_like(one_hot, 1.0 / self.num_coarse)
        return (1.0 - self.smooth) * one_hot + self.smooth * uniform


@COARSE_SOURCES.register("trained")
class TrainedSource(CoarseSource):
    """Softmax of a separate, frozen coarse CNN run on the input image."""

    def __init__(self, coarse_net: torch.nn.Module):
        self.coarse_net = coarse_net
        self.coarse_net.eval()
        for p in self.coarse_net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_coarse_dist(self, images, fine_targets):
        logits = self.coarse_net(images)
        return F.softmax(logits, dim=1)


def build_coarse_source(cfg: Config, device: torch.device) -> CoarseSource:
    """Instantiate the coarse source named by cfg.coarse.source."""
    source = cfg.coarse.source
    if source == "soft":
        return SoftSource(smooth=cfg.coarse.soft.smooth)
    if source == "trained":
        from ..models.coarse_net import build_coarse_net
        net = build_coarse_net()
        state = torch.load(cfg.coarse.trained.ckpt, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        net.to(device)
        return TrainedSource(net)
    return COARSE_SOURCES.create(source)  # none | oracle
