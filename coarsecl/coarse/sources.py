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

    def get_coarse_dist(self, images: torch.Tensor, fine_targets: torch.Tensor,
                        uids: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """`uids` are the stable per-image ids from the dataset; only sources with a
        fixed per-image property (noisy) need them."""
        raise NotImplementedError

    def _one_hot(self, fine_targets: torch.Tensor) -> torch.Tensor:
        coarse = coarse_of(fine_targets)
        return F.one_hot(coarse, self.num_coarse).float()


@COARSE_SOURCES.register("none")
class NoneSource(CoarseSource):
    def get_coarse_dist(self, images, fine_targets, uids=None):
        return None


@COARSE_SOURCES.register("oracle")
class OracleSource(CoarseSource):
    """Ground-truth superclass as a one-hot distribution."""

    def get_coarse_dist(self, images, fine_targets, uids=None):
        return self._one_hot(fine_targets)


@COARSE_SOURCES.register("soft")
class SoftSource(CoarseSource):
    """dist = (1 - smooth) * one_hot(true) + smooth * uniform.
    smooth=0 -> oracle; smooth=1 -> uniform (no information)."""

    def __init__(self, smooth: float = 0.3):
        self.smooth = float(smooth)

    def get_coarse_dist(self, images, fine_targets, uids=None):
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
    def get_coarse_dist(self, images, fine_targets, uids=None):
        logits = self.coarse_net(images)
        return F.softmax(logits, dim=1)


class CrossFitTrainedSource(CoarseSource):
    """`trained`, cross-fitted: K frozen coarse nets, and image uid goes to net
    uid % K, which never trained on it (train uid == train index; see train_coarse).
    So the signal is held-out quality on the fine net's TRAINING images too. A single
    net that memorized its train set is ~100% right there and ~85% at test, so the fine
    net would learn to trust a signal that then breaks at eval.
    Selected via coarse.source: trained + coarse.trained.crossfit > 0."""

    def __init__(self, coarse_nets):
        self.nets = coarse_nets
        for net in self.nets:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def get_coarse_dist(self, images, fine_targets, uids=None):
        if uids is None:
            raise ValueError("cross-fitted trained source needs per-image uids")
        fold = (uids % len(self.nets)).to(images.device)
        out = images.new_empty(images.size(0), self.num_coarse)
        for k, net in enumerate(self.nets):
            sel = fold == k
            if sel.any():
                out[sel] = F.softmax(net(images[sel]), dim=1)
        return out


@COARSE_SOURCES.register("noisy")
class NoisySource(CoarseSource):
    """Oracle one-hot with a controlled fraction `rate` of images given a WRONG
    superclass, chosen uniformly from the other 19.

    The corruption is fixed per image (keyed on the dataset uid), so an image's
    wrong label is the same on every epoch, every replay draw and every eval, the way a
    coarse net's mistakes stick to the images it finds hard. Re-drawing per call
    would average out to `soft` smoothing and let the fine net see the truth most epochs.

    Nested across rates: image i is corrupted iff u_i < rate, with u_i and its wrong
    offset drawn once from `seed`. Raising the rate only adds corrupted images, so a
    sweep is a coupled dose-response curve and not independent draws.
    rate=0 -> oracle; rate=19/20 -> the label is uniform over all 20 (no
    information); rate > 19/20 is anti-informative (true superclass under-represented).
    The signal's accuracy is 1 - rate, directly comparable to a coarse net's accuracy.
    """

    def __init__(self, rate: float = 0.0, seed: int = 0):
        from ..data.cifar100 import NUM_IMAGES
        self.rate = float(rate)
        g = torch.Generator().manual_seed(seed)
        self._u = torch.rand(NUM_IMAGES, generator=g)
        self._offset = torch.randint(1, self.num_coarse, (NUM_IMAGES,), generator=g)

    def observed_coarse(self, fine_targets: torch.Tensor, uids: torch.Tensor) -> torch.Tensor:
        if uids is None:
            raise ValueError("noisy source needs per-image uids (dataset items are "
                             "(image, fine, uid); pass the uid to get_coarse_dist)")
        uids = uids.cpu()
        corrupt = (self._u[uids] < self.rate).to(fine_targets.device)
        offset = self._offset[uids].to(fine_targets.device)
        coarse = coarse_of(fine_targets)
        return torch.where(corrupt, (coarse + offset) % self.num_coarse, coarse)

    def get_coarse_dist(self, images, fine_targets, uids=None):
        return F.one_hot(self.observed_coarse(fine_targets, uids), self.num_coarse).float()


def build_coarse_source(cfg: Config, device: torch.device,
                        source: Optional[str] = None) -> CoarseSource:
    """Instantiate the coarse source named by cfg.coarse.source, or by `source` when
    given (schema-guided replay builds its own schema independently of the run's)."""
    source = source or cfg.coarse.source
    if source == "soft":
        return SoftSource(smooth=cfg.coarse.soft.smooth)
    if source == "noisy":
        return NoisySource(rate=cfg.coarse.noisy.rate, seed=cfg.coarse.noisy.seed)
    if source == "trained":
        from ..models.coarse_net import load_coarse_net
        tc = cfg.coarse.trained
        if tc.crossfit:
            return CrossFitTrainedSource([load_coarse_net(tc.ckpt.format(fold=k), device)
                                          for k in range(tc.crossfit)])
        return TrainedSource(load_coarse_net(tc.ckpt, device))
    return COARSE_SOURCES.create(source)  # none | oracle
