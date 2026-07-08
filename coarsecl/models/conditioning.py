"""Conditioning strategies: how the coarse distribution enters the fine classifier.

Each strategy owns the fine head (and any auxiliary head), plus the loss. This is
where the input-vs-target asymmetry lives:
  - concat:   the 20-d distribution is projected and concatenated as an INPUT.
  - aux_loss: the 20-d distribution is a soft TARGET for an auxiliary head.
  - none:     baseline; no coarse signal at all.
All consume the same `coarse_dist` (FloatTensor[B, 20] or None), so corrupting the
signal upstream affects every mechanism identically.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_COARSE, NUM_FINE
from ..registry import Registry

CONDITIONING: "Registry[ConditioningStrategy]" = Registry("conditioning")


def masked_fine_logits(logits: torch.Tensor, seen_mask: torch.Tensor) -> torch.Tensor:
    """Set logits of not-yet-seen classes to -inf (Class-IL over the seen set)."""
    return logits.masked_fill(~seen_mask.to(logits.device), float("-inf"))


class ConditioningStrategy(nn.Module):
    def __init__(self, feature_dim: int, num_fine: int = NUM_FINE,
                 num_coarse: int = NUM_COARSE, **kwargs) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_fine = num_fine
        self.num_coarse = num_coarse

    def forward(self, features: torch.Tensor,
                coarse_dist: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_loss(self, outputs: Dict[str, torch.Tensor], fine_targets: torch.Tensor,
                     coarse_dist: Optional[torch.Tensor], seen_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_logits(self, outputs: Dict[str, torch.Tensor],
                    coarse_dist: Optional[torch.Tensor]) -> torch.Tensor:
        """Fine logits used for the Class-IL prediction at eval. Default: the raw
        fine head. `gate` overrides this to fold the coarse signal in as a prior."""
        return outputs["fine_logits"]


@CONDITIONING.register("none")
class NoneConditioning(ConditioningStrategy):
    """Baseline: plain fine classifier; the coarse signal is ignored entirely."""

    def __init__(self, feature_dim, num_fine=NUM_FINE, num_coarse=NUM_COARSE, **kwargs):
        super().__init__(feature_dim, num_fine, num_coarse)
        self.fine_head = nn.Linear(feature_dim, num_fine)

    def forward(self, features, coarse_dist=None):
        return {"fine_logits": self.fine_head(features)}

    def compute_loss(self, outputs, fine_targets, coarse_dist, seen_mask):
        logits = masked_fine_logits(outputs["fine_logits"], seen_mask)
        return F.cross_entropy(logits, fine_targets)


@CONDITIONING.register("concat")
class ConcatConditioning(ConditioningStrategy):
    """Project the 20-d distribution through a learned embedding matrix and
    concatenate it to the penultimate features before the fine head."""

    def __init__(self, feature_dim, num_fine=NUM_FINE, num_coarse=NUM_COARSE,
                 embed_dim: int = 32, **kwargs):
        super().__init__(feature_dim, num_fine, num_coarse)
        self.embed_dim = embed_dim
        # dist @ E, with E = coarse_embed.weight.T; a one-hot recovers a lookup.
        self.coarse_embed = nn.Linear(num_coarse, embed_dim, bias=False)
        self.fine_head = nn.Linear(feature_dim + embed_dim, num_fine)

    def forward(self, features, coarse_dist):
        if coarse_dist is None:
            raise ValueError("concat conditioning requires a coarse distribution")
        emb = self.coarse_embed(coarse_dist)
        h = torch.cat([features, emb], dim=1)
        return {"fine_logits": self.fine_head(h)}

    def compute_loss(self, outputs, fine_targets, coarse_dist, seen_mask):
        logits = masked_fine_logits(outputs["fine_logits"], seen_mask)
        return F.cross_entropy(logits, fine_targets)


@CONDITIONING.register("aux_loss")
class AuxLossConditioning(ConditioningStrategy):
    """Auxiliary coarse head off the penultimate features; the distribution is a
    soft target. The coarse signal shapes representations via the loss only and is
    not needed as an input at inference time."""

    def __init__(self, feature_dim, num_fine=NUM_FINE, num_coarse=NUM_COARSE,
                 aux_loss_weight: float = 0.5, **kwargs):
        super().__init__(feature_dim, num_fine, num_coarse)
        self.aux_loss_weight = aux_loss_weight
        self.fine_head = nn.Linear(feature_dim, num_fine)
        self.coarse_head = nn.Linear(feature_dim, num_coarse)

    def forward(self, features, coarse_dist=None):
        return {
            "fine_logits": self.fine_head(features),
            "coarse_logits": self.coarse_head(features),
        }

    def compute_loss(self, outputs, fine_targets, coarse_dist, seen_mask):
        logits = masked_fine_logits(outputs["fine_logits"], seen_mask)
        loss = F.cross_entropy(logits, fine_targets)
        if coarse_dist is not None:
            log_p = F.log_softmax(outputs["coarse_logits"], dim=1)
            aux = -(coarse_dist * log_p).sum(dim=1).mean()  # soft-target CE
            loss = loss + self.aux_loss_weight * aux
        return loss


@CONDITIONING.register("gate")
class GateConditioning(ConditioningStrategy):
    """The coarse signal is used ONLY at inference, as a multiplicative prior on the
    fine output — never as a training input. The fine head is trained plainly (like
    `none`); at eval we add log P(superclass) to each fine logit, grouped by the
    fine->coarse map. A one-hot (oracle) becomes a hard mask to the true superclass;
    a soft/trained distribution becomes a graded prior; uniform is a no-op.

    This gives the coarse scale STRUCTURE (it constrains the output space) rather than
    being a plastic input the model must learn — and re-learn — to exploit."""

    def __init__(self, feature_dim, num_fine=NUM_FINE, num_coarse=NUM_COARSE,
                 gate_eps: float = 1e-8, **kwargs):
        super().__init__(feature_dim, num_fine, num_coarse)
        self.gate_eps = gate_eps
        self.fine_head = nn.Linear(feature_dim, num_fine)
        # fine(0..99) -> coarse(0..19); a buffer so it moves with .to(device).
        from ..data.cifar100 import fine_to_coarse_tensor
        self.register_buffer("fine_to_coarse", fine_to_coarse_tensor(), persistent=False)

    def forward(self, features, coarse_dist=None):
        return {"fine_logits": self.fine_head(features)}

    def compute_loss(self, outputs, fine_targets, coarse_dist, seen_mask):
        logits = masked_fine_logits(outputs["fine_logits"], seen_mask)
        return F.cross_entropy(logits, fine_targets)

    def eval_logits(self, outputs, coarse_dist):
        logits = outputs["fine_logits"]
        if coarse_dist is None:
            return logits
        log_coarse = torch.log(coarse_dist.clamp_min(self.gate_eps))  # [B, 20]
        log_prior = log_coarse[:, self.fine_to_coarse]                # [B, 100]
        return logits + log_prior


def build_conditioning(mechanism: str, feature_dim: int, cond_cfg) -> ConditioningStrategy:
    """Instantiate a strategy from config. `cond_cfg` is the CondCfg dataclass."""
    return CONDITIONING.create(
        mechanism,
        feature_dim=feature_dim,
        embed_dim=cond_cfg.embed_dim,
        aux_loss_weight=cond_cfg.aux_loss_weight,
    )
