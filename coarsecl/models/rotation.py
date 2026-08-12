"""Self-supervised rotation auxiliary task (approach B).

The rotation label is generated per example at load time (not derived from the class
label), so unlike the superclass signal there is genuinely new structure for the
shared trunk to internalize. A deliberately shallow head off an intermediate trunk
tap predicts the rotation; its loss weight anneals to zero, so the coarse-flavoured
pressure is applied early and removed, leaving the trunk to carry what it absorbed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_batch(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of {0,90,180,270} degrees per example (uniform). Returns the
    rotated batch and the integer rotation labels [0..3]."""
    k = torch.randint(0, 4, (images.size(0),), device=images.device)
    out = images.clone()
    for kk in (1, 2, 3):
        m = k == kk
        if m.any():
            out[m] = torch.rot90(images[m], kk, dims=[2, 3])
    return out, k


class RotationHead(nn.Module):
    """Shallow head: GAP the spatial tap map, then near-linear to 4 rotation classes.
    Kept small on purpose — an expressive head would solve rotation with its own
    capacity and spare the shared trunk, defeating the point (internalization)."""

    def __init__(self, in_channels: int, hidden: int = 128, num_rot: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, num_rot)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # [B, C]
        return self.fc2(F.relu(self.fc1(z)))


def lambda_schedule(step: int, total: int, lambda_max: float, schedule: str,
                    task_id: int = 0, num_tasks: int = 1) -> float:
    """Rotation loss weight.
      constant:   lambda_max always — the aux task is a permanent anchor (paired
                  with a deep tap: fights feature drift by keeping a fixed objective
                  on the shared features every task, rather than fading out).
      per_task:   cosine anneal lambda_max->0 over steps within the current task.
      curriculum: cosine anneal lambda_max->0 over tasks (high on task 0)."""
    if schedule == "constant":
        return lambda_max
    if schedule == "per_task":
        frac = step / max(total, 1)
    else:  # curriculum
        frac = task_id / max(num_tasks - 1, 1)
    frac = min(frac, 1.0)
    return lambda_max * 0.5 * (1.0 + math.cos(math.pi * frac))
