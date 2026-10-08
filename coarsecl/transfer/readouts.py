"""Readouts on frozen features, batched over E episodes that share one layout:
support [E, n*k, D] and query [E, n*q, D], both class-major.

Features are CL2N-normalized first (SimpleShot: subtract the pretraining-class mean,
then L2-normalize), so NCM and the probe see the same geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cl2n(feats: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return F.normalize(feats - mean, dim=-1)


def _labels(n: int, per_class: int, device) -> torch.Tensor:
    return torch.arange(n, device=device).repeat_interleave(per_class)


def ncm_accuracy(support: torch.Tensor, query: torch.Tensor, n: int) -> torch.Tensor:
    """[E] accuracy of nearest class mean (Euclidean on CL2N features)."""
    E, nk, D = support.shape
    protos = support.view(E, n, nk // n, D).mean(2)
    pred = torch.cdist(query, protos).argmin(-1)
    return (pred == _labels(n, query.shape[1] // n, query.device)).float().mean(1)


def probe_accuracy(support: torch.Tensor, query: torch.Tensor, n: int, wd: float,
                   steps: int, lr: float) -> torch.Tensor:
    """[E] accuracy of an L2-regularized multinomial logistic regression fit per episode.

    One weight matrix per episode; the loss is the SUM of per-episode objectives, so
    each episode's gradient is exactly its own problem's and Adam (per-parameter)
    optimizes them independently. Zero init: deterministic, no seed needed."""
    E, nk, D = support.shape
    y = _labels(n, nk // n, support.device).repeat(E)
    W = torch.zeros(E, D, n, device=support.device, requires_grad=True)
    b = torch.zeros(E, 1, n, device=support.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            logits = torch.baddbmm(b, support, W)
            loss = F.cross_entropy(logits.reshape(-1, n), y, reduction="sum") / nk \
                + 0.5 * wd * W.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = torch.baddbmm(b, query, W).argmax(-1)
        return (pred == _labels(n, query.shape[1] // n, query.device)).float().mean(1)
