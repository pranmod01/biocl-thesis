"""Standard continual-learning machinery: the third orthogonal config axis.

Until now every condition ran with no anti-forgetting mechanism at all, which put
the baseline at the degenerate floor (old-task accuracy exactly 0.000 — the model
predicts only the newest task). Nothing can be measured against a hard zero, so
these methods exist to lift the substrate off the floor; the coarse-signal
question then becomes "does coarse add anything ON TOP of standard CL machinery".

Two hooks, called from the training loop, are enough for both methods:
  - extra_loss(...)  per-step additive term (replay CE / EWC quadratic penalty)
  - after_task(...)  task boundary (fill the buffer / estimate Fisher)

`cl.methods` is a list, so `[replay, ewc]` composes without any special case.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..config import NUM_FINE, Config
from ..registry import Registry
from ..utils import seen_mask_from_classes

CL_METHODS: "Registry[CLMethod]" = Registry("cl_method")


class CLMethod:
    """Both hooks default to no-ops so a method implements only what it needs."""

    def extra_loss(self, model, coarse_source, seen_mask, device) -> Optional[torch.Tensor]:
        return None

    def after_task(self, model, task_id: int, stream, coarse_source, device) -> None:
        pass


@CL_METHODS.register("replay")
class ReplayCL(CLMethod):
    """Experience replay from a class-balanced exemplar buffer.

    The buffer holds *dataset indices*, not image tensors: memory is negligible and
    exemplars keep going through the same augmentation pipeline as fresh data (a
    stored-tensor buffer would replay the same fixed crops every epoch and overfit
    them). `buffer_size` is the total exemplar count, split evenly over all classes
    seen so far, so it stays a fixed memory budget as the class set grows.

    The replay CE is always masked to the full SEEN set regardless of
    `train.loss_scope` — old classes have to be valid targets for replay to mean
    anything. The coarse signal is recomputed for replay batches by the same
    coarse_source, so oracle/soft/trained all compose for free.
    """

    def __init__(self, buffer_size: int = 2000, batch_size: int = 32,
                 weight: float = 1.0, num_workers: int = 4, seed: int = 0) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.weight = weight
        self.num_workers = num_workers
        self.rng = np.random.RandomState(seed)
        self.indices: List[int] = []
        self._iter = None

    def extra_loss(self, model, coarse_source, seen_mask, device):
        if self._iter is None:  # nothing buffered yet (during task 0)
            return None
        images, fine = next(self._iter)
        images = images.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        coarse_dist = coarse_source.get_coarse_dist(images, fine)
        # `seen_mask` is the CURRENT mask (old + current-task classes), not the one
        # cached when the buffer was filled. This is the whole point of replay: an
        # old exemplar must compete against the new task's classes in one softmax,
        # which is what pushes the inflated new-class logits back down. Masking to
        # old classes only lets the replay loss be satisfied without ever contesting
        # the new classes, and old-task accuracy stays at exactly 0.
        loss = model.compute_loss(images, fine, coarse_dist, seen_mask)
        return self.weight * loss

    def after_task(self, model, task_id, stream, coarse_source, device):
        seen = stream.seen_classes(task_id)
        per_class = max(1, self.buffer_size // len(seen))
        # Re-sample the whole buffer from scratch each boundary: with a fixed budget
        # split evenly, old classes shed exemplars as new ones arrive, which is the
        # same accounting as herding-based methods minus the herding.
        self.indices = []
        for c in seen:
            pool = stream.train_indices_of_class(c)
            take = min(per_class, len(pool))
            self.indices.extend(self.rng.choice(pool, size=take, replace=False).tolist())

        loader = DataLoader(Subset(stream.train_set, self.indices),
                            batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, drop_last=True)
        self._iter = _cycle(loader)


@CL_METHODS.register("ewc")
class EWCCL(CLMethod):
    """Elastic Weight Consolidation (diagonal, online-accumulated Fisher).

    After each task we estimate the diagonal empirical Fisher on that task's data
    and snapshot the parameters; the penalty is lambda/2 * sum F_i (theta_i - theta*_i)^2.
    Fisher is accumulated across tasks (one running sum, one anchor) rather than
    kept per-task, which is the standard memory-bounded variant.

    Expect this to do little here: EWC constrains WEIGHT DRIFT, and the frozen-
    feature experiment already showed the forgetting in this setup is head-level
    inter-task discrimination, not feature drift. It is included as a documented
    control, not a hoped-for fix.
    """

    def __init__(self, ewc_lambda: float = 5000.0, fisher_batches: int = 32,
                 batch_size: int = 128, num_workers: int = 4) -> None:
        self.ewc_lambda = ewc_lambda
        self.fisher_batches = fisher_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fisher: dict = {}
        self.anchor: dict = {}

    def extra_loss(self, model, coarse_source, seen_mask, device):
        if not self.fisher:
            return None
        penalty = None
        for name, param in model.named_parameters():
            if name not in self.fisher or not param.requires_grad:
                continue
            term = (self.fisher[name] * (param - self.anchor[name]).pow(2)).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return None
        return 0.5 * self.ewc_lambda * penalty

    def after_task(self, model, task_id, stream, coarse_source, device):
        seen_mask = seen_mask_from_classes(stream.seen_classes(task_id), NUM_FINE, device)
        loader = DataLoader(stream.train_subset(task_id), batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers, drop_last=True)
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}

        was_training = model.training
        model.train()
        batches = 0
        for images, fine in loader:
            if batches >= self.fisher_batches:
                break
            images = images.to(device)
            fine = fine.to(device)
            coarse_dist = coarse_source.get_coarse_dist(images, fine)
            model.zero_grad(set_to_none=True)
            loss = model.compute_loss(images, fine, coarse_dist, seen_mask)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            batches += 1
        model.zero_grad(set_to_none=True)
        model.train(was_training)

        for name, val in fisher.items():
            val /= max(batches, 1)
            self.fisher[name] = self.fisher.get(name, torch.zeros_like(val)) + val
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}


def _cycle(loader):
    """Endless iterator over a DataLoader (replay is sampled per step, not per epoch)."""
    while True:
        for batch in loader:
            yield batch


def build_cl_methods(cfg: Config) -> List[CLMethod]:
    """Instantiate the methods named by cfg.cl.methods (empty list = no machinery)."""
    methods: List[CLMethod] = []
    for name in cfg.cl.methods:
        if name == "replay":
            methods.append(ReplayCL(buffer_size=cfg.cl.replay.buffer_size,
                                    batch_size=cfg.cl.replay.batch_size,
                                    weight=cfg.cl.replay.weight,
                                    num_workers=cfg.data.num_workers,
                                    seed=cfg.run.seed))
        elif name == "ewc":
            methods.append(EWCCL(ewc_lambda=cfg.cl.ewc.ewc_lambda,
                                 fisher_batches=cfg.cl.ewc.fisher_batches,
                                 batch_size=cfg.train.batch_size,
                                 num_workers=cfg.data.num_workers))
        else:  # unreachable: validate() checks against the registry first
            raise ValueError(f"unknown cl method {name!r}")
    return methods
