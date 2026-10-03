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

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset

from ..coarse.sources import build_coarse_source
from ..config import NUM_COARSE, NUM_FINE, Config
from ..data.cifar100 import fine_to_coarse_tensor
from ..models.conditioning import masked_fine_logits
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

    PRIORITY (`priority` != none) changes only WHICH exemplars are replayed, never
    what is stored. Every `priority_refresh` steps the whole buffer is scored on
    clean (non-augmented) images with the model in eval mode, and replay draws
    follow (1 - mix) * uniform + mix * score / sum(score).
      - schema:   routing error, 1 - <schema(x), P_fine(superclass | x)>. The schema
                  (the coarse system) decides what the episodic store replays. It
                  targets exemplars whose fine prediction leaves the right superclass.
      - hardness: 1 - P_fine(true class). The schema-free control: without it, a gain
                  from `schema` could be prioritized replay, not the schema.
    Both read the model's own `fine_logits` (for gate: the raw head without the
    prior; for decay: the deployed student), scored against the CURRENT seen mask.
    The sampler reads the live weights lazily, so a re-score reaches the loss after
    the DataLoader's prefetch (~prefetch_factor * num_workers batches).

    SUPERCLASS LEVEL (`priority_level: superclass`, schema only). Per-exemplar schema
    priority collapsed onto the few exemplars not yet memorized (ESS 0.40) and did
    worse than uniform. Here the scores are averaged per superclass,
    R_s = sum_j q_j(s) r_j / sum_j q_j(s), and every replay exemplar i gets weight
    sum_s q_i(s) R_s, so a superclass's weight is spread over its whole pool. The
    priority unit is then what the schema actually knows about (between-category
    structure), and replay can't narrow below one superclass's pool. Weighting by
    R_s per EXEMPLAR (not picking a superclass w.p. R_s) keeps superclasses in
    proportion to their pool size; otherwise a superclass with one seen fine class
    would soak up as much replay as one with five.

    PROBES (`priority_probe_per_class` = k > 0, superclass level only). The first k
    of each class's stored exemplars are held out: never replayed, used only to
    estimate R_s. They were trained on during their own task but not refreshed
    since, so their routing error tracks forgetting rather than buffer
    memorization. They count against `buffer_size`, so the replay pool shrinks by
    k * seen classes and no extra memory is used.
    """

    def __init__(self, buffer_size: int = 2000, batch_size: int = 32,
                 weight: float = 1.0, num_workers: int = 4, seed: int = 0,
                 priority: str = "none", schema_source=None,
                 priority_mix: float = 0.5, priority_refresh: int = 50,
                 priority_level: str = "exemplar",
                 priority_probe_per_class: int = 0) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.weight = weight
        self.num_workers = num_workers
        self.rng = np.random.RandomState(seed)
        self.indices: List[int] = []
        self._iter = None

        self.priority = priority
        self.schema_source = schema_source  # only used by priority == "schema"
        self.priority_mix = priority_mix
        self.priority_refresh = priority_refresh
        self.priority_level = priority_level
        self.probe_per_class = priority_probe_per_class
        self.probe_indices: List[int] = []  # held out from replay, scored only
        # Separate stream so buffer contents match a uniform run with the same seed.
        self.sample_rng = np.random.RandomState(seed + 1)
        self.probs: Optional[np.ndarray] = None  # replay distribution over buffer slots
        self._steps = 0
        self._score_log: List[float] = []  # mean buffer score per re-score, this task
        self.history: List[dict] = []  # one entry per task boundary (diagnostics)

    def extra_loss(self, model, coarse_source, seen_mask, device):
        if self._iter is None:  # nothing buffered yet (during task 0)
            return None
        if self.priority != "none" and self._steps % self.priority_refresh == 0:
            self._rescore(model, coarse_source, seen_mask)
        self._steps += 1
        images, fine, uid = next(self._iter)
        images = images.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        coarse_dist = coarse_source.get_coarse_dist(images, fine, uid)
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
        # Probes are carved out of the same draw, so stored = probes + replay is
        # exactly the buffer a uniform run with this seed would hold.
        self.indices, self.probe_indices = [], []
        k = self.probe_per_class
        for c in seen:
            pool = stream.train_indices_of_class(c)
            take = min(per_class, len(pool))
            chosen = self.rng.choice(pool, size=take, replace=False).tolist()
            self.probe_indices.extend(chosen[:k])
            self.indices.extend(chosen[k:])

        if self.priority == "none":
            loader = DataLoader(Subset(stream.train_set, self.indices),
                                batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, drop_last=True)
            self._iter = _cycle(loader)
            return

        self._log_task(task_id)
        self._cache_clean_buffer(stream, device)
        self.probs = np.full(len(self.indices), 1.0 / len(self.indices))
        self._steps = 0  # new buffer: re-score before the first replay step
        loader = DataLoader(Subset(stream.train_set, self.indices),
                            batch_size=self.batch_size,
                            sampler=_PrioritySampler(self, len(self.indices)),
                            num_workers=self.num_workers, drop_last=True)
        self._iter = iter(loader)  # the sampler is infinite, so no _cycle

    # -- priority replay ---------------------------------------------------
    @torch.no_grad()
    def _cache_clean_buffer(self, stream, device) -> None:
        """Images of the SCORED set (the replay buffer, or the probes) through the
        TEST transform (no crop/flip noise in the scores), plus the schema's
        distribution for each. All fixed until the next boundary (the trained coarse
        net is frozen), so they are computed once."""
        clean = copy.copy(stream.train_set)  # shallow: shares the in-memory arrays
        clean.transform = stream.test_set.transform

        def load(idx):
            xs, ys, us = zip(*DataLoader(Subset(clean, idx), batch_size=500, shuffle=False))
            return torch.cat(xs).to(device), torch.cat(ys).to(device), torch.cat(us)

        def schema(x, y, u):
            return torch.cat([self.schema_source.get_coarse_dist(xb, yb, ub)
                              for xb, yb, ub in zip(x.split(500), y.split(500), u.split(500))])

        self._score_x, self._score_y, self._score_u = load(self.probe_indices or self.indices)
        self._score_q = None
        if self.priority == "schema":
            self._score_q = schema(self._score_x, self._score_y, self._score_u)
            self._f2c = fine_to_coarse_tensor(device)
        if self.priority_level == "superclass":  # superclass membership of replay slots
            self._replay_q = (schema(*load(self.indices)) if self.probe_indices
                              else self._score_q)

    @torch.no_grad()
    def buffer_scores(self, model, coarse_source, seen_mask) -> torch.Tensor:
        """Per-exemplar priority score in [0, 1] for the scored set (the replay
        buffer, or the probes when `priority_probe_per_class` > 0)."""
        was_training = model.training
        model.eval()
        scores = []
        xs, ys, us = (self._score_x.split(500), self._score_y.split(500),
                      self._score_u.split(500))
        qs = self._score_q.split(500) if self._score_q is not None else [None] * len(xs)
        for x, y, u, q in zip(xs, ys, us, qs):
            outputs = model(x, coarse_source.get_coarse_dist(x, y, u))
            p = F.softmax(masked_fine_logits(outputs["fine_logits"], seen_mask), dim=1)
            if self.priority == "hardness":
                scores.append(1.0 - p.gather(1, y[:, None]).squeeze(1))
            else:  # schema: mass the fine model puts outside the schema's superclass
                marginal = p.new_zeros(p.size(0), NUM_COARSE).index_add_(1, self._f2c, p)
                scores.append(1.0 - (q * marginal).sum(dim=1))
        model.train(was_training)
        return torch.cat(scores).clamp_(0.0, 1.0)

    def superclass_scores(self, s: torch.Tensor) -> torch.Tensor:
        """R_s: the schema-weighted mean score of each superclass, [NUM_COARSE]."""
        q = self._score_q
        return (q * s[:, None]).sum(0) / q.sum(0).clamp_min(1e-12)

    def _rescore(self, model, coarse_source, seen_mask) -> None:
        s = self.buffer_scores(model, coarse_source, seen_mask)
        self._score_log.append(float(s.mean()))
        if self.priority_level == "superclass":
            R = self.superclass_scores(s)
            self._last_R = R
            s = (self._replay_q * R[None, :]).sum(dim=1)  # one weight per replay slot
        s = s.double().cpu().numpy()
        n = len(s)
        pri = s / s.sum() if s.sum() > 0 else np.full(n, 1.0 / n)
        self.probs = (1.0 - self.priority_mix) / n + self.priority_mix * pri

    def _log_task(self, task_id: int) -> None:
        """Summarize the task that just ended (its re-scores ran during it)."""
        if not self._score_log:  # task 0 has no replay
            return
        n = len(self.probs)
        ess = 1.0 / (n * float((self.probs ** 2).sum()))  # 1 = uniform draws
        entry = {"task": task_id, "mean_score_first": self._score_log[0],
                 "mean_score_last": self._score_log[-1], "ess_frac_last": ess}
        tag, spread = self.priority, ""
        if self.priority_level == "superclass":
            tag += "/super" + ("+probe" if self.probe_indices else "")
            R = self._last_R[self._last_R > 0]
            if len(R):
                entry["R_max_over_mean"] = float(R.max() / R.mean())
                spread = f", R max/mean {entry['R_max_over_mean']:.2f}"
        self.history.append(entry)
        scored = "probe" if self.probe_indices else "buffer"
        print(f"[replay:{tag}] task {task_id + 1}: {scored} score "
              f"{entry['mean_score_first']:.3f} -> {entry['mean_score_last']:.3f}, "
              f"ESS {ess:.2f} of buffer{spread}")
        self._score_log = []


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
        for images, fine, uid in loader:
            if batches >= self.fisher_batches:
                break
            images = images.to(device)
            fine = fine.to(device)
            coarse_dist = coarse_source.get_coarse_dist(images, fine, uid)
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


class _PrioritySampler(Sampler):
    """Infinite sampler over buffer slots that reads the owner's CURRENT `probs` at
    every batch, so a re-score takes effect without rebuilding the DataLoader."""

    def __init__(self, owner: ReplayCL, n: int) -> None:
        self.owner = owner
        self.n = n

    def __iter__(self):
        while True:
            yield from self.owner.sample_rng.choice(
                self.n, size=self.owner.batch_size, p=self.owner.probs).tolist()


def build_cl_methods(cfg: Config, device: Optional[torch.device] = None) -> List[CLMethod]:
    """Instantiate the methods named by cfg.cl.methods (empty list = no machinery)."""
    methods: List[CLMethod] = []
    for name in cfg.cl.methods:
        if name == "replay":
            rp = cfg.cl.replay
            schema_source = None
            if rp.priority == "schema":
                schema_source = build_coarse_source(
                    cfg, device or torch.device("cpu"), source=rp.priority_schema)
            methods.append(ReplayCL(buffer_size=rp.buffer_size,
                                    batch_size=rp.batch_size,
                                    weight=rp.weight,
                                    num_workers=cfg.data.num_workers,
                                    seed=cfg.run.seed,
                                    priority=rp.priority,
                                    schema_source=schema_source,
                                    priority_mix=rp.priority_mix,
                                    priority_refresh=rp.priority_refresh,
                                    priority_level=rp.priority_level,
                                    priority_probe_per_class=rp.priority_probe_per_class))
        elif name == "ewc":
            methods.append(EWCCL(ewc_lambda=cfg.cl.ewc.ewc_lambda,
                                 fisher_batches=cfg.cl.ewc.fisher_batches,
                                 batch_size=cfg.train.batch_size,
                                 num_workers=cfg.data.num_workers))
        else:  # unreachable: validate() checks against the registry first
            raise ValueError(f"unknown cl method {name!r}")
    return methods
