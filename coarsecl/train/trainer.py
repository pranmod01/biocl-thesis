"""The continual training loop — generic across all four conditions.

It names no concrete coarse source, conditioning mechanism or CL method: all three
arrive already built from config. A fresh optimizer per task; by default only the
current task's data is seen (older classes act as negatives through the masked
cross-entropy), unless `cl.methods` adds replay/EWC on top.
Returns the lower-triangular accuracy matrix A[i][j] = acc on task j after task i.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import NUM_FINE, Config
from ..coarse.sources import build_coarse_source
from ..data.cifar100 import get_datasets
from ..data.task_stream import TaskStream
from ..models.classifier import build_classifier
from ..models.rotation import lambda_schedule, rotate_batch
from ..utils import resolve_device, seen_mask_from_classes, set_seed
from .cl_methods import build_cl_methods
from .evaluate import evaluate_task


def _build_optimizer(params, cfg: Config):
    tc = cfg.train
    if tc.optimizer == "sgd":
        return torch.optim.SGD(params, lr=tc.lr, momentum=tc.momentum,
                               weight_decay=tc.weight_decay)
    if tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    raise ValueError(f"unknown optimizer {tc.optimizer!r}")


def run_continual(cfg: Config, return_model: bool = False):
    """Run the continual loop and return the accuracy matrix. If return_model, also
    return (model, stream, device) so diagnostics can probe the trained weights."""
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)

    train_set, test_set = get_datasets(cfg.data.data_root)
    stream = TaskStream(train_set, test_set, cfg.data.num_tasks,
                        cfg.data.classes_per_task, cfg.data.order_seed)
    coarse_source = build_coarse_source(cfg, device)
    model = build_classifier(cfg).to(device)
    cl_methods = build_cl_methods(cfg)

    acc_matrix: List[List[float]] = []

    for task_id in range(cfg.data.num_tasks):
        seen = stream.seen_classes(task_id)
        seen_mask = seen_mask_from_classes(seen, NUM_FINE, device)
        # CE masking scope: full seen set (default) or current-task classes only,
        # the latter keeping old-class logits from being suppressed as negatives.
        if cfg.train.loss_scope == "current":
            loss_mask = seen_mask_from_classes(stream.task_classes[task_id], NUM_FINE, device)
        else:
            loss_mask = seen_mask
        loader = DataLoader(stream.train_subset(task_id), batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=cfg.data.num_workers, drop_last=False)
        optimizer = _build_optimizer(model.parameters(), cfg)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs_per_task)

        rot = cfg.rotation
        total_steps = cfg.train.epochs_per_task * len(loader)  # for the per-task anneal
        step = 0
        rot_correct = rot_total = 0

        model.train()
        for epoch in range(cfg.train.epochs_per_task):
            for images, fine in loader:
                images = images.to(device)
                fine = fine.to(device)
                coarse_dist = coarse_source.get_coarse_dist(images, fine)
                optimizer.zero_grad()
                loss = model.compute_loss(images, fine, coarse_dist, loss_mask)
                for method in cl_methods:
                    term = method.extra_loss(model, coarse_source, seen_mask, device)
                    if term is not None:
                        loss = loss + term
                if rot.enabled:
                    rot_images, rot_labels = rotate_batch(images)
                    rot_logits = model.rotation_logits(rot_images)
                    lam = lambda_schedule(step, total_steps, rot.lambda_max,
                                          rot.schedule, task_id, cfg.data.num_tasks)
                    loss = loss + lam * F.cross_entropy(rot_logits, rot_labels)
                    rot_correct += (rot_logits.argmax(1) == rot_labels).sum().item()
                    rot_total += rot_labels.numel()
                loss.backward()
                optimizer.step()
                step += 1
            sched.step()

        # Task boundary: buffers refilled / Fisher estimated on the just-finished
        # task, before the next task's optimizer is built.
        for method in cl_methods:
            method.after_task(model, task_id, stream, coarse_source, device)

        row = [
            evaluate_task(model, stream.test_subset(j), coarse_source, seen_mask,
                          device, num_workers=cfg.data.num_workers,
                          withhold=cfg.conditioning.eval_withhold)
            for j in range(task_id + 1)
        ]
        acc_matrix.append(row)
        seen_avg = sum(row) / len(row)
        rot_note = ""
        if rot.enabled:
            # mean rotation-head train accuracy over the task (chance = 0.25); if this
            # stays near chance the aux task carries no gradient and retention numbers
            # below say nothing about internalization.
            rot_note = f"  rot_acc={rot_correct / max(rot_total, 1):.3f}"
        print(f"[{cfg.run.name}] task {task_id + 1}/{cfg.data.num_tasks} "
              f"avg_seen_acc={seen_avg:.4f}  per_task={[round(a, 3) for a in row]}{rot_note}")

    if return_model:
        return acc_matrix, model, stream, device
    return acc_matrix
