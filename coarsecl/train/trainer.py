"""The continual training loop — generic across all four conditions.

It names no concrete coarse source or conditioning mechanism: both arrive already
built from config. No replay; a fresh optimizer per task; only the current task's
data is seen (older classes act as negatives through the masked cross-entropy).
Returns the lower-triangular accuracy matrix A[i][j] = acc on task j after task i.
"""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader

from ..config import NUM_FINE, Config
from ..coarse.sources import build_coarse_source
from ..data.cifar100 import get_datasets
from ..data.task_stream import TaskStream
from ..models.classifier import build_classifier
from ..utils import resolve_device, seen_mask_from_classes, set_seed
from .evaluate import evaluate_task


def _build_optimizer(params, cfg: Config):
    tc = cfg.train
    if tc.optimizer == "sgd":
        return torch.optim.SGD(params, lr=tc.lr, momentum=tc.momentum,
                               weight_decay=tc.weight_decay)
    if tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.lr, weight_decay=tc.weight_decay)
    raise ValueError(f"unknown optimizer {tc.optimizer!r}")


def run_continual(cfg: Config) -> List[List[float]]:
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)

    train_set, test_set = get_datasets(cfg.data.data_root)
    stream = TaskStream(train_set, test_set, cfg.data.num_tasks,
                        cfg.data.classes_per_task, cfg.data.order_seed)
    coarse_source = build_coarse_source(cfg, device)
    model = build_classifier(cfg).to(device)

    acc_matrix: List[List[float]] = []

    for task_id in range(cfg.data.num_tasks):
        seen = stream.seen_classes(task_id)
        seen_mask = seen_mask_from_classes(seen, NUM_FINE, device)
        loader = DataLoader(stream.train_subset(task_id), batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=cfg.data.num_workers, drop_last=False)
        optimizer = _build_optimizer(model.parameters(), cfg)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs_per_task)

        model.train()
        for epoch in range(cfg.train.epochs_per_task):
            for images, fine in loader:
                images = images.to(device)
                fine = fine.to(device)
                coarse_dist = coarse_source.get_coarse_dist(images, fine)
                optimizer.zero_grad()
                loss = model.compute_loss(images, fine, coarse_dist, seen_mask)
                loss.backward()
                optimizer.step()
            sched.step()

        row = [
            evaluate_task(model, stream.test_subset(j), coarse_source, seen_mask,
                          device, num_workers=cfg.data.num_workers,
                          withhold=cfg.conditioning.eval_withhold)
            for j in range(task_id + 1)
        ]
        acc_matrix.append(row)
        seen_avg = sum(row) / len(row)
        print(f"[{cfg.run.name}] task {task_id + 1}/{cfg.data.num_tasks} "
              f"avg_seen_acc={seen_avg:.4f}  per_task={[round(a, 3) for a in row]}")

    return acc_matrix
