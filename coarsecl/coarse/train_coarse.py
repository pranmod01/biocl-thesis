"""Train the condition-(d) coarse classifier once, upfront, and save a checkpoint.

Run:  python -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml

Trains CoarseNet on the full CIFAR-100 train set using superclass labels, then
saves to cfg.coarse.trained.ckpt. With coarse.trained.crossfit = K, run once per
`--fold k` (k < K): each net trains on the train images with uid % K != k and is
saved to ckpt.format(fold=k). The continual variant (cfg.coarse.trained.
continual=true) is a dormant hook and intentionally unimplemented here.
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..config import load_config
from ..data.cifar100 import coarse_of, get_datasets
from ..models.coarse_net import build_coarse_net
from ..utils import resolve_device, set_seed


def evaluate_coarse(net, loader, device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, fine, _ in loader:
            images = images.to(device)
            coarse = coarse_of(fine.to(device))
            pred = net(images).argmax(dim=1)
            correct += (pred == coarse).sum().item()
            total += coarse.numel()
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-fitting: train the net that holds out fold k "
                         "(needs coarse.trained.crossfit > 0)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    K = cfg.coarse.trained.crossfit
    if (args.fold is None) != (K == 0):
        raise SystemExit("--fold is required exactly when coarse.trained.crossfit > 0")
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    tc = cfg.coarse_train

    train_set, test_set = get_datasets(cfg.data.data_root)
    held_loader = None
    if K:
        # Fold k = train uids with uid % K == k (uid == train index). This net never
        # sees fold k, so the cross-fitted source can label those images honestly.
        idx = np.arange(len(train_set))
        held = copy.copy(train_set)
        held.transform = test_set.transform
        held_loader = DataLoader(Subset(held, idx[idx % K == args.fold].tolist()),
                                 batch_size=256, shuffle=False,
                                 num_workers=cfg.data.num_workers)
        train_set = Subset(train_set, idx[idx % K != args.fold].tolist())
    train_loader = DataLoader(train_set, batch_size=tc.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=False,
                              pin_memory=True, persistent_workers=cfg.data.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=cfg.data.num_workers)

    net = build_coarse_net(arch=tc.arch).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=tc.lr, momentum=tc.momentum,
                          weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)

    for epoch in range(tc.epochs):
        net.train()
        for images, fine, _ in train_loader:
            images = images.to(device)
            coarse = coarse_of(fine.to(device))
            opt.zero_grad()
            loss = F.cross_entropy(net(images), coarse)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate_coarse(net, test_loader, device)
        print(f"[coarse] epoch {epoch + 1}/{tc.epochs}  test_acc={acc:.4f}")

    state = {"model": net.state_dict(), "arch": tc.arch, "test_acc": acc}
    ckpt = cfg.coarse.trained.ckpt
    if K:
        state.update(fold=args.fold, folds=K,
                     heldout_acc=evaluate_coarse(net, held_loader, device))
        ckpt = ckpt.format(fold=args.fold)
        print(f"[coarse] fold {args.fold}/{K} held-out train acc="
              f"{state['heldout_acc']:.4f}")
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save(state, ckpt)
    print(f"[coarse] saved checkpoint -> {ckpt}  (final test_acc={acc:.4f})")


if __name__ == "__main__":
    main()
