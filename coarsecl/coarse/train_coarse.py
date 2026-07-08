"""Train the condition-(d) coarse classifier once, upfront, and save a checkpoint.

Run:  python -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml

Trains CoarseNet on the full CIFAR-100 train set using superclass labels, then
saves to cfg.coarse.trained.ckpt. The continual variant (cfg.coarse.trained.
continual=true) is a dormant hook and intentionally unimplemented here.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.cifar100 import coarse_of, get_datasets
from ..models.coarse_net import build_coarse_net
from ..utils import resolve_device, set_seed


def evaluate_coarse(net, loader, device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, fine in loader:
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
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    tc = cfg.coarse_train

    train_set, test_set = get_datasets(cfg.data.data_root)
    train_loader = DataLoader(train_set, batch_size=tc.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=cfg.data.num_workers)

    net = build_coarse_net().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=tc.lr, momentum=tc.momentum,
                          weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)

    for epoch in range(tc.epochs):
        net.train()
        for images, fine in train_loader:
            images = images.to(device)
            coarse = coarse_of(fine.to(device))
            opt.zero_grad()
            loss = F.cross_entropy(net(images), coarse)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate_coarse(net, test_loader, device)
        print(f"[coarse] epoch {epoch + 1}/{tc.epochs}  test_acc={acc:.4f}")

    ckpt = cfg.coarse.trained.ckpt
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "test_acc": acc}, ckpt)
    print(f"[coarse] saved checkpoint -> {ckpt}  (final test_acc={acc:.4f})")


if __name__ == "__main__":
    main()
