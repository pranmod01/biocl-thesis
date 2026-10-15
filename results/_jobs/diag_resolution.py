"""Premise check for the resolution-curriculum idea: does low resolution keep
superclass information better than fine information?

One joint (non-continual) run of the fine backbone at one input resolution, with fine
(100-way) or coarse (20-way) labels. Images are downsampled to `--res` and upsampled
back to 32, at train AND test, so the architecture is identical across resolutions and
only the information content changes. Same recipe as diag_ceiling.py's from-scratch
ceiling (80 ep, SGD lr 0.1, cosine), so the res=32 fine run should land near 0.77.

Saves results/diag_resolution/<labels>_r<res>_s<seed>.json: args, test accuracy,
the test confusion matrix (rows = true, cols = predicted) and, for fine labels, the
superclass accuracy of the collapsed fine predictions. diag_resolution_analyze.py
reads them.

Run: python results/_jobs/diag_resolution.py --res 8 --labels fine --seed 0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coarsecl.data.cifar100 import coarse_of, get_datasets
from coarsecl.models.backbone import ResNetBackbone
from coarsecl.utils import resolve_device, set_seed


def degrade(x: torch.Tensor, res: int) -> torch.Tensor:
    """Down to res x res (antialiased, so it is a low-pass filter, not aliasing), back
    up to the input size. A no-op at full resolution."""
    size = x.shape[-1]
    if res >= size:
        return x
    small = F.interpolate(x, size=res, mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(small, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(bb, head, loader, res, labels, num_out, dev):
    bb.eval(); head.eval()
    conf = torch.zeros(num_out, num_out, dtype=torch.long)
    super_correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(dev), y.to(dev)
        pred = head(bb(degrade(x, res))).argmax(1)
        target = coarse_of(y) if labels == "coarse" else y
        conf += torch.bincount(target * num_out + pred, minlength=num_out * num_out
                               ).view(num_out, num_out).cpu()
        if labels == "fine":
            super_correct += (coarse_of(pred) == coarse_of(y)).sum().item()
        total += y.numel()
    acc = conf.diag().sum().item() / total
    super_acc = super_correct / total if labels == "fine" else None
    return acc, super_acc, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True, help="8 | 16 | 32")
    ap.add_argument("--labels", choices=["fine", "coarse"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke: cap batches/epoch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="results/diag_resolution")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = resolve_device(args.device)
    train_set, test_set = get_datasets(args.data_root)
    tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers)
    vl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)

    num_out = 20 if args.labels == "coarse" else 100
    bb = ResNetBackbone(feature_dim=512).to(dev)
    head = nn.Linear(512, num_out).to(dev)
    opt = torch.optim.SGD(list(bb.parameters()) + list(head.parameters()), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tag = f"{args.labels}_r{args.res}_s{args.seed}"
    t0 = time.time()
    for ep in range(args.epochs):
        bb.train(); head.train()
        correct = seen = 0
        for b, (x, y, _) in enumerate(tl):
            if args.max_batches and b >= args.max_batches:
                break
            x, y = x.to(dev), y.to(dev)
            target = coarse_of(y) if args.labels == "coarse" else y
            logits = head(bb(degrade(x, args.res)))
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == target).sum().item()
            seen += y.numel()
        sched.step()
        if ep % 10 == 9 or ep == args.epochs - 1:
            print(f"[{tag}] ep {ep + 1}/{args.epochs} train_acc={correct / max(seen, 1):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    train_acc = correct / max(seen, 1)

    acc, super_acc, conf = evaluate(bb, head, vl, args.res, args.labels, num_out, dev)
    note = f"  super_acc={super_acc:.4f}" if super_acc is not None else ""
    print(f"[RES] {tag}  test_acc={acc:.4f}{note}  last_epoch_train_acc={train_acc:.4f}",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as fh:
        json.dump({"args": vars(args), "test_acc": acc, "super_acc": super_acc,
                   "last_epoch_train_acc": train_acc, "confusion": conf.tolist()}, fh)


if __name__ == "__main__":
    main()
