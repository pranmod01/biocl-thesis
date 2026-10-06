"""Train the condition-(d) coarse classifier once, upfront, and save a checkpoint.

Run:  python -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml

Trains CoarseNet on the full CIFAR-100 train set using superclass labels, then
saves to cfg.coarse.trained.ckpt. With coarse.trained.crossfit = K, run once per
`--fold k` (k < K): each net trains on the train images with uid % K != k and is
saved to ckpt.format(fold=k).

Transfer experiment (coarsecl/transfer/folds.py): `--split-fold k` trains only on
fold k's 45 pretrain classes, labelled by coarse_train.label_mode (15 known
superclasses, optionally shuffled, or 45-way fine), and saves to
ckpt.format(fold=k, seed=run.seed). test_acc is then accuracy on the test images of
those 45 classes under the net's own labels. The continual variant (cfg.coarse.trained.
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

from ..config import NUM_COARSE, load_config
from ..data.cifar100 import FINE_TO_COARSE, fine_to_coarse_tensor, get_datasets
from ..models.coarse_net import build_coarse_net
from ..transfer.folds import build_fold, pretrain_labels, sibling_agreement
from ..utils import resolve_device, set_seed


def evaluate_coarse(net, loader, device, table=None) -> float:
    """Accuracy against table[fine] (default: the 20 superclasses)."""
    table = (fine_to_coarse_tensor() if table is None else table).to(device)
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, fine, _ in loader:
            images = images.to(device)
            target = table[fine.to(device)]
            pred = net(images).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()
    return correct / max(total, 1)


def label_table(labels: dict) -> torch.Tensor:
    """LongTensor[100]: fine class -> label, -1 for classes outside `labels`."""
    table = torch.full((len(FINE_TO_COARSE),), -1, dtype=torch.long)
    for c, lab in labels.items():
        table[c] = lab
    return table


def class_subset(ds, classes, transform=None) -> Subset:
    keep = set(classes)
    if transform is not None:
        ds = copy.copy(ds)
        ds.transform = transform
    return Subset(ds, [i for i, t in enumerate(ds.targets) if t in keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-fitting: train the net that holds out fold k "
                         "(needs coarse.trained.crossfit > 0)")
    ap.add_argument("--split-fold", type=int, default=None,
                    help="transfer experiment: pretrain on fold k's 45 pretrain classes")
    ap.add_argument("--seed", type=int, default=None, help="override run.seed")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base,
                      overrides=None if args.seed is None else {"run": {"seed": args.seed}})
    K = cfg.coarse.trained.crossfit
    if (args.fold is None) != (K == 0):
        raise SystemExit("--fold is required exactly when coarse.trained.crossfit > 0")
    split = args.split_fold is not None
    if split and K:
        raise SystemExit("--split-fold and coarse.trained.crossfit are mutually exclusive")
    if split and "{fold}" not in cfg.coarse.trained.ckpt:
        raise SystemExit("--split-fold needs a '{fold}' placeholder in coarse.trained.ckpt")
    if not split and (cfg.coarse_train.label_mode != "coarse" or cfg.coarse_train.shuffle_frac):
        raise SystemExit("label_mode/shuffle_frac other than plain coarse need --split-fold")
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    tc = cfg.coarse_train

    full_train, full_test = get_datasets(cfg.data.data_root)
    train_set, test_set = full_train, full_test
    held_loader = None
    table, num_out = fine_to_coarse_tensor(), NUM_COARSE
    if split:
        fold = build_fold(args.split_fold)
        labels = pretrain_labels(fold, tc.label_mode, tc.shuffle_frac, seed=cfg.run.seed)
        table, num_out = label_table(labels), len(set(labels.values()))
        train_set = class_subset(full_train, fold.pretrain_classes)
        test_set = class_subset(full_test, fold.pretrain_classes)
        print(f"[coarse] fold {fold.index}: {len(fold.pretrain_classes)} pretrain classes, "
              f"{tc.label_mode} labels ({num_out}-way), shuffle_frac={tc.shuffle_frac}, "
              f"sibling agreement={sibling_agreement(labels):.3f}")
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

    net = build_coarse_net(num_out, arch=tc.arch).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=tc.lr, momentum=tc.momentum,
                          weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)

    for epoch in range(tc.epochs):
        net.train()
        for images, fine, _ in train_loader:
            images = images.to(device)
            target = table.to(device)[fine.to(device)]
            opt.zero_grad()
            loss = F.cross_entropy(net(images), target)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate_coarse(net, test_loader, device, table)
        print(f"[coarse] epoch {epoch + 1}/{tc.epochs}  test_acc={acc:.4f}")

    state = {"model": net.state_dict(), "arch": tc.arch, "num_out": num_out, "test_acc": acc}
    ckpt = cfg.coarse.trained.ckpt
    if split:
        def acc_on(ds, tbl):
            return evaluate_coarse(net, DataLoader(ds, batch_size=256,
                                   num_workers=cfg.data.num_workers), device, tbl)

        # train_acc next to test_acc shows how much each label scheme had to memorize.
        report = dict(fold=fold.index, seed=cfg.run.seed, label_mode=tc.label_mode,
                      shuffle_frac=tc.shuffle_frac,
                      sibling_agreement=sibling_agreement(labels),
                      train_acc=acc_on(class_subset(full_train, fold.pretrain_classes,
                                                    transform=full_test.transform), table))
        if tc.label_mode == "coarse":
            # Do the learned groups generalize to new members? Known-arm test images
            # scored against their true superclass's group index (the unshuffled identity).
            group = {s: i for i, s in enumerate(fold.known_superclasses)}
            report["heldout_super_acc"] = acc_on(
                class_subset(full_test, fold.known_classes),
                label_table({c: group[FINE_TO_COARSE[c]] for c in fold.known_classes}))
        print("[coarse] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in report.items()))
        state["transfer"] = dict(report, labels=labels)
        ckpt = ckpt.format(fold=fold.index, seed=cfg.run.seed)
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
