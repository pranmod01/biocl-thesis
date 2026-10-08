"""Few-shot evaluation of the frozen transfer-pretrained nets.

Run:  python -m coarsecl.transfer.evaluate --config configs/transfer/eval_val.yaml
      [--only pre_coarse]   # restrict to one condition (parallel jobs, same config)

For every (condition, fold, seed) checkpoint: one forward pass over all 60k images
collects avg-pooled features at each tap; then every episode (episodes.py) is read
out with NCM and a linear probe per probe_wd. Accuracy is averaged over the
`episodes` support draws of each class set, so one row = one class set at one
(tap, readout, k). The bootstrap in the analysis resamples class sets and folds,
which is the level the design pairs at; episodes are only there to average out
support-draw noise.

Writes results/<run.name>/units/<condition>_f<fold>_s<seed>.csv (skipped if it
exists, so a run is resumable and can be split across jobs) plus a .json of the
checkpoint's pretraining metadata, and the resolved config.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import dump_config, load_config
from ..data.cifar100 import CIFAR100_MEAN, CIFAR100_STD, get_datasets
from ..models.coarse_net import build_coarse_net, load_coarse_net
from ..utils import resolve_device
from .episodes import ARMS, TASKS, class_sets, draw_support, image_pools, queries
from .folds import build_fold
from .readouts import cl2n, ncm_accuracy, probe_accuracy

MAX_ROWS = 400_000  # episode-batch size cap: E * n * (k + q) gathered feature rows


def load_images(data_root: str):
    """All 60k images (train then test), test-transform normalized, and their labels."""
    train, test = get_datasets(data_root)
    x = torch.from_numpy(np.concatenate([train.data, test.data])).permute(0, 3, 1, 2)
    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return (x.float() / 255 - mean) / std, train.targets, test.targets


@torch.no_grad()
def extract(net, images: torch.Tensor, taps: List[str], device,
            batch: int = 1000) -> Dict[str, torch.Tensor]:
    """{tap: [N, C]} avg-pooled features of a ResNetCoarseNet's backbone, one pass."""
    bb = net.backbone.eval()
    out = {t: [] for t in taps}
    for i in range(0, len(images), batch):
        h = F.relu(bb.bn1(bb.conv1(images[i:i + batch].to(device))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(bb, name)(h)
            if name in out:
                out[name].append(F.adaptive_avg_pool2d(h, 1).flatten(1))
    return {t: torch.cat(v) for t, v in out.items()}


def build_net(condition: str, path: str, seed: int, device):
    """(net, pretraining metadata). scratch = seeded random init, no checkpoint."""
    if condition == "scratch":
        torch.manual_seed(seed)
        return build_coarse_net(15, arch="resnet18").to(device), {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("arch") != "resnet18":
        raise ValueError(f"{path}: taps assume the resnet18 coarse net, got {state.get('arch')}")
    meta = {k: v for k, v in state.get("transfer", {}).items() if k != "labels"}
    meta["test_acc"] = state.get("test_acc")
    return load_coarse_net(path, device), meta


def evaluate_checkpoint(feats: Dict[str, torch.Tensor], fold, pools, te) -> List[dict]:
    rows = []
    for tap, f in feats.items():
        f = cl2n(f, f[[i for c in fold.pretrain_classes for i in pools.support[c]]
                      ].mean(0))
        for task in TASKS:
            for k in te.ks:
                # (arm, class set, rep) of every episode at this task and k
                eps = [(arm, cs, r) for arm in ARMS
                       for cs in class_sets(fold, task, arm, te.cross_sets, te.episode_seed)
                       for r in range(te.episodes)]
                n = len(eps[0][1])
                sup = np.stack([draw_support(pools, cs, k, r, te.episode_seed)
                                for _, cs, r in eps])
                qry = np.stack([queries(pools, cs) for _, cs, _ in eps])
                chunk = max(1, MAX_ROWS // (sup.shape[1] + qry.shape[1]))
                accs = {"ncm": []}
                accs.update({f"probe_wd{wd:g}": [] for wd in te.probe_wd})
                for i in range(0, len(eps), chunk):
                    s = f[torch.from_numpy(sup[i:i + chunk]).to(f.device)]
                    q = f[torch.from_numpy(qry[i:i + chunk]).to(f.device)]
                    accs["ncm"].append(ncm_accuracy(s, q, n))
                    for wd in te.probe_wd:
                        accs[f"probe_wd{wd:g}"].append(
                            probe_accuracy(s, q, n, wd, te.probe_steps, te.probe_lr))
                for readout, a in accs.items():
                    a = torch.cat(a).cpu().numpy().reshape(-1, te.episodes)
                    for (arm, cs, _), acc in zip(eps[::te.episodes], a.mean(1)):
                        rows.append(dict(tap=tap, readout=readout, task=task, arm=arm,
                                         unit="-".join(map(str, cs)), k=k,
                                         acc=float(acc), episodes=te.episodes))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="evaluate just this condition")
    args = ap.parse_args()
    cfg = load_config(args.config)
    te = cfg.transfer_eval
    device = resolve_device(cfg.run.device)
    out_dir = os.path.join(cfg.run.output_dir, cfg.run.name)
    os.makedirs(os.path.join(out_dir, "units"), exist_ok=True)
    dump_config(cfg, os.path.join(out_dir, "config.yaml"))

    images, train_labels, test_labels = load_images(cfg.data.data_root)
    pools = image_pools(train_labels, test_labels, te.split, te.val_per_class, te.split_seed)
    conditions = [args.only] if args.only else te.conditions
    for condition in conditions:
        for seed in te.seeds:
            # scratch features don't depend on the fold: extract once per seed
            cache = None
            for fold_idx in te.folds:
                stem = os.path.join(out_dir, "units", f"{condition}_f{fold_idx}_s{seed}")
                if os.path.exists(stem + ".csv"):
                    print(f"[eval] {stem}.csv exists, skipping")
                    continue
                fold = build_fold(fold_idx)
                if condition != "scratch" or cache is None:
                    net, meta = build_net(condition, te.ckpt.format(
                        condition=condition, fold=fold_idx, seed=seed), seed, device)
                    cache = extract(net, images, te.taps, device)
                    del net
                rows = evaluate_checkpoint(cache, fold, pools, te)
                df = pd.DataFrame(rows).assign(condition=condition, fold=fold_idx, seed=seed)
                df.to_csv(stem + ".csv", index=False)
                with open(stem + ".json", "w") as fh:
                    json.dump(meta, fh, indent=1)
                summary = df[(df.task == "sibling") & (df.k == te.ks[0])].groupby(
                    ["tap", "readout", "arm"]).acc.mean().unstack("arm")
                print(f"[eval] {condition} f{fold_idx} s{seed}: sibling k={te.ks[0]}\n"
                      f"{summary.round(3)}", flush=True)


if __name__ == "__main__":
    main()
