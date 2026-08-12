"""Joint (non-continual) fine-classification ceilings — how much fine information is
reachable, before any forgetting enters the picture.

(1) from-scratch fine backbone: the absolute ceiling for this arch/data.
(2) linear probe on FROZEN coarse-net features at blocks 1/2/3: how much fine info the
    coarse (20-way) representation preserved at each depth. Tests whether a mid-stack
    tap keeps more fine detail than the coarse-specialized penultimate layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coarsecl.config import load_config
from coarsecl.data.cifar100 import get_datasets
from coarsecl.models.backbone import build_backbone
from coarsecl.models.coarse_net import build_coarse_net
from coarsecl.utils import resolve_device, set_seed

cfg = load_config("configs/d_trained.yaml", "configs/base.yaml")
set_seed(cfg.run.seed)
dev = resolve_device(cfg.run.device)
train_set, test_set = get_datasets(cfg.data.data_root)
NW = cfg.data.num_workers


def loaders():
    tl = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=NW, drop_last=False)
    vl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=NW)
    return tl, vl


def run(tag, feat_fn, feat_dim, epochs, train_modules):
    """feat_fn: images -> [B, feat_dim]. train_modules: modules whose params to optimize
    (the head, plus the backbone for the from-scratch ceiling)."""
    head = nn.Linear(feat_dim, 100).to(dev)
    params = list(head.parameters())
    for m in train_modules:
        params += list(m.parameters())
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tl, vl = loaders()
    for ep in range(epochs):
        head.train()
        for m in train_modules:
            m.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(head(feat_fn(x)), y)
            loss.backward()
            opt.step()
        sched.step()
    head.eval()
    for m in train_modules:
        m.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(dev), y.to(dev)
            c += (head(feat_fn(x)).argmax(1) == y).sum().item()
            t += y.numel()
    print(f"[CEILING] {tag:28} joint fine test_acc={c / t:.4f}  (feat_dim={feat_dim}, epochs={epochs})", flush=True)
    return c / t


# (1) from-scratch fine backbone (trains backbone + head jointly)
bb = build_backbone(cfg.model.backbone, cfg.model.feature_dim).to(dev)
run("from_scratch_backbone", lambda x: bb(x), cfg.model.feature_dim, 80, [bb])

# (2) frozen coarse-net feature probes at blocks 1/2/3
cnet = build_coarse_net().to(dev)
ckpt = torch.load(cfg.coarse.trained.ckpt, map_location=dev)
cnet.load_state_dict(ckpt["model"])
cnet.eval()
for p in cnet.parameters():
    p.requires_grad_(False)
print(f"loaded coarse net (superclass test_acc={ckpt.get('test_acc')})", flush=True)

feats = cnet.features  # nn.Sequential; block ends at MaxPool indices 6/13/17
TAPS = [("block1_frozen", 6, 32 * 16 * 16),
        ("block2_frozen", 13, 64 * 8 * 8),
        ("block3_frozen_penult", 17, 128 * 4 * 4)]


def tap_fn(upto):
    def f(x):
        with torch.no_grad():  # backbone frozen; grads only flow to the linear head
            z = x
            for i, layer in enumerate(feats):
                z = layer(z)
                if i == upto:
                    break
            return z.flatten(1)
    return f


for tag, upto, dim in TAPS:
    run(tag, tap_fn(upto), dim, 50, [])  # linear probe: only the head trains
