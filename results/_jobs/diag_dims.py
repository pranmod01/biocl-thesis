"""Print the tensor shape at every tap point and the input width of every fine head,
so "where does the coarse signal enter, and how big is it there" is read off the
code rather than remembered. CPU, random weights, no data.

Run: python results/_jobs/diag_dims.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from coarsecl.config import NUM_COARSE
from coarsecl.models.backbone import ResNetBackbone, _FROZEN_TAPS
from coarsecl.models.coarse_net import CoarseNet
from coarsecl.models.conditioning import CONDITIONING

x = torch.zeros(2, 3, 32, 32)
torch.set_grad_enabled(False)

print("== fine backbone (ResNet-18), spatial taps ==")
bb = ResNetBackbone(feature_dim=512).eval()
for tap in ResNetBackbone.TAP_CHANNELS:
    t = bb.forward_tap(x, tap)
    print(f"  {tap:8s} {tuple(t.shape[1:])}  = {t[0].numel():6d} values")
feat = bb(x)
print(f"  pooled   {tuple(feat.shape[1:])}  <- input to every fine head")

print("\n== frozen coarse taps (small CNN, flattened, not pooled) ==")
cnet = CoarseNet().eval()
for name, upto in _FROZEN_TAPS.items():
    t = torch.nn.Sequential(*list(cnet.features)[: upto + 1])(x)
    print(f"  {name:22s} {tuple(t.shape[1:])}  -> head input {t[0].numel()}")
print(f"  coarse head input (small CNN): {cnet.head.in_features}")

print("\n== where the coarse signal enters each mechanism (fine head on 512-d) ==")
coarse = torch.full((2, NUM_COARSE), 1.0 / NUM_COARSE)
for mech in ("none", "concat", "aux_loss", "gate", "decay"):
    strat = CONDITIONING.create(mech, feature_dim=512, embed_dim=32,
                                aux_loss_weight=1.0, decay_weight=1.0).eval()
    out = strat(feat, coarse)
    heads = {n: m for n, m in strat.named_modules() if isinstance(m, torch.nn.Linear)}
    desc = ", ".join(f"{n}: {m.in_features}->{m.out_features}" for n, m in heads.items())
    print(f"  {mech:8s} {desc}")
    print(f"           outputs: {{{', '.join(f'{k}: {tuple(v.shape[1:])}' for k, v in out.items())}}}")
