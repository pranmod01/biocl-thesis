"""Assert the CL hooks actually fire, on CPU in seconds -- no GPU, no training.

Run: python tests/test_cl_methods.py

Covers the mask regression that cost a calibration job: the replay CE must score
old exemplars against the CURRENT seen set. Using the mask cached when the buffer
was filled (old classes only) keeps new-class logits out of the replay softmax, so
replay cannot contest them and old-task accuracy sits at exactly 0.000 -- looking
exactly like "replay does not help" rather than like a bug.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from coarsecl.config import load_config, NUM_FINE
from coarsecl.coarse.sources import build_coarse_source
from coarsecl.data.cifar100 import get_datasets
from coarsecl.data.task_stream import TaskStream
from coarsecl.models.classifier import build_classifier
from coarsecl.train.cl_methods import build_cl_methods
from coarsecl.utils import seen_mask_from_classes

cfg = load_config('configs/smoke_cl.yaml')
dev = torch.device('cpu')
tr, te = get_datasets(cfg.data.data_root)
stream = TaskStream(tr, te, cfg.data.num_tasks, cfg.data.classes_per_task, cfg.data.order_seed)
src = build_coarse_source(cfg, dev)
model = build_classifier(cfg).to(dev)
replay, ewc = build_cl_methods(cfg)
mask = seen_mask_from_classes(stream.seen_classes(0), NUM_FINE, dev)

# before any task boundary both must be inert
assert replay.extra_loss(model, src, mask, dev) is None, "replay fired before buffer filled"
assert ewc.extra_loss(model, src, mask, dev) is None, "ewc fired before Fisher estimated"
print("pre-boundary: both inert  OK")

replay.after_task(model, 0, stream, src, dev)
ewc.after_task(model, 0, stream, src, dev)

n, uniq = len(replay.indices), len(set(replay.indices))
classes = {int(tr.targets[i]) for i in replay.indices}
print(f"buffer: {n} idx, {uniq} unique, {len(classes)} classes (want {cfg.cl.replay.buffer_size}/50 seen)")
assert n == uniq, "duplicate exemplars in buffer"
assert classes == set(stream.seen_classes(0)), "buffer classes != seen classes"

rl = replay.extra_loss(model, src, mask, dev)
el = ewc.extra_loss(model, src, mask, dev)
print(f"replay loss = {rl.item():.4f} (grad_fn={rl.grad_fn is not None})")
fmax = max(f.max().item() for f in ewc.fisher.values())
print(f"ewc penalty = {el.item():.6f}  fisher_max = {fmax:.3e}  anchored_params = {len(ewc.anchor)}")
assert rl.item() > 0 and rl.grad_fn is not None
assert fmax > 0, "Fisher is all zeros"
assert el.item() == 0.0, "penalty should be 0 at theta == anchor"

# perturb weights -> penalty must become positive
with torch.no_grad():
    next(model.parameters()).add_(0.1)
el2 = ewc.extra_loss(model, src, mask, dev)
print(f"ewc penalty after perturbation = {el2.item():.4f}")
assert el2.item() > 0, "penalty did not respond to weight drift"

# buffer must rebalance at the next boundary (fixed budget over more classes)
replay.after_task(model, 1, stream, src, dev)
print(f"after task 2: {len(replay.indices)} idx over {len({int(tr.targets[i]) for i in replay.indices})} classes")
print("\nALL CHECKS PASSED")

# --- regression: the replay CE must score old exemplars against the CURRENT seen
# set. Masking to the buffer-time (old-only) mask makes replay unable to contest
# the new task's logits, and old-task accuracy collapses to exactly 0.000.
print("\n--- replay mask regression ---")
seen_now = seen_mask_from_classes(stream.seen_classes(1), NUM_FINE, dev)   # task 0+1
seen_old = seen_mask_from_classes(stream.seen_classes(0), NUM_FINE, dev)   # task 0 only
captured = {}
orig = model.compute_loss
def _spy(im, fi, cd, mask):
    captured['mask'] = mask
    return orig(im, fi, cd, mask)
model.compute_loss = _spy
replay.extra_loss(model, src, seen_now, dev)
model.compute_loss = orig
used = captured['mask']
print(f"  mask given to replay CE covers {int(used.sum())} classes "
      f"(current seen={int(seen_now.sum())}, buffer-time={int(seen_old.sum())})")
assert int(used.sum()) == int(seen_now.sum()), (
    "replay CE used the stale buffer-time mask -> new classes are absent from the "
    "softmax, replay cannot suppress them, old-task accuracy goes to 0")
print("  replay competes against current-task classes  OK")
print("\nALL CHECKS PASSED")
