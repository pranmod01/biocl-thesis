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

import numpy as np
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

# --- priority replay (schema / hardness) ---
# Scores must be the quantities the docstring claims, the replay distribution must
# be a valid mix with a uniform floor, and prioritizing must not change WHAT is
# stored (buffer contents identical to a uniform run with the same seed).
print("\n--- priority replay ---")
import copy
import torch.nn.functional as F
from coarsecl.data.cifar100 import coarse_of

def replay_with(priority, schema="oracle", level="exemplar", probes=0):
    c = copy.deepcopy(cfg)
    c.cl.methods = ["replay"]
    c.cl.replay.priority, c.cl.replay.priority_schema = priority, schema
    c.cl.replay.priority_level, c.cl.replay.priority_probe_per_class = level, probes
    c.validate()
    (r,) = build_cl_methods(c, dev)
    r.after_task(model, 0, stream, src, dev)
    return r

uniform = replay_with("none")
for priority in ("schema", "hardness"):
    r = replay_with(priority)
    assert r.indices == uniform.indices, f"{priority}: buffer differs from uniform run"
    n = len(r.indices)
    assert np.allclose(r.probs, 1.0 / n), "probs should start uniform"
    loss = r.extra_loss(model, src, mask, dev)  # step 0 -> triggers the first re-score
    assert loss.item() > 0 and loss.grad_fn is not None
    floor = (1 - r.priority_mix) / n
    assert abs(r.probs.sum() - 1) < 1e-9 and r.probs.min() >= floor - 1e-12, "bad replay dist"

    # independent recomputation of the score on the first 64 clean exemplars
    model.eval()
    with torch.no_grad():
        x, y = r._score_x[:64], r._score_y[:64]
        p = F.softmax(model(x, src.get_coarse_dist(x, y))["fine_logits"].masked_fill(~mask, float("-inf")), 1)
        if priority == "hardness":
            want = 1 - p[torch.arange(64), y]
        else:  # oracle schema: mass on fine classes of the TRUE superclass
            same = coarse_of(torch.arange(NUM_FINE))[None, :] == coarse_of(y)[:, None]
            want = 1 - (p * same).sum(1)
    model.train()
    got = r.buffer_scores(model, src, mask)[:64]
    assert torch.allclose(got, want, atol=1e-5), f"{priority}: score mismatch"

    # re-score cadence: exactly one re-score per priority_refresh steps
    for _ in range(r.priority_refresh):
        r.extra_loss(model, src, mask, dev)
    assert len(r._score_log) == 2, f"expected 2 re-scores, got {len(r._score_log)}"
    print(f"  {priority}: buffer={n}, score mean {got.mean():.3f}, "
          f"p range [{r.probs.min()*n:.2f}, {r.probs.max()*n:.2f}] x uniform  OK")
print("\nALL CHECKS PASSED")

# --- superclass-level priority (+ held-out probes) ---
# Probes must be carved out of the uniform run's buffer (same stored set, no extra
# memory), never replayed; R_s must be the per-superclass mean score; and with the
# oracle schema every replay slot in a superclass must get the same weight.
print("\n--- superclass priority ---")
for probes, schema in ((0, "oracle"), (2, "oracle"), (2, "trained")):
    r = replay_with("schema", schema, level="superclass", probes=probes)
    stored = set(r.indices) | set(r.probe_indices)
    assert not set(r.indices) & set(r.probe_indices), "probe exemplars leak into replay"
    assert stored == set(uniform.indices), "stored set differs from the uniform run"
    per_cls = {}
    for i in r.probe_indices:
        per_cls[int(tr.targets[i])] = per_cls.get(int(tr.targets[i]), 0) + 1
    assert probes == 0 or set(per_cls.values()) == {probes}, f"probe counts {set(per_cls.values())}"
    r.extra_loss(model, src, mask, dev)
    n = len(r.indices)
    assert abs(r.probs.sum() - 1) < 1e-9 and r.probs.min() >= (1 - r.priority_mix) / n - 1e-12
    s_ = r.buffer_scores(model, src, mask)
    R = r.superclass_scores(s_)
    if schema == "oracle":
        sc = coarse_of(r._score_y)
        for c_ in sc.unique().tolist():  # R_s = plain mean over that superclass
            assert torch.isclose(R[c_], s_[sc == c_].mean(), atol=1e-5), f"R[{c_}] wrong"
        rc = coarse_of(torch.tensor([int(tr.targets[i]) for i in r.indices]))
        for c_ in rc.unique().tolist():
            ps = r.probs[(rc == c_).numpy()]
            assert np.allclose(ps, ps[0]), f"superclass {c_}: unequal slot weights"
    print(f"  probes={probes} schema={schema}: replay {n} + probe {len(r.probe_indices)}, "
          f"R max/mean {float(R[R > 0].max() / R[R > 0].mean()):.2f}  OK")
print("\nALL CHECKS PASSED")

# --- noisy coarse source ---
# Corruption must be (1) at the requested rate, (2) always a WRONG superclass,
# (3) fixed per image across calls, augmentation and DataLoader passes, (4) nested
# across rates, and (5) rate 0 must reproduce the oracle exactly.
print("\n--- noisy source ---")
from coarsecl.coarse.sources import NoisySource, OracleSource
from coarsecl.data.cifar100 import NUM_TRAIN
from torch.utils.data import DataLoader, Subset
fine_all = torch.tensor(tr.targets)
uid_all = torch.arange(len(fine_all))
oracle_c = coarse_of(fine_all)
prev = None
for rate in (0.0, 0.1, 0.34, 0.95):
    ns = NoisySource(rate=rate, seed=0)
    obs = ns.observed_coarse(fine_all, uid_all)
    wrong = obs != oracle_c
    assert abs(wrong.float().mean().item() - rate) < 0.01, f"rate {rate}: got {wrong.float().mean():.3f}"
    if prev is not None:
        assert not (prev & ~wrong).any(), f"rate {rate}: corrupted sets not nested"
    prev = wrong
    print(f"  rate={rate}: corrupted {wrong.float().mean():.4f}  OK")
assert torch.equal(NoisySource(0.0).get_coarse_dist(None, fine_all[:500], uid_all[:500]),
                   OracleSource().get_coarse_dist(None, fine_all[:500])), "rate 0 != oracle"
cnt = torch.bincount(obs[wrong], minlength=20)  # obs/wrong are from rate 0.95 here
assert (cnt > 0).all(), "rate 0.95: some superclass never drawn as a wrong label"
ns = NoisySource(rate=0.34, seed=0)
# fixed per image through real (augmenting) loader passes, and train/test uids distinct
sub = Subset(tr, list(range(0, 2000, 7)))
passes = []
for _ in range(2):
    xs, ys, us = zip(*DataLoader(sub, batch_size=64, shuffle=True))
    y, u = torch.cat(ys), torch.cat(us)
    order = u.argsort()
    passes.append(ns.observed_coarse(y[order], u[order]))
assert torch.equal(passes[0], passes[1]), "corruption changed between loader passes"
_, _, u_te = te[0]
assert u_te == NUM_TRAIN, f"test uid offset wrong: {u_te}"
try:
    ns.get_coarse_dist(None, fine_all[:4], None)
    raise AssertionError("noisy source accepted missing uids")
except ValueError:
    pass
print("  fixed per image across loader passes, oracle at rate 0, uid offsets  OK")
print("\nALL CHECKS PASSED")
