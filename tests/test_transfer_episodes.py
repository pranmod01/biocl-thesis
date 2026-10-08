"""Assert the transfer episodes and readouts behave as the evaluation assumes.
CPU only, synthetic labels/features, no data.

Run: python tests/test_transfer_episodes.py

The failures that matter here are silent: a pretraining class leaking into an
episode, a query image also in the support set, or batched probes that are not
independent per episode would all bias the known-vs-novel gap without crashing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from coarsecl.data.cifar100 import FINE_TO_COARSE, NUM_TRAIN
from coarsecl.transfer.episodes import (ARMS, TASKS, class_sets, draw_support, image_pools,
                                        queries)
from coarsecl.transfer.folds import NUM_FOLDS, build_fold, chain_pairs
from coarsecl.transfer.readouts import cl2n, ncm_accuracy, probe_accuracy

rng = np.random.default_rng(0)
train_labels = np.repeat(np.arange(100), 500)[rng.permutation(50000)]
test_labels = np.repeat(np.arange(100), 100)[rng.permutation(10000)]

print("--- image pools ---")
val = image_pools(train_labels, test_labels, "val", 100, split_seed=0)
test = image_pools(train_labels, test_labels, "test", 100, split_seed=0)
for c in range(100):
    s, qv, qt = set(val.support[c]), set(val.query[c]), set(test.query[c])
    assert len(s) == 400 and len(qv) == 100 and len(qt) == 100
    assert not s & qv, f"class {c}: val query overlaps support"
    assert s | qv == set(np.flatnonzero(train_labels == c)), f"class {c}: train images lost"
    assert set(test.support[c]) == s, "support pool differs between val and test"
    assert qt == set(NUM_TRAIN + np.flatnonzero(test_labels == c)), "test queries wrong"
assert not set(image_pools(train_labels, test_labels, "val", 100, 1).query[0]) == set(val.query[0])
print("  400 support / 100 reserved per class, disjoint, shared across splits  OK")

print("--- class sets ---")
for f in range(NUM_FOLDS):
    fold = build_fold(f)
    pre = set(fold.pretrain_classes)
    held = {"known": set(fold.known_classes), "novel": set(fold.novel_classes)}
    for task in TASKS:
        for arm in ARMS:
            sets = class_sets(fold, task, arm, cross_sets=20, seed=0)
            assert sets == class_sets(fold, task, arm, cross_sets=20, seed=0), "not deterministic"
            for cs in sets:
                assert not set(cs) & pre, f"pretraining class in {task}/{arm} episode {cs}"
                assert set(cs) <= held[arm], f"{task}/{arm} episode {cs} outside its arm"
                supers = [FINE_TO_COARSE[c] for c in cs]
                if task == "sibling":
                    assert len(cs) == 2 and supers[0] == supers[1]
                else:
                    assert len(cs) == 5 and len(set(supers)) == 5, f"cross5 {cs} repeats a superclass"
    assert len(class_sets(fold, "sibling", "known")) == 15
    assert len(class_sets(fold, "sibling", "novel")) == 50
# pair-level pairing: every known-arm pair is a novel-arm pair in its superclass's novel fold
for s in range(20):
    for p in chain_pairs(s):
        assert p in class_sets(build_fold(s % NUM_FOLDS), "sibling", "novel")
print("  no pretraining class, arms respected, chain pairs retestable in novel fold  OK")

print("--- support draws ---")
cs = (3, 42)
for k in (1, 5, 400):
    for r in range(3):
        sup = draw_support(val, cs, k, r, seed=0)
        assert len(sup) == 2 * k and len(set(sup)) == 2 * k, "duplicate support image"
        for i, c in enumerate(cs):
            assert set(sup[i * k:(i + 1) * k]) <= set(val.support[c]), "not class-major / wrong pool"
        assert not set(sup) & set(queries(val, cs)), "support overlaps queries"
        assert np.array_equal(sup, draw_support(test, cs, k, r, seed=0)), "draw depends on split"
assert not np.array_equal(draw_support(val, cs, 5, 0), draw_support(val, cs, 5, 1))
print("  no duplicates, class-major, disjoint from queries, same draw in val and test  OK")

print("--- readouts ---")
torch.manual_seed(0)
E, n, k, q, D = 6, 2, 5, 50, 16
centers = torch.randn(E, n, D) * 3
sup = (centers.repeat_interleave(k, 1) + 0.1 * torch.randn(E, n * k, D))
qry = (centers.repeat_interleave(q, 1) + 0.1 * torch.randn(E, n * q, D))
assert torch.all(ncm_accuracy(sup, qry, n) == 1)
assert torch.all(probe_accuracy(sup, qry, n, 1e-3, 200, 0.05) == 1)
# NCM against a direct computation
noisy_q = qry + 3 * torch.randn_like(qry)
protos = sup.view(E, n, k, D).mean(2)
pred = torch.stack([torch.cdist(noisy_q[e], protos[e]).argmin(1) for e in range(E)])
want = (pred == torch.arange(n).repeat_interleave(q)).float().mean(1)
assert torch.allclose(ncm_accuracy(sup, noisy_q, n), want)
# batched probes are independent: each episode alone gives the same answer
sup_hard = sup + 6 * torch.randn_like(sup)
q_hard = qry + 8 * torch.randn_like(qry)
batched = probe_accuracy(sup_hard, q_hard, n, 1e-2, 150, 0.05)
alone = torch.cat([probe_accuracy(sup_hard[e:e + 1], q_hard[e:e + 1], n, 1e-2, 150, 0.05)
                   for e in range(E)])
assert torch.allclose(batched, alone, atol=1e-6), f"{batched} vs {alone}"
assert 0.55 < batched.mean() < 0.9, f"hard case should be neither trivial nor hopeless: {batched.mean()}"
f = cl2n(torch.randn(10, D) + 5, torch.full((D,), 5.0))
assert torch.allclose(f.norm(dim=-1), torch.ones(10))
print(f"  separable -> 1.0, NCM matches direct, batched == per-episode probe "
      f"(hard-case acc {batched.mean():.2f})  OK")

print("\nALL CHECKS PASSED")
