"""Assert the transfer-experiment folds and pretrain labels have the structure the
design depends on. Pure Python, no data, no GPU.

Run: python tests/test_transfer_folds.py

A wrong split here would not crash anything downstream; it would quietly leak a
test class into pretraining or unbalance the arms, and every known-vs-novel gap
measured on top of it would be meaningless.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from coarsecl.data.cifar100 import FINE_TO_COARSE
from coarsecl.transfer.folds import (NUM_FOLDS, build_fold, chain_pairs, pretrain_labels,
                                     shuffled_classes, sibling_agreement)

folds = [build_fold(f) for f in range(NUM_FOLDS)]
assert folds == [build_fold(f) for f in range(NUM_FOLDS)], "folds not deterministic"

novel_count, known_count = Counter(), Counter()
for fo in folds:
    pre, known, novel = set(fo.pretrain_classes), set(fo.known_classes), set(fo.novel_classes)
    assert (len(pre), len(known), len(novel)) == (45, 30, 25), (fo.index, len(pre), len(known), len(novel))
    assert not (pre & known or pre & novel or known & novel), f"fold {fo.index}: arms overlap"
    assert pre | known | novel == set(range(100)), f"fold {fo.index}: classes missing"
    assert len(fo.known_superclasses) == 15 and len(fo.novel_superclasses) == 5
    # novel superclasses contribute nothing to pretraining; known ones exactly 3 classes
    per_super = Counter(FINE_TO_COARSE[c] for c in pre)
    assert set(per_super) == set(fo.known_superclasses) and set(per_super.values()) == {3}
    # one held-out sibling pair per known superclass
    assert sorted(FINE_TO_COARSE[a] for a, b in fo.known_pairs) == list(fo.known_superclasses)
    assert all(FINE_TO_COARSE[a] == FINE_TO_COARSE[b] for a, b in fo.known_pairs)
    assert len(fo.novel_pairs) == 50
    assert all(FINE_TO_COARSE[a] == FINE_TO_COARSE[b] for a, b in fo.novel_pairs)
    novel_count.update(novel)
    known_count.update(known)
print("per-fold structure (45 / 30 / 25, disjoint, 3 per known superclass)  OK")

assert all(novel_count[c] == 1 for c in range(100)), "a class is novel more or less than once"
assert all(known_count[c] >= 1 for c in range(100)), "a class is never held out while known"
assert sorted(Counter(known_count.values()).items()) == [(1, 80), (2, 20)], known_count
print("counterbalancing (every class novel once, known-held-out 1-2x)  OK")

# pair-level pairing: each superclass's known-arm pairs are testable in its novel fold
for s in range(20):
    novel_fold = folds[s % NUM_FOLDS]
    for pair in chain_pairs(s):
        assert pair in novel_fold.novel_pairs, (s, pair)
        assert any(pair in fo.known_pairs for fo in folds if fo.index != s % NUM_FOLDS)
print("chain pairs reappear as novel-arm sibling pairs  OK")

for fo in folds:
    base = pretrain_labels(fo, "coarse")
    assert sorted(Counter(base.values()).values()) == [3] * 15
    assert sibling_agreement(base) == 1.0
    fine = pretrain_labels(fo, "fine")
    assert sorted(fine.values()) == list(range(45))
    prev = set()
    for p in (0.25, 0.5, 1.0):
        lab = pretrain_labels(fo, "coarse", p, seed=0)
        assert sorted(Counter(lab.values()).values()) == [3] * 15, f"p={p}: group sizes broke"
        moved = set(shuffled_classes(fo, p, seed=0))
        assert len(moved) == round(45 * p) and moved >= prev, f"p={p}: shuffled sets not nested"
        assert {c for c in lab if lab[c] != base[c]} <= moved, f"p={p}: unselected class moved"
        prev = moved
    assert pretrain_labels(fo, "coarse", 0.0, seed=3) == base, "p=0 must be the real grouping"
    assert pretrain_labels(fo, "coarse", 1.0, seed=1) != pretrain_labels(fo, "coarse", 1.0, seed=2)

agree = [sibling_agreement(pretrain_labels(folds[0], "coarse", p, seed=s))
         for p in (0.25, 0.5, 1.0) for s in range(20)]
a25, a50, a100 = (sum(agree[i * 20:(i + 1) * 20]) / 20 for i in range(3))
print(f"sibling agreement by shuffle_frac: 0->1.000  0.25->{a25:.3f}  0.5->{a50:.3f}  "
      f"1.0->{a100:.3f} (chance 2/44={2 / 44:.3f})")
assert 1.0 > a25 > a50 > a100 and abs(a100 - 2 / 44) < 0.03
print("shuffle dose (group sizes kept, monotone agreement, ~chance at p=1)  OK")
