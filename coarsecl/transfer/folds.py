"""Deterministic class splits for the transfer experiment: given a learned coarse
structure, are new fine classes in a KNOWN superclass learnable from fewer examples
than new classes in a NOVEL one?

Four folds. Superclass s is novel in fold s % 4: all 5 of its fine classes are held
out from pretraining. In each of its other three (known) folds it holds out one
sibling pair and pretrains on its remaining 3 classes. The three pairs form a chain
over its 5 classes, (a,b), (c,d), (e,a). So every class is held out at least once
while its superclass is known, and once while it is novel, and the same three pairs
can be tested as sibling pairs in the superclass's novel fold (pair-level pairing,
not just class-level). The chain starts at member s % 5, so the class held out twice
varies across superclasses.

Per fold: 45 pretrain classes (15 known superclasses x 3), 30 known-arm classes
(15 sibling pairs), 25 novel-arm classes (5 superclasses x 5). 45 + 30 + 25 = 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from ..config import NUM_COARSE
from ..data.cifar100 import FINE_TO_COARSE

NUM_FOLDS = 4
LABEL_MODES = ("coarse", "fine")

Pair = Tuple[int, int]


def superclass_members(s: int) -> List[int]:
    """The 5 fine classes of superclass s, ascending."""
    return [c for c, sc in enumerate(FINE_TO_COARSE) if sc == s]


def chain_pairs(s: int) -> List[Pair]:
    """Superclass s's held-out pair in its 1st, 2nd and 3rd known fold. Pairs are
    ascending, so they compare equal to the same pair in Fold.novel_pairs."""
    m = superclass_members(s)
    m = m[s % 5:] + m[:s % 5]
    return [tuple(sorted(p)) for p in ((m[0], m[1]), (m[2], m[3]), (m[4], m[0]))]


def known_folds(s: int) -> List[int]:
    return [f for f in range(NUM_FOLDS) if f != s % NUM_FOLDS]


@dataclass(frozen=True)
class Fold:
    index: int
    known_superclasses: Tuple[int, ...]  # 15
    novel_superclasses: Tuple[int, ...]  # 5
    pretrain_classes: Tuple[int, ...]    # 45
    known_pairs: Tuple[Pair, ...]        # 15 held-out sibling pairs, one per known superclass
    novel_classes: Tuple[int, ...]       # 25

    @property
    def known_classes(self) -> Tuple[int, ...]:
        return tuple(sorted(c for p in self.known_pairs for c in p))

    @property
    def novel_pairs(self) -> Tuple[Pair, ...]:
        """All 50 within-superclass pairs of the novel arm (10 per superclass)."""
        return tuple(p for s in self.novel_superclasses
                     for p in combinations(superclass_members(s), 2))


def build_fold(f: int) -> Fold:
    if not 0 <= f < NUM_FOLDS:
        raise ValueError(f"fold must be in [0, {NUM_FOLDS}), got {f}")
    known = tuple(s for s in range(NUM_COARSE) if s % NUM_FOLDS != f)
    novel = tuple(s for s in range(NUM_COARSE) if s % NUM_FOLDS == f)
    pairs, pretrain = [], []
    for s in known:
        pair = chain_pairs(s)[known_folds(s).index(f)]
        pairs.append(pair)
        pretrain += [c for c in superclass_members(s) if c not in pair]
    return Fold(index=f, known_superclasses=known, novel_superclasses=novel,
                pretrain_classes=tuple(sorted(pretrain)), known_pairs=tuple(pairs),
                novel_classes=tuple(c for s in novel for c in superclass_members(s)))


def pretrain_labels(fold: Fold, mode: str, shuffle_frac: float = 0.0,
                    seed: int = 0) -> Dict[int, int]:
    """Fine class -> training label for the fold's 45 pretrain classes.

    coarse: index of its superclass among the 15 known ones (15-way).
    fine:   index among the pretrain classes (45-way).
    coarse with shuffle_frac p: a seeded fraction p of the classes swap group labels
    among themselves (a permutation, so every group keeps 3 classes). p = 1 is a
    uniformly random grouping. Classes are picked in one seeded order, so the
    shuffled sets are nested across p, the same way the noisy source nests its rates.
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"label mode {mode!r} not in {LABEL_MODES}")
    classes = fold.pretrain_classes
    if mode == "fine":
        if shuffle_frac:
            raise ValueError("shuffle_frac only applies to label mode 'coarse'")
        return {c: i for i, c in enumerate(classes)}
    group = {s: i for i, s in enumerate(fold.known_superclasses)}
    labels = {c: group[FINE_TO_COARSE[c]] for c in classes}
    moved = shuffled_classes(fold, shuffle_frac, seed)
    rng = np.random.default_rng([seed, fold.index, len(moved)])
    for c, lab in zip(moved, rng.permutation([labels[c] for c in moved])):
        labels[c] = int(lab)
    return labels


def shuffled_classes(fold: Fold, shuffle_frac: float, seed: int = 0) -> List[int]:
    """The pretrain classes whose group labels get permuted: a prefix of one seeded
    order, so the set at a smaller shuffle_frac is contained in the set at a larger one.
    (A selected class can still draw its own label back by chance.)"""
    order = np.random.default_rng([seed, fold.index]).permutation(len(fold.pretrain_classes))
    return [fold.pretrain_classes[i] for i in order[:round(shuffle_frac * len(order))]]


def sibling_agreement(labels: Dict[int, int]) -> float:
    """Fraction of same-label class pairs that are true siblings: 1.0 for the real
    superclasses, about 2/44 for a random grouping into 3s. This is the dose axis of
    the shuffle control, measured rather than assumed."""
    same = [(a, b) for a, b in combinations(sorted(labels), 2) if labels[a] == labels[b]]
    return sum(FINE_TO_COARSE[a] == FINE_TO_COARSE[b] for a, b in same) / max(len(same), 1)
