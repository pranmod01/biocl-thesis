"""Few-shot episodes over a fold's held-out classes (no pretraining class ever appears).

Two tasks:
- sibling: 2-way, the two classes of one within-superclass pair. The primary outcome.
  known arm = the fold's 15 held-out pairs (their superclass was in pretraining);
  novel arm = all 50 pairs of the 5 novel superclasses.
- cross5: 5-way, one class from each of 5 different superclasses. Sanity check only:
  the coarse net was trained to separate exactly these superclasses.

Images are indexed into one array of all 60k CIFAR-100 images: train index i is i,
test index j is NUM_TRAIN + j (the dataset's uid convention).

Each class's train images are split once (split_seed) into a reserved validation
query set and a support pool. Support is drawn from the pool in both splits, so val
and test runs see the same support distribution; only the queries differ.

Support draws are seeded by (episode_seed, k, rep, classes) and nothing else, so the
same class set gets the SAME support images under every pretraining condition and
seed, and in both the fold where its superclass is known and the one where it is
novel. Comparisons are then paired down to the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.cifar100 import FINE_TO_COARSE, NUM_TRAIN
from .folds import Fold, superclass_members

TASKS = ("sibling", "cross5")
ARMS = ("known", "novel")


@dataclass(frozen=True)
class ImagePools:
    support: Dict[int, np.ndarray]  # class -> unreserved train indices
    query: Dict[int, np.ndarray]    # class -> query indices (reserved train, or test)


def image_pools(train_labels: Sequence[int], test_labels: Sequence[int], split: str,
                val_per_class: int, split_seed: int) -> ImagePools:
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    support, query = {}, {}
    for c in range(100):
        idx = np.flatnonzero(train_labels == c)
        idx = idx[np.random.default_rng([split_seed, c]).permutation(len(idx))]
        reserved, support[c] = np.sort(idx[:val_per_class]), np.sort(idx[val_per_class:])
        query[c] = reserved if split == "val" else NUM_TRAIN + np.flatnonzero(test_labels == c)
    return ImagePools(support, query)


def class_sets(fold: Fold, task: str, arm: str, cross_sets: int = 20,
               seed: int = 0) -> List[Tuple[int, ...]]:
    """The class sets (episodes' label spaces) of one task and arm in this fold."""
    if task == "sibling":
        return list(fold.known_pairs if arm == "known" else fold.novel_pairs)
    if task != "cross5":
        raise ValueError(f"unknown task {task!r}")
    rng = np.random.default_rng([seed, fold.index, ARMS.index(arm)])
    if arm == "known":
        by_super = {FINE_TO_COARSE[p[0]]: p for p in fold.known_pairs}
        supers = sorted(by_super)
        pick = lambda s: by_super[s]
    else:
        supers = list(fold.novel_superclasses)
        pick = superclass_members
    sets = []
    for _ in range(cross_sets):
        chosen = sorted(rng.choice(supers, 5, replace=False)) if len(supers) > 5 else supers
        sets.append(tuple(int(rng.choice(pick(s))) for s in chosen))
    return sets


def draw_support(pools: ImagePools, classes: Tuple[int, ...], k: int, rep: int,
                 seed: int = 0) -> np.ndarray:
    """[n * k] image indices, class-major (class i's k images at i*k:(i+1)*k)."""
    rng = np.random.default_rng([seed, k, rep, *classes])
    return np.concatenate([rng.choice(pools.support[c], k, replace=False) for c in classes])


def queries(pools: ImagePools, classes: Tuple[int, ...]) -> np.ndarray:
    """All query image indices of the class set, class-major."""
    return np.concatenate([pools.query[c] for c in classes])
