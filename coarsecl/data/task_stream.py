"""Class-incremental task stream: split the 100 fine classes into N tasks.

A fixed random class order (seeded) is partitioned into `num_tasks` contiguous
chunks. For each task we expose the train subset (current-task classes only) and a
per-task test subset (used to measure forgetting on earlier tasks).
"""

from __future__ import annotations

from typing import List

import numpy as np
from torch.utils.data import Dataset, Subset


def task_class_order(num_tasks: int, classes_per_task: int,
                     order_seed: int = 0) -> List[List[int]]:
    """The class->task partition, as a pure function of the config. Exposed so the
    null-model baseline in metrics.py can be computed without loading the dataset."""
    rng = np.random.RandomState(order_seed)
    order = rng.permutation(num_tasks * classes_per_task).tolist()
    return [sorted(order[i * classes_per_task:(i + 1) * classes_per_task])
            for i in range(num_tasks)]


class TaskStream:
    def __init__(
        self,
        train_set: Dataset,
        test_set: Dataset,
        num_tasks: int,
        classes_per_task: int,
        order_seed: int = 0,
    ) -> None:
        self.train_set = train_set
        self.test_set = test_set
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        self.task_classes: List[List[int]] = task_class_order(
            num_tasks, classes_per_task, order_seed)

        self._train_by_class = self._index_by_class(train_set)
        self._test_by_class = self._index_by_class(test_set)

    @staticmethod
    def _index_by_class(ds: Dataset) -> dict:
        targets = np.asarray(ds.targets)
        return {c: np.where(targets == c)[0].tolist() for c in range(int(targets.max()) + 1)}

    def _subset(self, ds: Dataset, by_class: dict, classes: List[int]) -> Subset:
        idx: List[int] = []
        for c in classes:
            idx.extend(by_class[c])
        return Subset(ds, idx)

    def train_subset(self, task_id: int) -> Subset:
        return self._subset(self.train_set, self._train_by_class, self.task_classes[task_id])

    def test_subset(self, task_id: int) -> Subset:
        """Test samples belonging to one task's classes (for per-task forgetting)."""
        return self._subset(self.test_set, self._test_by_class, self.task_classes[task_id])

    def train_indices_of_class(self, c: int) -> List[int]:
        """Train-set indices for one fine class (exemplar buffers sample from here)."""
        return self._train_by_class[c]

    def seen_classes(self, up_to_task: int) -> List[int]:
        """All fine classes introduced in tasks 0..up_to_task (inclusive)."""
        seen: List[int] = []
        for t in range(up_to_task + 1):
            seen.extend(self.task_classes[t])
        return sorted(seen)
