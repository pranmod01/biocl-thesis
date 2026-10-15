"""Continual-learning metrics from the lower-triangular accuracy matrix.

A[i][j] = accuracy on task j after training through task i (j <= i).
- average_accuracy: mean over tasks of the final-row accuracies.
- average_forgetting: mean over tasks j<N-1 of (max_i A[i][j] - A[N-1][j]).
- average_incremental_accuracy: mean of avg_acc_curve — the accuracy trajectory
  over the whole run (standard CL metric; rewards retention throughout, not just
  at the end).
- retained_accuracy: mean of the strictly-below-diagonal entries A[i][j] (i>j) —
  accuracy on a task at every step *after* it stopped being current. This is the
  direct measure of forgetting reduction: a total-forgetting baseline scores ~0
  here, so it separates conditions that the peak-dominated average_forgetting
  scalar can misrank.
- avg_acc_curve: average accuracy over seen tasks after each task (for plotting).

NULL MODEL. Raw accuracy is not comparable across conditions that restrict the
output space by different amounts. The `gate` mechanism masks predictions to the
true superclass using information handed to it at TEST time, so it is scored over
~5 candidate classes where the baseline is scored over all seen classes — and
early in the run, when few of a superclass's 5 members have been introduced, over
as few as 1. Part of a gated run's accuracy is therefore arithmetic, not
knowledge.

The null model makes that explicit: it has learned NOTHING (uniform fine logits),
but is scored through the exact same masking the real run uses, so it guesses
uniformly over whatever candidate set survives. Every headline metric gets a
`null_*` twin and a ratio. ratio > 1 means the network contributed something over
and above its output-space restriction; ratio < 1 means it did worse than chance
within its own mask (which recency bias makes entirely possible — it
systematically prefers the most recently learned member of the candidate set).
Compare like with like: null_average_accuracy against average_accuracy,
null_retained_accuracy against retained_accuracy.

SUPERCLASS SCALE. The same fine predictions are also scored by superclass (see
train/evaluate.py) and summarised under `super_*` keys, with their own null: a
uniform guess over the seen classes lands in the true superclass with probability
(#seen classes in it) / (#seen), and a superclass-masked null always does (1.0).
Fine forgetting with super retained well above its null means the model still
routes old images to the right branch and loses only the leaf.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def null_kind(mechanism: str, source: str, eval_withhold: bool = False,
              smooth: float = 0.0, noisy_rate: float = 0.0) -> Optional[str]:
    """Which candidate set a zero-knowledge model would be left choosing from.

    'seen'       -> all classes seen so far (no output-space restriction)
    'superclass' -> seen classes sharing the sample's true superclass
    None         -> not computable in closed form

    A gate adds log P(superclass) to every fine logit, and that prior is CONSTANT
    within a superclass. So a model with uniform logits has all members of the
    top-prior superclass tied, and argmax falls uniformly among the seen ones.
    For oracle (and any soft smooth < 1) the top-prior superclass is the true one,
    which is why both give the same null. `trained` is excluded: its top superclass
    is right only ~66% of the time and which samples those are depends on the
    coarse net's per-image predictions, so no closed form exists — returning None
    is deliberate, a wrong null would be worse than none.
    """
    if eval_withhold:
        return "seen"  # a uniform dist is fed at eval; the gate is a no-op
    if mechanism != "gate" or source == "none":
        return "seen"
    if source == "trained":
        return None
    if source == "noisy" and noisy_rate > 0.0:
        return None  # a corrupted image is masked to the wrong superclass; like trained,
        # the null depends on which images are corrupted and on seen-class coverage
    if source == "soft" and smooth >= 1.0:
        return "seen"  # uniform prior, no restriction
    return "superclass"


def null_acc_matrix(task_classes: List[List[int]], kind: str,
                    fine_to_coarse: Optional[List[int]] = None,
                    scale: str = "fine") -> List[List[float]]:
    """Expected accuracy matrix of a model that learned nothing but is scored
    through the same masking as the real run. Pure combinatorics on the class order.
    scale="super" scores whether the guess lands in the true superclass."""
    n = len(task_classes)
    if fine_to_coarse is None:
        from .data.cifar100 import fine_to_coarse_tensor
        fine_to_coarse = fine_to_coarse_tensor().tolist()

    matrix: List[List[float]] = []
    for i in range(n):
        seen = [c for t in range(i + 1) for c in task_classes[t]]
        row = []
        for j in range(i + 1):
            if scale == "super":
                if kind == "superclass":
                    cell = 1.0  # the guess is confined to the true superclass
                else:
                    accs = [sum(1 for k in seen
                                if fine_to_coarse[k] == fine_to_coarse[c]) / len(seen)
                            for c in task_classes[j]]
                    cell = sum(accs) / len(accs)
            elif kind == "seen":
                cell = 1.0 / len(seen)
            else:
                # per true class: 1 / (# seen classes in its superclass)
                accs = [1.0 / sum(1 for k in seen
                                  if fine_to_coarse[k] == fine_to_coarse[c])
                        for c in task_classes[j]]
                cell = sum(accs) / len(accs)
            row.append(cell)
        matrix.append(row)
    return matrix


def _summary(matrix: List[List[float]]) -> Dict[str, float]:
    """avg / AIA / retained of one lower-triangular matrix."""
    n = len(matrix)
    curve = [sum(r) / len(r) for r in matrix]
    old = [matrix[i][j] for i in range(n) for j in range(i)]
    return {"average_accuracy": sum(matrix[-1]) / n,
            "average_incremental_accuracy": sum(curve) / n,
            "retained_accuracy": sum(old) / len(old) if old else 0.0}


def compute_metrics(acc_matrix: List[List[float]],
                    task_classes: Optional[List[List[int]]] = None,
                    kind: Optional[str] = None,
                    super_matrix: Optional[List[List[float]]] = None,
                    source_matrix: Optional[List[List[float]]] = None) -> Dict:
    n = len(acc_matrix)
    final_row = acc_matrix[-1]

    average_accuracy = sum(final_row) / n

    forgets: List[float] = []
    for j in range(n - 1):  # last task has no later step to forget over
        max_acc = max(acc_matrix[i][j] for i in range(j, n))
        forgets.append(max_acc - final_row[j])
    average_forgetting = sum(forgets) / len(forgets) if forgets else 0.0

    avg_acc_curve = [sum(row) / len(row) for row in acc_matrix]
    average_incremental_accuracy = sum(avg_acc_curve) / n

    # strictly-below-diagonal entries: accuracy on each task after it stopped being current
    old_evals = [acc_matrix[i][j] for i in range(n) for j in range(i)]
    retained_accuracy = sum(old_evals) / len(old_evals) if old_evals else 0.0

    out = {
        "average_accuracy": average_accuracy,
        "average_forgetting": average_forgetting,
        "average_incremental_accuracy": average_incremental_accuracy,
        "retained_accuracy": retained_accuracy,
        "final_per_task": final_row,
        "avg_acc_curve": avg_acc_curve,
        "acc_matrix": acc_matrix,
    }

    if task_classes is not None and kind is not None:
        null = null_acc_matrix(task_classes, kind)
        null_final = null[-1]
        null_old = [null[i][j] for i in range(n) for j in range(i)]
        null_curve = [sum(r) / len(r) for r in null]
        out.update({
            "null_kind": kind,
            "null_average_accuracy": sum(null_final) / n,
            "null_average_incremental_accuracy": sum(null_curve) / n,
            "null_retained_accuracy": (sum(null_old) / len(null_old)
                                       if null_old else 0.0),
        })
        # ratio > 1: the network beat chance-within-its-own-mask; < 1: it did worse
        for key in ("average_accuracy", "average_incremental_accuracy",
                    "retained_accuracy"):
            denom = out[f"null_{key}"]
            out[f"{key}_vs_null"] = (out[key] / denom) if denom > 0 else None
    else:
        out["null_kind"] = kind  # None = no closed-form null for this condition

    if super_matrix is not None:
        out["super_acc_matrix"] = super_matrix
        out["super_avg_acc_curve"] = [sum(r) / len(r) for r in super_matrix]
        for key, val in _summary(super_matrix).items():
            out[f"super_{key}"] = val
        if task_classes is not None and kind is not None:
            for key, val in _summary(null_acc_matrix(task_classes, kind,
                                                     scale="super")).items():
                out[f"null_super_{key}"] = val
    if source_matrix is not None:
        # the coarse signal's own accuracy; a constant per class for a fixed source,
        # so the final row (every class) is the reference value
        out["coarse_source_accuracy"] = sum(source_matrix[-1]) / n

    return out
