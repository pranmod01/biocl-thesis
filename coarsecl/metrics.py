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
"""

from __future__ import annotations

from typing import Dict, List


def compute_metrics(acc_matrix: List[List[float]]) -> Dict:
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

    return {
        "average_accuracy": average_accuracy,
        "average_forgetting": average_forgetting,
        "average_incremental_accuracy": average_incremental_accuracy,
        "retained_accuracy": retained_accuracy,
        "final_per_task": final_row,
        "avg_acc_curve": avg_acc_curve,
        "acc_matrix": acc_matrix,
    }
