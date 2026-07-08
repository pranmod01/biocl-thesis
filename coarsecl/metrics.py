"""Continual-learning metrics from the lower-triangular accuracy matrix.

A[i][j] = accuracy on task j after training through task i (j <= i).
- average_accuracy: mean over tasks of the final-row accuracies.
- average_forgetting: mean over tasks j<N-1 of (max_i A[i][j] - A[N-1][j]).
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

    return {
        "average_accuracy": average_accuracy,
        "average_forgetting": average_forgetting,
        "final_per_task": final_row,
        "avg_acc_curve": avg_acc_curve,
        "acc_matrix": acc_matrix,
    }
