"""Superclass-scale metrics: the null and the summary keys. Pure Python, no data.

Run: python tests/test_super_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coarsecl.metrics import compute_metrics, null_acc_matrix

# 4 fine classes, superclasses {0,1} -> 0 and {2,3} -> 1; task 0 = [0, 2], task 1 = [1, 3]
f2c = [0, 0, 1, 1]
tasks = [[0, 2], [1, 3]]

seen = null_acc_matrix(tasks, "seen", f2c, scale="super")
# after task 0: each seen class is the only member of its superclass among 2 seen -> 1/2
# after task 1: 2 of 4 seen share each superclass -> 2/4
assert seen == [[0.5], [0.5, 0.5]], seen
masked = null_acc_matrix(tasks, "superclass", f2c, scale="super")
assert masked == [[1.0], [1.0, 1.0]], masked
# fine-scale null unchanged by the new argument
assert null_acc_matrix(tasks, "seen", f2c) == [[0.5], [0.25, 0.25]]
print("super null  OK")

fine = [[0.9], [0.0, 0.8]]
sup = [[0.95], [0.6, 0.9]]
m = compute_metrics(fine, super_matrix=sup, source_matrix=[[0.7], [0.7, 0.7]])
assert abs(m["super_retained_accuracy"] - 0.6) < 1e-9
assert abs(m["super_average_accuracy"] - 0.75) < 1e-9
assert abs(m["coarse_source_accuracy"] - 0.7) < 1e-9
assert m["retained_accuracy"] == 0.0  # fine forgot, super kept the branch
print("super summary  OK")
print("\nALL CHECKS PASSED")
