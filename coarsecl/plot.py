"""Plot all conditions on one chart for direct comparison.

Run:  python -m coarsecl.plot --runs results/a_baseline results/b_oracle ... \
                              --out results/comparison.png

Reads each run's metrics.json and draws two panels:
  (left)  average accuracy over seen tasks after each task (the forgetting curve)
  (right) final average accuracy and average forgetting, as grouped bars.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(run_path: str) -> Dict:
    with open(os.path.join(run_path, "metrics.json")) as fh:
        payload = json.load(fh)
    payload["_name"] = payload["config"]["run"]["name"]
    return payload


def plot_runs(run_paths: List[str], out_path: str) -> None:
    runs = [_load(p) for p in run_paths]
    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))

    for r in runs:
        curve = r["metrics"]["avg_acc_curve"]
        ax_curve.plot(range(1, len(curve) + 1), curve, marker="o", label=r["_name"])
    ax_curve.set_xlabel("tasks seen")
    ax_curve.set_ylabel("avg accuracy over seen tasks")
    ax_curve.set_title("Class-IL accuracy after each task")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    names = [r["_name"] for r in runs]
    accs = [r["metrics"]["average_accuracy"] for r in runs]
    forgets = [r["metrics"]["average_forgetting"] for r in runs]
    x = range(len(names))
    width = 0.38
    ax_bar.bar([i - width / 2 for i in x], accs, width, label="avg accuracy")
    ax_bar.bar([i + width / 2 for i in x], forgets, width, label="avg forgetting")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(names, rotation=20, ha="right")
    ax_bar.set_title("Final average accuracy & forgetting")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved comparison plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories")
    ap.add_argument("--out", default="results/comparison.png")
    args = ap.parse_args()
    plot_runs(args.runs, args.out)


if __name__ == "__main__":
    main()
