"""Entrypoint for one condition.

Run:  python -m coarsecl.run --config configs/b_oracle.yaml

Loads + validates the config, runs the continual loop, computes metrics, and writes
results/<run_name>/{config.yaml, metrics.json, acc_matrix.csv}.
"""

from __future__ import annotations

import argparse

from .config import load_config
from .logging_io import save_run
from .metrics import compute_metrics
from .train.trainer import run_continual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    acc_matrix = run_continual(cfg)
    metrics = compute_metrics(acc_matrix)
    path = save_run(cfg, metrics)

    print(f"\n[{cfg.run.name}] done.")
    print(f"  average_accuracy   = {metrics['average_accuracy']:.4f}")
    print(f"  average_forgetting = {metrics['average_forgetting']:.4f}")
    print(f"  results -> {path}")


if __name__ == "__main__":
    main()
