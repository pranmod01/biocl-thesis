"""Entrypoint for one condition.

Run:  python -m coarsecl.run --config configs/b_oracle.yaml

Loads + validates the config, runs the continual loop, computes metrics, and writes
results/<run_name>/{config.yaml, metrics.json, acc_matrix.csv}.
"""

from __future__ import annotations

import argparse

from .config import load_config
from .data.task_stream import task_class_order
from .logging_io import save_run
from .metrics import compute_metrics, null_kind
from .models.classifier import conditioning_for
from .train.trainer import run_continual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    acc_matrix = run_continual(cfg)
    metrics = compute_metrics(
        acc_matrix,
        task_classes=task_class_order(cfg.data.num_tasks, cfg.data.classes_per_task,
                                      cfg.data.order_seed),
        kind=null_kind(conditioning_for(cfg), cfg.coarse.source,
                       cfg.conditioning.eval_withhold, cfg.coarse.soft.smooth),
    )
    path = save_run(cfg, metrics)

    print(f"\n[{cfg.run.name}] done.")
    print(f"  average_accuracy   = {metrics['average_accuracy']:.4f}")
    print(f"  average_forgetting = {metrics['average_forgetting']:.4f}")
    print(f"  retained_accuracy  = {metrics['retained_accuracy']:.4f}")
    if metrics.get("null_average_accuracy") is not None:
        print(f"  vs null model ({metrics['null_kind']}): "
              f"avg {metrics['null_average_accuracy']:.4f} "
              f"(ratio {metrics['average_accuracy_vs_null']:.2f}), "
              f"retained {metrics['null_retained_accuracy']:.4f} "
              f"(ratio {metrics['retained_accuracy_vs_null']:.2f})")
    else:
        print("  vs null model: no closed form for this condition")
    print(f"  results -> {path}")


if __name__ == "__main__":
    main()
