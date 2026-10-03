"""Entrypoint for one condition.

Run:  python -m coarsecl.run --config configs/b_oracle.yaml
      python -m coarsecl.run --config configs/b_oracle.yaml --seed 1

Loads + validates the config, runs the continual loop, computes metrics, and writes
results/<run_name>/{config.yaml, metrics.json, acc_matrix.csv}.

`--seed N` overrides `run.seed` and suffixes the run name with `_s<N>`, so a seed
sweep needs no extra YAMLs and cannot overwrite the seed-0 result. It varies the
training seed only (init, shuffling, replay sampling); `data.order_seed` is left
alone, so every seed sees the same task stream and the arms stay paired.
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
    ap.add_argument("--seed", type=int, default=None,
                    help="override run.seed; appends _s<N> to the run name")
    ap.add_argument("--name", default=None,
                    help="override run.name (and so the output dir)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.seed is not None:
        overrides["run"] = {"seed": args.seed}
    if args.name is not None:
        overrides.setdefault("run", {})["name"] = args.name
    cfg = load_config(args.config, args.base, overrides or None)
    if args.seed is not None and args.name is None:
        cfg.run.name = f"{cfg.run.name}_s{args.seed}"
    acc_matrix = run_continual(cfg)
    metrics = compute_metrics(
        acc_matrix,
        task_classes=task_class_order(cfg.data.num_tasks, cfg.data.classes_per_task,
                                      cfg.data.order_seed),
        kind=null_kind(conditioning_for(cfg), cfg.coarse.source,
                       cfg.conditioning.eval_withhold, cfg.coarse.soft.smooth,
                       cfg.coarse.noisy.rate),
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
