"""Persist a run: resolved config + metrics (config embedded), JSON and CSV.

results/<run_name>/
  config.yaml      resolved config, verbatim
  metrics.json     config + metrics + full accuracy matrix
  acc_matrix.csv   lower-triangular A[i][j], one row per task
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict

from .config import Config, dump_config


def run_dir(cfg: Config) -> str:
    path = os.path.join(cfg.run.output_dir, cfg.run.name)
    os.makedirs(path, exist_ok=True)
    return path


def save_run(cfg: Config, metrics: Dict) -> str:
    path = run_dir(cfg)
    dump_config(cfg, os.path.join(path, "config.yaml"))

    payload = {"config": cfg.to_dict(), "metrics": metrics}
    with open(os.path.join(path, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    with open(os.path.join(path, "acc_matrix.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["after_task"] + [f"task_{j}" for j in range(len(metrics["acc_matrix"]))])
        for i, row in enumerate(metrics["acc_matrix"]):
            writer.writerow([i] + [f"{a:.6f}" for a in row])

    return path
