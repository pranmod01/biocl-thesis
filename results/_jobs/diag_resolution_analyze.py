"""Read diag_resolution.py outputs and answer the premise question.

Q1  As resolution drops, does superclass accuracy hold up better than fine accuracy?
    Compared on chance-normalized accuracy, (acc - chance) / (acc_at_32 - chance), so a
    20-way and a 100-way task are on the same 0..1 "fraction of full-res skill" scale.
    Reported for the coarse-trained net (is superclass decodable at low res?) and for
    the fine net's collapsed predictions (does the fine net still route to the branch?).

Q2  Are the fine net's errors organized by superclass, more so at low resolution?
    Within-superclass share of off-diagonal errors, against the same share under 1000
    random partitions of the 100 classes into 20 groups of 5. The random partitions
    are the control for "errors just cluster among a few confusable classes": only an
    excess over random groupings is superclass structure.

Run: python results/_jobs/diag_resolution_analyze.py [--dir results/diag_resolution]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from coarsecl.data.cifar100 import FINE_TO_COARSE


def within_group_share(conf: np.ndarray, groups: np.ndarray) -> float:
    off = conf.copy()
    np.fill_diagonal(off, 0)
    same = groups[:, None] == groups[None, :]
    return off[same].sum() / max(off.sum(), 1)


def random_partition_shares(conf, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(20), 5)
    return np.array([within_group_share(conf, rng.permutation(base)) for _ in range(n_perm)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/diag_resolution")
    args = ap.parse_args()

    runs = defaultdict(list)  # (labels, res) -> [run dicts over seeds]
    for path in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        r = json.load(open(path))
        runs[(r["args"]["labels"], r["args"]["res"])].append(r)
    if not runs:
        sys.exit(f"no runs in {args.dir}")
    resolutions = sorted({res for _, res in runs}, reverse=True)
    mean = lambda xs: sum(xs) / len(xs)

    def norm(acc, chance, full):
        return (acc - chance) / (full - chance) if full is not None and full > chance else float("nan")

    full = {lab: mean([r["test_acc"] for r in runs[(lab, 32)]]) if (lab, 32) in runs else None
            for lab in ("fine", "coarse")}
    full_super = (mean([r["super_acc"] for r in runs[("fine", 32)]])
                  if ("fine", 32) in runs else None)

    print("Q1  accuracy and fraction of full-res skill kept (chance-normalized)")
    print(f"{'res':>4} | {'fine acc':>8} {'kept':>5} | {'coarse-net acc':>14} {'kept':>5} "
          f"| {'fine->super acc':>15} {'kept':>5} | seeds")
    for res in resolutions:
        cells = []
        for lab, key, chance, ref in (("fine", "test_acc", 0.01, full["fine"]),
                                      ("coarse", "test_acc", 0.05, full["coarse"]),
                                      ("fine", "super_acc", 0.05, full_super)):
            rs = runs.get((lab, res))
            if not rs:
                cells.append(("-", "-"))
                continue
            a = mean([r[key] for r in rs])
            cells.append((f"{a:.4f}", f"{norm(a, chance, ref):.2f}"))
        n = len(runs.get(("fine", res), [])), len(runs.get(("coarse", res), []))
        print(f"{res:>4} | {cells[0][0]:>8} {cells[0][1]:>5} | {cells[1][0]:>14} {cells[1][1]:>5} "
              f"| {cells[2][0]:>15} {cells[2][1]:>5} | fine {n[0]}, coarse {n[1]}")
    print("  premise holds on Q1 if 'kept' falls more slowly for coarse than for fine.\n")

    print("Q2  share of the fine net's errors that stay inside a group of 5")
    print(f"{'res':>4} | {'true superclass':>15} | {'random groups mean [95% range]':>32} "
          f"| {'z':>6} | {'excess':>6}")
    supers = np.array(FINE_TO_COARSE)
    for res in resolutions:
        for r in runs.get(("fine", res), []):
            conf = np.array(r["confusion"])
            true_share = within_group_share(conf, supers)
            rnd = random_partition_shares(conf)
            lo, hi = np.percentile(rnd, [2.5, 97.5])
            z = (true_share - rnd.mean()) / rnd.std()
            print(f"{res:>4} | {true_share:>15.3f} | {rnd.mean():>10.3f} [{lo:.3f}, {hi:.3f}]"
                  f"{'':>8} | {z:>6.1f} | {true_share - rnd.mean():>6.3f}  "
                  f"(seed {r['args']['seed']})")
    print("  chance for a structure-free error is 4/99 = 0.040. The premise holds on Q2 if the\n"
          "  excess over random groups is positive at every res and GROWS as res drops.")


if __name__ == "__main__":
    main()
