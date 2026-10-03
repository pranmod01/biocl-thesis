"""Paired seed analysis for the Item 12 sweep.

Each arm is paired with the replay-2000 control at the SAME training seed (the task
stream is identical across seeds, since only run.seed varies), so the test is a paired
one over seeds. That removes the seed-to-seed difficulty of the stream, which is the
dominant nuisance term, and is why 4 seeds can say something a 4-sample unpaired band
could not.

Reported per metric: each seed's delta (arm - control), the mean delta with its
standard error, a paired t with df = n-1, and a two-sided p from a t-CDF computed by
continued fraction (no scipy in the venv).

Usage (from the repo root):  .venv/bin/python results/_jobs/seed_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import sys

CONTROL = "i_replay_baseline_hippo"
ARMS = ["i_replay_decay_oracle", "i_replay_aux_oracle", "i_replay_gate_trained"]
SEEDS = [0, 1, 2, 3]
METRICS = ["average_accuracy", "retained_accuracy", "average_incremental_accuracy"]
SHORT = {"average_accuracy": "final avg", "retained_accuracy": "retained",
         "average_incremental_accuracy": "AIA"}

# Extra same-config seed-0 samples of the control, from earlier jobs. They quantify
# pure run-to-run noise (identical config, identical seed) and are NOT paired samples.
UNPAIRED_CONTROL = ["i_replay_baseline", "i_replay_cal2000", "i_replay_prio_mix0"]


def run_dir(name: str, seed: int) -> str:
    return f"results/{name}" if seed == 0 else f"results/{name}_s{seed}"


def load(name: str, seed: int):
    path = os.path.join(run_dir(name, seed), "metrics.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))["metrics"]


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: int) -> float:
    """Two-sided p for Student's t."""
    if df <= 0:
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def main() -> None:
    ctrl = {s: load(CONTROL, s) for s in SEEDS}
    seeds = [s for s in SEEDS if ctrl[s] is not None]
    if not seeds:
        print(f"control {CONTROL} has no runs; run the sweep first.")
        sys.exit(1)
    pending = [s for s in SEEDS if ctrl[s] is None]
    if pending:
        print(f"NOTE: control seeds {pending} not finished; pairing on {seeds} only.\n")

    print(f"Paired seed analysis — control = {CONTROL}, seeds {seeds}\n")
    for metric in METRICS:
        print(f"== {SHORT[metric]} ({metric})")
        cvals = [ctrl[s][metric] for s in seeds]
        print("   control    " + "  ".join(f"s{s}={v:.3f}" for s, v in zip(seeds, cvals))
              + f"   mean {sum(cvals)/len(cvals):.3f}")
        for arm in ARMS:
            vals, deltas, got = [], [], []
            for s in seeds:
                m = load(arm, s)
                if m is None:
                    continue
                got.append(s)
                vals.append(m[metric])
                deltas.append(m[metric] - ctrl[s][metric])
            if not deltas:
                print(f"   {arm:26} no runs found")
                continue
            n = len(deltas)
            mean = sum(deltas) / n
            tag = f"{arm.replace('i_replay_', ''):18}"
            detail = "  ".join(f"s{s}={v:+.3f}" for s, v in zip(got, deltas))
            if n < 2:
                print(f"   {tag} {detail}   mean {mean:+.3f}  (n=1, no test)")
                continue
            sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
            sem = sd / math.sqrt(n)
            t = mean / sem if sem > 0 else float("inf")
            p = t_pvalue(t, n - 1)
            verdict = "SIG" if p < 0.05 else "ns "
            print(f"   {tag} {detail}   mean {mean:+.3f} ± {sem:.3f} (sem)  "
                  f"t={t:+.2f} df={n-1} p={p:.3f}  {verdict}")
        print()

    extra = [(r, load(r, 0)) for r in UNPAIRED_CONTROL]
    extra = [(r, m) for r, m in extra if m]
    if extra:
        print("Reference — identical-config seed-0 control samples (pure run-to-run noise):")
        for r, m in extra:
            print(f"   {r:30} avg={m['average_accuracy']:.3f} "
                  f"ret={m['retained_accuracy']:.3f} "
                  f"aia={m['average_incremental_accuracy']:.3f}")


if __name__ == "__main__":
    main()
