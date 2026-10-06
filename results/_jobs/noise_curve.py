"""Item 22: controlled-noise dose-response for the coarse signal, with replay 2000.

x = coarse-signal error rate (fraction of images whose superclass is wrong):
  0      -> i_replay_<mech>_oracle
  r      -> n_replay_<mech>_noisy<r>  (fixed per-image corruption, nested across r)
  0.95   -> no information (concat only)
Trained coarse nets are placed at their measured TEST error (1 - ckpt test_acc; the
mean over folds for cross-fitted nets), as
hollow markers, so you can see whether a real net sits on, above or below the
controlled-noise curve at its own error rate.

The grey band is the replay-2000 control (no coarse signal): mean ± sd over every
i_replay_baseline* sample available.

Gate caveat: gate masks predictions to the coarse signal's superclass at test time,
so its raw accuracy includes output-space arithmetic (see metrics.py). Noisy and
trained gate runs have no closed-form null, so the gate curve is raw accuracy;
compare gate points with each other, not against the concat curve.

Usage (from the repo root):  .venv/bin/python results/_jobs/noise_curve.py
Writes results/noise_curve.png.
"""

from __future__ import annotations

import glob
import json
import os
import statistics

import torch

MECHS = ["concat", "gate"]
RATES = [0.10, 0.20, 0.34, 0.50, 0.95]
METRICS = [("retained_accuracy", "retained"), ("average_incremental_accuracy", "AIA"),
           ("average_accuracy", "final avg")]
TRAINED = {  # run suffix -> (label, checkpoint(s); "{fold}" = cross-fitted, averaged)
    "trained": ("small CNN", "checkpoints/coarse_net.pt"),
    "trained_r18": ("ResNet-18, memorized train", "checkpoints/coarse_net_r18.pt"),
    "trained_r18cf": ("ResNet-18, cross-fit", "checkpoints/coarse_net_r18_cf4_f{fold}.pt"),
}
MARKER = {"small CNN": "D", "ResNet-18, memorized train": "s", "ResNet-18, cross-fit": "^"}
CONTROL_GLOB = ["results/i_replay_baseline_hippo*", "results/i_replay_baseline",
                "results/i_replay_cal2000", "results/i_replay_prio_mix0"]

COLOR = {"concat": "#2a78d6", "gate": "#eb6834"}  # reference palette slots 1-2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load(run: str):
    path = os.path.join("results", run, "metrics.json")
    return json.load(open(path))["metrics"] if os.path.exists(path) else None


def ckpt_err(path: str):
    paths = sorted(glob.glob(path.replace("{fold}", "*"))) if "{fold}" in path else [path]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    accs = [float(torch.load(p, map_location="cpu")["test_acc"]) for p in paths]
    return 1.0 - sum(accs) / len(accs)


def main() -> None:
    controls = sorted({d for g in CONTROL_GLOB for d in glob.glob(g)})
    ctrl = {k: [load(os.path.basename(d))[k] for d in controls
                if load(os.path.basename(d))] for k, _ in METRICS}

    curves = {}  # mech -> [(rate, metrics)]
    trained = {}  # mech -> [(label, err, metrics)]
    for m in MECHS:
        pts = [(0.0, load(f"i_replay_{m}_oracle"))]
        pts += [(r, load(f"n_replay_{m}_noisy{round(r * 100):02d}")) for r in RATES]
        curves[m] = [(r, mm) for r, mm in pts if mm]
        trained[m] = []
        for suffix, (label, ck) in TRAINED.items():
            run = f"i_replay_{m}_{suffix}" if suffix == "trained" else f"n_replay_{m}_{suffix}"
            mm, err = load(run), ckpt_err(ck)
            if mm and err is not None:
                trained[m].append((label, err, mm))

    print(f"control (no coarse), n={len(controls)}: " + "  ".join(
        f"{s} {statistics.mean(ctrl[k]):.3f}±{statistics.stdev(ctrl[k]):.3f}"
        for k, s in METRICS))
    for m in MECHS:
        print(f"\n== {m}")
        print(f"  {'signal':<22}{'err':>6}" + "".join(f"{s:>11}" for _, s in METRICS))
        for r, mm in curves[m]:
            name = "oracle" if r == 0 else f"noisy {r:.2f}"
            print(f"  {name:<22}{r:>6.2f}" + "".join(f"{mm[k]:>11.3f}" for k, _ in METRICS))
        for label, err, mm in trained[m]:
            print(f"  {'trained ' + label:<22}{err:>6.2f}"
                  + "".join(f"{mm[k]:>11.3f}" for k, _ in METRICS))
        # at the trained net's error rate, what does the controlled-noise curve predict?
        xs = [r for r, _ in curves[m]]
        for label, err, mm in trained[m]:
            if len(xs) < 2 or not xs[0] <= err <= xs[-1]:
                continue
            k = "retained_accuracy"
            ys = [c[k] for _, c in curves[m]]
            i = max(j for j in range(len(xs)) if xs[j] <= err)
            i = min(i, len(xs) - 2)
            pred = ys[i] + (ys[i + 1] - ys[i]) * (err - xs[i]) / (xs[i + 1] - xs[i])
            print(f"  -> {label} at err {err:.2f}: noise curve predicts retained "
                  f"{pred:.3f}, got {mm[k]:.3f} ({mm[k] - pred:+.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (k, short) in zip(axes, METRICS[:2]):
        ax.set_facecolor(SURFACE)
        mu, sd = statistics.mean(ctrl[k]), statistics.stdev(ctrl[k])
        ax.axhspan(mu - sd, mu + sd, color="#d9d8d3", alpha=0.6, lw=0)
        ax.axhline(mu, color=INK2, lw=1, ls="--")
        ax.text(0.02, mu - sd, "no coarse signal (mean ± sd)", color=INK2, fontsize=8,
                ha="left", va="bottom", transform=ax.get_yaxis_transform())
        for m in MECHS:
            xs = [r for r, _ in curves[m]]
            ys = [mm[k] for _, mm in curves[m]]
            ax.plot(xs, ys, color=COLOR[m], lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{m}, controlled noise")
            if xs:
                ax.annotate(m, (xs[0], ys[0]), xytext=(6, 4), textcoords="offset points",
                            color=INK, fontsize=9)
            for label, err, mm in trained[m]:
                ax.plot([err], [mm[k]], marker=MARKER[label], ms=8, mfc=SURFACE,
                        mec=COLOR[m], mew=2, ls="none")
        ax.axvline(0.95, color=GRID, lw=1)
        ax.set_xlabel("coarse-signal error rate (fraction of images, wrong superclass)",
                      color=INK2, fontsize=9)
        ax.set_title(short, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    handles = [plt.Line2D([], [], color=COLOR[m], lw=2, marker="o", label=f"{m}, controlled noise")
               for m in MECHS]
    handles += [plt.Line2D([], [], color=INK2, marker=mk, mfc=SURFACE, mew=1.5, ls="none",
                           label=f"trained: {label}") for label, mk in MARKER.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.text(0.5, 0.005, "Hollow markers: trained coarse nets at their test error, "
             "colored by mechanism. Replay 2000, seed 0.", ha="center", color=INK2,
             fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = "results/noise_curve.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
