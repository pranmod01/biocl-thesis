# coarsecl — project summary

A minimal continual-learning experiment testing whether **coarse-grained category
knowledge reduces catastrophic forgetting** when fine-grained classes are learned
sequentially. An existence-proof before any complex architecture work.

Dataset: CIFAR-100, using its native 20-superclass (coarse) / 100-subclass (fine)
hierarchy. Setup is class-incremental — the 100 fine classes are split into N
sequential tasks (e.g. 10 × 10), no task ID at test time, and **no replay buffer**,
so the only thing under test is the coarse signal itself.

## Four conditions (same fine backbone, directly comparable)
- **(a) baseline** — fine classifier trained continually, no coarse information.
- **(b) oracle** — also receives the ground-truth coarse label (one-hot). A
  sanity-check upper bound; if it doesn't beat baseline, that's a bug to debug.
- **(c) soft proxy** — a synthetic coarse distribution, most mass on the true
  class, fidelity set by one `smooth` knob (oracle → uniform). Sweepable.
- **(d) trained** — softmax output of a separate, independent coarse CNN trained
  upfront on the coarse labels.

The hypothesis: coarse signal, at varying fidelity (oracle → soft proxy → real
trained classifier), measurably reduces forgetting vs. no signal — and the benefit
degrades as the signal gets less perfect. Communication is one-directional: the
coarse signal informs the fine classifier only, nothing flows back.

## How it's wired
The coarse signal is always a distribution over the 20 superclasses (or `None` for
baseline) and enters the fine classifier through a swappable conditioning
mechanism. Two orthogonal config axes — `coarse.source` (`none`/`oracle`/`soft`/
`trained`) and `conditioning.mechanism` (`concat`/`aux_loss`) — are both resolved
from a registry, so adding a condition or mechanism is a config change, not a
training-loop rewrite.

## Evaluation & metrics
After each task, fine-grained accuracy over all classes seen so far (Class-IL).
Reported: average accuracy at end of training, and average forgetting (max-minus-
final accuracy per task, averaged), plotted for all four conditions on one chart.

## Layout
- `coarsecl/` — package: `data/`, `coarse/`, `models/`, `train/`, plus
  `metrics.py`, `plot.py`, `logging_io.py`, `run.py`.
- `configs/` — one YAML per run (`a_baseline`, `b_oracle`, `c_soft`, `d_trained`),
  all inheriting `base.yaml`.
- `results/<run>/` — resolved config + metrics (JSON/CSV, config embedded).

## Constraints
Keep it simple: no predictive coding, no sparse autoencoders, no replay. Structure
the code so a fifth condition later (e.g. continually-trained coarse classifier, or
a noisy-label oracle variant) is a config change plus a new condition class.

**Design choices and their rationale: [DESIGN.md](DESIGN.md). Usage: [README.md](README.md).**
