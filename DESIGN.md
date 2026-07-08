# Design choices

The resolved implementation decisions for `coarsecl` and the rationale behind each.
CLAUDE.md is the project summary; this file records *why* the code is built the way
it is. Settled 2026-06-23.

## Unifying abstraction: the coarse signal is always a distribution
In every condition the coarse signal is a distribution over the 20 superclasses,
`FloatTensor[B, 20]` (or `None` for baseline) — never a discrete label. This is
what makes the four conditions interchangeable behind one interface. The oracle is
just the one-hot special case.

## Two orthogonal axes, both resolved from config strings via a registry
- **Axis 1 — `coarse.source`** (`coarsecl/coarse/sources.py`): `none` (baseline) |
  `oracle` (one-hot of true superclass) | `soft` (synthetic, see below) | `trained`
  (frozen coarse CNN softmax).
- **Axis 2 — `conditioning.mechanism`** (`coarsecl/models/conditioning.py`):
  `concat` (project the 20-vector through a learned coarse-embedding matrix
  `E[20, embed_dim]` — `dist @ E` — and concatenate to the penultimate layer;
  coarse signal used as an INPUT) | `aux_loss` (auxiliary coarse head off the
  penultimate features, soft-target cross-entropy against the distribution; coarse
  signal used as a TARGET).

The trainer is generic and names no concrete variant. When `coarse.source` is
`none`, the factory selects a `none` conditioning (identity) regardless of the
mechanism field, since the mechanism is meaningless without a signal. Adding a
fifth condition = one new registered `CoarseSource` class + one config value, no
training-loop edit.

## Soft distribution proxy (condition c) = mass-mixing, single knob
`dist = (1 - smooth) * one_hot(true_superclass) + smooth * uniform`, controlled by
one parameter `coarse.soft.smooth`: 0 → one-hot (= oracle), 1 → uniform (= no
info). Chosen over label-flip because (c) is literally a "soft distribution with
most mass on the true class." Label-flip (a confidently-wrong peak) is a DIFFERENT
corruption, deferred to the future "noisy oracle" fifth condition as its own
`CoarseSource`, not folded into (c).

## Class-incremental head: fixed 100-way
Single fixed 100-unit fine head for all tasks (not a physically growing head).
Training loss is masked to the classes seen so far (future-unseen logits set to
-inf); evaluation argmaxes over the seen-class set only. Same choice in all four
conditions, so it cannot bias the baseline-vs-coarse comparison.

## Condition (d) coarse net: trained once upfront, frozen
The independent coarse CNN (`coarsecl/models/coarse_net.py`) is trained once on all
superclasses via `coarsecl/coarse/train_coarse.py`, saved to a checkpoint, and
loaded frozen during the continual run. `coarse.trained.continual` exists as a
dormant config flag (only the `false` branch is implemented) for the future
continually-trained variant.

## Other settled defaults
- **Backbone:** small CIFAR ResNet-18 (3×3 stem, no max-pool), 512-dim penultimate
  features, shared identically across all conditions and mechanisms.
- **Continual procedure:** no replay; a fresh optimizer per task; train only on the
  current task's data (older classes act as negatives via the masked CE).
- **Coarse signal at eval:** provided by the same source used in training (for
  `oracle`/`soft` this means using the test sample's true superclass — the intended
  upper-bound framing; for `trained` the coarse net runs on the test image, no
  leak). Under `aux_loss` the fine prediction ignores the signal at inference.
