# coarsecl

Does a coarse-category signal reduce catastrophic forgetting when learning
fine-grained classes sequentially? A minimal Class-IL experiment on CIFAR-100.
See [CLAUDE.md](CLAUDE.md) for the project summary and [DESIGN.md](DESIGN.md) for
the design rationale.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the four conditions

```bash
# (a) baseline, (b) oracle, (c) soft proxy
.venv/bin/python -m coarsecl.run --config configs/a_baseline.yaml
.venv/bin/python -m coarsecl.run --config configs/b_oracle.yaml
.venv/bin/python -m coarsecl.run --config configs/c_soft.yaml

# (d) trained coarse classifier: train the coarse net once, then run
.venv/bin/python -m coarsecl.coarse.train_coarse --config configs/d_trained.yaml
.venv/bin/python -m coarsecl.run --config configs/d_trained.yaml
```

Each run writes `results/<name>/{config.yaml, metrics.json, acc_matrix.csv}`.

## Compare

```bash
.venv/bin/python -m coarsecl.plot \
  --runs results/a_baseline results/b_oracle results/c_soft results/d_trained \
  --out results/comparison.png
```

## Quick pipeline check

```bash
.venv/bin/python -m coarsecl.run --config configs/smoke.yaml   # 2 tasks, 1 epoch
```

## Knobs that matter

| What to change            | Where                                   |
|---------------------------|-----------------------------------------|
| Condition                 | `coarse.source` (none/oracle/soft/trained) |
| Conditioning mechanism    | `conditioning.mechanism` (concat/aux_loss) |
| Soft-proxy fidelity       | `coarse.soft.smooth` (0=oracle … 1=uniform) |
| Task split                | `data.num_tasks`, `data.classes_per_task` |
| Epochs / LR / batch       | `train.*`                               |

Adding a fifth condition = one new `CoarseSource` in `coarsecl/coarse/sources.py`
plus a config value. The training loop does not change.
