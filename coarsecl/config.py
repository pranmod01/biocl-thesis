"""Typed config schema (nested dataclasses) + YAML load/merge/validate.

One YAML per run, inheriting `configs/base.yaml`. The resolved config is dumped
verbatim into each run's results dir, so every result is self-describing.
"""

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional

import yaml

NUM_FINE = 100
NUM_COARSE = 20


@dataclass
class RunCfg:
    name: str = "run"
    seed: int = 0
    output_dir: str = "results"
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class DataCfg:
    dataset: str = "cifar100"
    num_tasks: int = 10
    classes_per_task: int = 10
    order_seed: int = 0
    data_root: str = "data"
    num_workers: int = 4


@dataclass
class ModelCfg:
    backbone: str = "resnet18_small"
    feature_dim: int = 512


@dataclass
class SoftCfg:
    smooth: float = 0.3  # 0 -> one-hot (oracle), 1 -> uniform (no info)


@dataclass
class TrainedCfg:
    ckpt: str = "checkpoints/coarse_net.pt"
    continual: bool = False  # dormant hook for the continually-trained variant


@dataclass
class CoarseCfg:
    source: str = "none"  # none | oracle | soft | trained
    soft: SoftCfg = field(default_factory=SoftCfg)
    trained: TrainedCfg = field(default_factory=TrainedCfg)


@dataclass
class CondCfg:
    mechanism: str = "concat"  # concat | aux_loss  (ignored when source == none)
    embed_dim: int = 32  # concat only
    aux_loss_weight: float = 0.5  # aux_loss only


@dataclass
class TrainCfg:
    epochs_per_task: int = 20
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "sgd"  # sgd | adam


@dataclass
class CoarseTrainCfg:
    """Standalone training of the condition-(d) coarse net (train_coarse.py)."""

    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    coarse: CoarseCfg = field(default_factory=CoarseCfg)
    conditioning: CondCfg = field(default_factory=CondCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    coarse_train: CoarseTrainCfg = field(default_factory=CoarseTrainCfg)

    # -- (de)serialization -------------------------------------------------
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        return _build(Config, d or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> "Config":
        from .coarse.sources import COARSE_SOURCES
        from .models.conditioning import CONDITIONING

        if self.data.num_tasks * self.data.classes_per_task != NUM_FINE:
            raise ValueError(
                f"num_tasks * classes_per_task must equal {NUM_FINE}, got "
                f"{self.data.num_tasks} * {self.data.classes_per_task}"
            )
        if self.coarse.source not in COARSE_SOURCES.keys():
            raise ValueError(
                f"coarse.source {self.coarse.source!r} not in {COARSE_SOURCES.keys()}"
            )
        if self.coarse.source != "none" and self.conditioning.mechanism not in CONDITIONING.keys():
            raise ValueError(
                f"conditioning.mechanism {self.conditioning.mechanism!r} "
                f"not in {CONDITIONING.keys()}"
            )
        if not 0.0 <= self.coarse.soft.smooth <= 1.0:
            raise ValueError("coarse.soft.smooth must be in [0, 1]")
        if self.coarse.trained.continual:
            raise NotImplementedError(
                "coarse.trained.continual is a dormant hook; only the upfront-"
                "trained (false) branch is implemented."
            )
        return self


def _build(cls, d: Dict[str, Any]):
    """Recursively construct a (possibly nested) dataclass from a plain dict,
    rejecting unknown keys so config typos fail loudly."""
    if not is_dataclass(cls):
        return d
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    for key, val in d.items():
        if key not in field_map:
            raise ValueError(f"unknown config key {key!r} for {cls.__name__}")
        f = field_map[key]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[key] = _build(f.type, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, base_path: Optional[str] = "configs/base.yaml") -> Config:
    """Load a run YAML, deep-merged onto base.yaml, into a validated Config."""
    merged: Dict[str, Any] = {}
    if base_path:
        try:
            with open(base_path) as fh:
                merged = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            merged = {}
    with open(path) as fh:
        run_cfg = yaml.safe_load(fh) or {}
    merged = _deep_merge(merged, run_cfg)
    return Config.from_dict(merged).validate()


def dump_config(cfg: Config, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
