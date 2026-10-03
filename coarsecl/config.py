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
    crossfit: int = 0  # K > 0: K nets, net k never saw train images with uid % K == k;
    # every image (train or test) is labelled by its fold's net. ckpt holds "{fold}".


@dataclass
class NoisyCfg:
    rate: float = 0.0  # fraction of images given a fixed wrong superclass; 0 -> oracle
    seed: int = 0      # corruption draw, independent of run.seed (like a fixed coarse net)


@dataclass
class CoarseCfg:
    source: str = "none"  # none | oracle | soft | noisy | trained
    soft: SoftCfg = field(default_factory=SoftCfg)
    noisy: NoisyCfg = field(default_factory=NoisyCfg)
    trained: TrainedCfg = field(default_factory=TrainedCfg)


@dataclass
class CondCfg:
    mechanism: str = "concat"  # concat | aux_loss | gate | decay  (ignored when source == none)
    embed_dim: int = 32  # concat, decay
    aux_loss_weight: float = 0.5  # aux_loss only
    decay_weight: float = 1.0  # decay only: weight on the KL(teacher||student) reliance term
    eval_withhold: bool = False  # concat ablation: feed uniform (no info) at eval time


@dataclass
class RotationCfg:
    """Self-supervised rotation auxiliary task (approach B): a small head off an
    intermediate trunk tap predicts which of 4 rotations was applied to the input.
    The rotation label is generated at load time and is NOT a function of the fine
    label, so there is genuinely new structure for the shared trunk to internalize.
    Its loss weight anneals to 0, so at the end of training only the fine task
    remains and (with source=none) the run reduces exactly to a_baseline."""

    enabled: bool = False
    lambda_max: float = 0.5
    schedule: str = "per_task"  # per_task (anneal within each task) | curriculum (anneal across tasks)
    tap: str = "layer2"  # ResNetBackbone spatial tap: layer1 | layer2 | layer3
    hidden: int = 128  # keep shallow: force the pressure onto shared features, not the head


@dataclass
class ReplayCfg:
    buffer_size: int = 2000  # total exemplars, split evenly over all seen classes
    batch_size: int = 32  # replay samples drawn per training step
    weight: float = 1.0  # weight on the replay CE term
    # Which exemplars get replayed. none = uniform shuffle (the original runs, bit-
    # for-bit). schema = weight by ROUTING error, 1 - <schema(x), P_fine(superclass|x)>:
    # the fine model's probability mass outside the superclass the schema assigns.
    # hardness = weight by 1 - P_fine(true class): the schema-free control.
    priority: str = "none"  # none | hardness | schema
    priority_schema: str = "oracle"  # schema only: oracle | soft | trained. Independent
    # of coarse.source, so a plain baseline can have schema-guided replay. oracle reads
    # the superclass off the STORED label, so it uses no test-time information.
    priority_mix: float = 0.5  # share of replay draws taken by priority; the rest are
    # uniform, so every exemplar keeps a floor of (1 - mix) / buffer_size.
    priority_refresh: int = 50  # training steps between re-scoring the whole buffer
    priority_level: str = "exemplar"  # exemplar | superclass (schema only): average the
    # scores per superclass and weight every exemplar by its superclass's mean.
    priority_probe_per_class: int = 0  # superclass only: hold out k stored exemplars per
    # class, never replayed, scored instead of the (memorized) replay buffer. Counts
    # against buffer_size.


@dataclass
class EWCCfg:
    ewc_lambda: float = 5000.0
    fisher_batches: int = 32  # batches used to estimate the diagonal Fisher per task


@dataclass
class CLCfg:
    """Standard CL machinery, orthogonal to coarse.source and conditioning.mechanism.
    A list, so `[replay, ewc]` composes; empty (the default) reproduces the original
    no-machinery runs exactly."""

    methods: list = field(default_factory=list)  # subset of {replay, ewc}
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    ewc: EWCCfg = field(default_factory=EWCCfg)


@dataclass
class TrainCfg:
    epochs_per_task: int = 20
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "sgd"  # sgd | adam
    loss_scope: str = "seen"  # seen | current. 'current' masks CE to the current
    # task's classes only, so old-class logits aren't suppressed as negatives
    # (cheapest anti-forgetting head; eval still scores over the full seen set).


@dataclass
class CoarseTrainCfg:
    """Standalone training of the condition-(d) coarse net (train_coarse.py)."""

    arch: str = "small"  # small (the original CNN) | resnet18 (the fine backbone's arch)
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
    rotation: RotationCfg = field(default_factory=RotationCfg)
    cl: CLCfg = field(default_factory=CLCfg)
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
        from .train.cl_methods import CL_METHODS

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
        if not 0.0 <= self.coarse.noisy.rate <= 1.0:
            raise ValueError("coarse.noisy.rate must be in [0, 1]")
        if self.train.loss_scope not in ("seen", "current"):
            raise ValueError("train.loss_scope must be 'seen' or 'current'")
        if not isinstance(self.cl.methods, list):
            raise ValueError("cl.methods must be a list, e.g. [] or [replay, ewc]")
        for m in self.cl.methods:
            if m not in CL_METHODS.keys():
                raise ValueError(f"cl method {m!r} not in {CL_METHODS.keys()}")
        if len(set(self.cl.methods)) != len(self.cl.methods):
            raise ValueError(f"duplicate entries in cl.methods: {self.cl.methods}")
        if "replay" in self.cl.methods and self.cl.replay.buffer_size < NUM_FINE:
            raise ValueError(
                f"cl.replay.buffer_size must be >= {NUM_FINE} (one exemplar per class)"
            )
        rp = self.cl.replay
        if rp.priority not in ("none", "hardness", "schema"):
            raise ValueError("cl.replay.priority must be 'none', 'hardness' or 'schema'")
        if rp.priority_schema not in ("oracle", "soft", "trained"):
            raise ValueError("cl.replay.priority_schema must be 'oracle', 'soft' or 'trained'")
        if not 0.0 <= rp.priority_mix <= 1.0:
            raise ValueError("cl.replay.priority_mix must be in [0, 1]")
        if rp.priority_refresh < 1:
            raise ValueError("cl.replay.priority_refresh must be >= 1")
        if rp.priority_level not in ("exemplar", "superclass"):
            raise ValueError("cl.replay.priority_level must be 'exemplar' or 'superclass'")
        if rp.priority_level == "superclass" and rp.priority != "schema":
            raise ValueError("cl.replay.priority_level: superclass needs priority: schema")
        if rp.priority_probe_per_class < 0:
            raise ValueError("cl.replay.priority_probe_per_class must be >= 0")
        if rp.priority_probe_per_class > 0:
            if rp.priority_level != "superclass":
                raise ValueError("cl.replay.priority_probe_per_class needs priority_level: "
                                 "superclass (probe scores can't map onto other exemplars)")
            if rp.buffer_size // NUM_FINE <= rp.priority_probe_per_class:
                raise ValueError(
                    f"cl.replay.buffer_size // {NUM_FINE} must exceed priority_probe_per_class, "
                    "or no replay exemplars are left once all classes are seen")
        if self.rotation.enabled:
            if self.rotation.schedule not in ("per_task", "curriculum", "constant"):
                raise ValueError("rotation.schedule must be 'per_task', 'curriculum' or 'constant'")
            if self.rotation.tap not in ("layer1", "layer2", "layer3", "layer4"):
                raise ValueError("rotation.tap must be layer1|layer2|layer3|layer4")
            if self.model.backbone != "resnet18_small":
                raise ValueError("rotation aux requires the plastic resnet18_small backbone")
        if self.coarse.trained.crossfit < 0:
            raise ValueError("coarse.trained.crossfit must be >= 0")
        if self.coarse.trained.crossfit and "{fold}" not in self.coarse.trained.ckpt:
            raise ValueError("coarse.trained.crossfit needs a '{fold}' placeholder in ckpt")
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


def load_config(path: str, base_path: Optional[str] = "configs/base.yaml",
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a run YAML, deep-merged onto base.yaml, into a validated Config.

    `overrides` is merged last, after the run YAML, so CLI flags win. It goes
    through the same validation, so an unknown key still raises.
    """
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
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config.from_dict(merged).validate()


def dump_config(cfg: Config, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
