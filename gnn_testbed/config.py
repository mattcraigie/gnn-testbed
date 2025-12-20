from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, get_type_hints

import yaml

from gnn_testbed.data.pointcloud import PairFieldConfig, TriangleFieldConfig


@dataclass
class DataSplitConfig:
    size: int = 100
    seed: int = 0
    batch_size: int = 128
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class DataConfig:
    task: str = "triangle"  # ["triangle", "pair"]
    pair: PairFieldConfig = field(default_factory=PairFieldConfig)
    triangle: TriangleFieldConfig = field(default_factory=TriangleFieldConfig)
    signal_points: int = 96
    noise_points: int = 32
    pair_distance: float = 1.0
    normalize: str = "box"
    jitter_std: float = 0.01
    jitter_clip: Optional[float] = 0.05
    train: DataSplitConfig = field(
        default_factory=lambda: DataSplitConfig(
            size=100, seed=123, batch_size=128, shuffle=True, drop_last=True
        )
    )
    val: DataSplitConfig = field(
        default_factory=lambda: DataSplitConfig(
            size=100, seed=456, batch_size=256, shuffle=False, drop_last=False
        )
    )
    test: DataSplitConfig = field(
        default_factory=lambda: DataSplitConfig(
            size=100, seed=789, batch_size=256, shuffle=False, drop_last=False
        )
    )


@dataclass
class ModelConfig:
    type: str = "lss_gnn"  # ["lss_gnn", "standard_egnn", "triangle_lss", "simple_point_mlp"]
    in_dim: int = 2
    hidden: int = 128

    # GNN options
    hidden_dim: int = 64
    num_layers: int = 2
    spin_symmetry: int = 2  # Used by LSS-GNN only
    graph_mode: str = "knn"
    k: int = 5
    tri_graph_mode: str = "adjacent"
    k_tri: int = 6


@dataclass
class TrainingConfig:
    device: str = "cuda"
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True

    scheduler: str = "cosine"  # ["cosine", "step", "none"]
    warmup_epochs: int = 5
    step_size: int = 50
    gamma: float = 0.5
    min_lr: float = 1e-6

    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    monitor: str = "val_loss"  # ["val_loss", "val_acc", "val_f1"]
    mode: str = "min"  # ["min", "max"]

    work_dir: str = "./runs/pointcloud"
    save_best_only: bool = True

    threshold: float = 0.5

    tb_flush_secs: int = 10
    tb_log_histograms: bool = False
    tb_log_grads: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _get_default_value(field_obj):
    if field_obj.default is not MISSING:
        return field_obj.default
    if field_obj.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field_obj.default_factory()
    raise KeyError(f"Field {field_obj.name} requires a value")


def dataclass_from_dict(cls, data: Dict[str, Any]):
    data = data or {}
    kwargs = {}
    type_hints = get_type_hints(cls)

    for f in fields(cls):
        target_type = type_hints.get(f.name, f.type)
        if f.name in data:
            value = data[f.name]
            if is_dataclass(target_type):
                kwargs[f.name] = dataclass_from_dict(target_type, value)
            else:
                kwargs[f.name] = value
        else:
            kwargs[f.name] = _get_default_value(f)
    return cls(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = dataclass_from_dict(DataConfig, raw.get("data", {}))
    model_cfg = dataclass_from_dict(ModelConfig, raw.get("model", {}))
    training_cfg = dataclass_from_dict(TrainingConfig, raw.get("training", {}))

    return ExperimentConfig(data=data_cfg, model=model_cfg, training=training_cfg)


__all__ = [
    "DataSplitConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_experiment_config",
    "dataclass_from_dict",
    "PairFieldConfig",
    "TriangleFieldConfig",
]
