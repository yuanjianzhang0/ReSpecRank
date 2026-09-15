"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    feature_dim: int = 12
    lookback: int = 20
    graph_window: int = 60
    graph_min_observations: int = 55
    graph_neighbors: int = 10
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    purge_dates: int = 5
    low_frequency_bins: tuple[int, ...] = (0, 1, 2)


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    joint_layers: int = 2
    graph_order: int = 3
    temporal_dilations: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    num_bases: int = 4
    router_hidden_dim: int = 16
    dropout: float = 0.1
    coefficient_init_scale: float = 0.1
    routing_mode: str = "adaptive"


@dataclass
class LossConfig:
    sampled_pairs: int = 4096
    huber_weight: float = 0.1
    diversity_weight: float = 0.01
    huber_delta: float = 1.0
    response_grid_size: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 80
    dates_per_batch: int = 8
    main_learning_rate: float = 1e-3
    router_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    router_warmup_epochs: int = 5
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    early_stopping_patience: int | None = None


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs/respecrank"

    def validate(self) -> None:
        if self.data.lookback <= max(self.model.temporal_dilations):
            raise ValueError("lookback must exceed the largest temporal dilation")
        if self.model.graph_order < 0:
            raise ValueError("graph_order must be non-negative")
        if self.model.joint_layers < 1:
            raise ValueError("joint_layers must be positive")
        if self.model.num_bases < 1:
            raise ValueError("num_bases must be positive")
        if self.model.routing_mode not in {"adaptive", "uniform", "feature", "direct"}:
            raise ValueError("unknown routing_mode")
        if self.model.routing_mode in {"uniform", "feature", "direct"}:
            self.loss.diversity_weight = 0.0
        if len(self.data.split_ratios) != 3 or abs(sum(self.data.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must contain three values that sum to one")
        if self.training.router_warmup_epochs < 0:
            raise ValueError("router_warmup_epochs must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    for name in ("temporal_dilations", "split_ratios", "low_frequency_bins"):
        if name in values:
            values[name] = tuple(values[name])
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        data=_construct(DataConfig, raw.get("data")),
        model=_construct(ModelConfig, raw.get("model")),
        loss=_construct(LossConfig, raw.get("loss")),
        training=_construct(TrainingConfig, raw.get("training")),
        output_dir=raw.get("output_dir", "outputs/respecrank"),
    )
    config.validate()
    return config
