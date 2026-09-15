"""Safe model reconstruction from project checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from respecrank.config import ExperimentConfig, config_from_dict
from respecrank.model import ReSpecRank


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ReSpecRank, ExperimentConfig, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("implementation_version") != 2:
        raise ValueError(
            "legacy checkpoint uses stacked temporal filtering; retrain with version 2"
        )
    config = config_from_dict(payload["config"])
    model = ReSpecRank.from_config(config.data, config.model)
    model.load_state_dict(payload["model_state"])
    model.checkpoint_warmup = payload["epoch"] <= config.training.router_warmup_epochs
    model.to(device)
    return model, config, payload
