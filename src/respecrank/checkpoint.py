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
    config = config_from_dict(payload["config"])
    model = ReSpecRank.from_config(config.data, config.model)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, config, payload

