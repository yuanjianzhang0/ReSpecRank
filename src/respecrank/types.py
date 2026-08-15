"""Typed containers used across data, model, and training code."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DateBatch:
    """One complete cross-section for a prediction date."""

    date: str
    symbols: tuple[str, ...]
    features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    market_state: torch.Tensor
    targets: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> DateBatch:
        return DateBatch(
            date=self.date,
            symbols=self.symbols,
            features=self.features.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            market_state=self.market_state.to(device),
            targets=None if self.targets is None else self.targets.to(device),
        )

    @property
    def num_stocks(self) -> int:
        return int(self.features.shape[0])

