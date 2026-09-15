"""Processed point-in-time cross-section dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from respecrank.types import DateBatch
from respecrank.utils import read_json


class CrossSectionDataset(Dataset[DateBatch]):
    """Load variable-sized dates without padding their stock universes."""

    def __init__(self, processed_dir: str | Path, split: str) -> None:
        self.root = Path(processed_dir)
        manifest = read_json(self.root / "manifest.json")
        if split not in manifest["splits"]:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.files = tuple(manifest["splits"][split])
        self.metadata = manifest.get("metadata", {})
        if self.metadata.get("implementation_version") != 2:
            raise ValueError("rebuild legacy data with version 2 past-only candidate selection")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> DateBatch:
        path = self.root / self.files[index]
        with np.load(path, allow_pickle=False) as item:
            targets = torch.from_numpy(item["targets"].astype(np.float32, copy=False))
            return DateBatch(
                date=str(item["date"].item()),
                symbols=tuple(str(value) for value in item["symbols"].tolist()),
                features=torch.from_numpy(item["features"].astype(np.float32, copy=False)),
                edge_index=torch.from_numpy(item["edge_index"].astype(np.int64, copy=False)),
                edge_weight=torch.from_numpy(item["edge_weight"].astype(np.float32, copy=False)),
                market_state=torch.from_numpy(
                    item["market_state"].astype(np.float32, copy=False)
                ),
                targets=targets,
            )


def collate_dates(items: list[DateBatch]) -> list[DateBatch]:
    """Preserve native universes instead of padding a mini-batch."""
    return items

