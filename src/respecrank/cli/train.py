"""Train ReSpecRank on a prepared local dataset."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from respecrank.config import load_config
from respecrank.data import CrossSectionDataset, collate_dates
from respecrank.model import ReSpecRank
from respecrank.training import train_model
from respecrank.utils import count_parameters, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_dataset = CrossSectionDataset(config.data.processed_dir, "train")
    validation_dataset = CrossSectionDataset(config.data.processed_dir, "validation")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.dates_per_batch,
        shuffle=True,
        collate_fn=collate_dates,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.dates_per_batch,
        shuffle=False,
        collate_fn=collate_dates,
        num_workers=config.training.num_workers,
    )
    model = ReSpecRank.from_config(config.data, config.model)
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": count_parameters(model),
                "train_dates": len(train_dataset),
                "validation_dates": len(validation_dataset),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_model(model, train_loader, validation_loader, config, device)


if __name__ == "__main__":
    main()

