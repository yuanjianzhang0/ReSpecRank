"""Evaluate a ReSpecRank checkpoint and export date-level predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from respecrank.checkpoint import load_checkpoint
from respecrank.data import CrossSectionDataset, collate_dates
from respecrank.evaluation import evaluate_model
from respecrank.utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, config, payload = load_checkpoint(args.checkpoint, device)
    processed_dir = args.processed_dir or config.data.processed_dir
    dataset = CrossSectionDataset(processed_dir, args.split)
    loader = DataLoader(
        dataset,
        batch_size=config.training.dates_per_batch,
        shuffle=False,
        collate_fn=collate_dates,
    )
    result = evaluate_model(model, loader, device)
    output_dir = Path(args.output_dir or config.output_dir) / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    date_rows = []
    for date_result in result.dates:
        date_rows.append(
            {
                "date": date_result.date,
                "rank_ic": date_result.rank_ic,
                "ic": date_result.ic,
                "temporal_centroid": date_result.temporal_centroid,
                "graph_centroid": date_result.graph_centroid,
                **{
                    f"router_weight_{index}": float(value)
                    for index, value in enumerate(date_result.router_weights)
                },
            }
        )
        rows.extend(
            {
                "date": date_result.date,
                "symbol": symbol,
                "score": float(score),
                "target": float(target),
            }
            for symbol, score, target in zip(
                date_result.symbols,
                date_result.scores,
                date_result.targets,
                strict=True,
            )
        )
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(date_rows).to_csv(output_dir / "daily_metrics.csv", index=False)
    summary = {
        **result.summary.to_dict(),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": args.split,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

