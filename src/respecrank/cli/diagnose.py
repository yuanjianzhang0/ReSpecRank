"""Run response-centroid and state-shuffle diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from respecrank.checkpoint import load_checkpoint
from respecrank.data import CrossSectionDataset, collate_dates
from respecrank.evaluation import collect_shuffled_states, evaluate_model
from respecrank.utils import resolve_device, write_json


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, config, _ = load_checkpoint(args.checkpoint, device)
    dataset = CrossSectionDataset(args.processed_dir or config.data.processed_dir, "test")
    loader = DataLoader(
        dataset,
        batch_size=config.training.dates_per_batch,
        shuffle=False,
        collate_fn=collate_dates,
    )
    ordinary = evaluate_model(model, loader, device)
    shuffled_states = collect_shuffled_states(loader, args.seed)
    shuffled = evaluate_model(model, loader, device, shuffled_states=shuffled_states)
    output_dir = Path(args.output_dir or config.output_dir) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    response_frame = pd.DataFrame(
        [
            {
                "date": result.date,
                "rank_ic": result.rank_ic,
                "ic": result.ic,
                "temporal_centroid": result.temporal_centroid,
                "graph_centroid": result.graph_centroid,
                "normalized_volatility": float(result.market_state[1]),
                "normalized_dispersion": float(result.market_state[2]),
            }
            for result in ordinary.dates
        ]
    )
    tercile_labels = pd.qcut(
        response_frame["normalized_volatility"].rank(method="first"),
        q=3,
        labels=("low", "mid", "high"),
    )
    response_frame["volatility_tercile"] = tercile_labels
    response_frame.to_csv(output_dir / "response_by_date.csv", index=False)
    terciles = {
        str(label): {
            "num_dates": int(len(group)),
            "rank_ic": float(group["rank_ic"].mean()),
            "temporal_centroid": float(group["temporal_centroid"].mean()),
            "graph_centroid": float(group["graph_centroid"].mean()),
        }
        for label, group in response_frame.groupby("volatility_tercile", observed=True)
    }
    summary = {
        "ordinary": ordinary.summary.to_dict(),
        "state_shuffle": shuffled.summary.to_dict(),
        "rank_ic_drop": ordinary.summary.rank_ic - shuffled.summary.rank_ic,
        "shuffle_seed": args.seed,
        "volatility_temporal_centroid_correlation": _correlation(
            response_frame["normalized_volatility"].to_numpy(),
            response_frame["temporal_centroid"].to_numpy(),
        ),
        "dispersion_graph_centroid_correlation": _correlation(
            response_frame["normalized_dispersion"].to_numpy(),
            response_frame["graph_centroid"].to_numpy(),
        ),
        "volatility_terciles": terciles,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
