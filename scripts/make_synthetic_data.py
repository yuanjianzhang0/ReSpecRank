"""Create a deterministic local dataset for pipeline smoke tests.

The generated series are not intended for scientific evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/synthetic_raw")
    parser.add_argument("--dates", type=int, default=180)
    parser.add_argument("--stocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dates < 100 or args.stocks < 3:
        raise ValueError("use at least 100 dates and 3 stocks")
    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range("2020-01-02", periods=args.dates)
    symbols = [f"S{index:04d}" for index in range(args.stocks)]
    market = rng.normal(0.0003, 0.01, size=args.dates)
    rows = []
    for stock_index, symbol in enumerate(symbols):
        loading = 0.6 + 0.3 * stock_index / max(args.stocks - 1, 1)
        returns = loading * market + rng.normal(0.0, 0.008, size=args.dates)
        close = 50.0 * np.exp(np.cumsum(returns))
        overnight = rng.normal(0.0, 0.002, size=args.dates)
        open_price = close * np.exp(overnight)
        spread = np.abs(rng.normal(0.006, 0.002, size=args.dates))
        high = np.maximum(open_price, close) * (1.0 + spread)
        low = np.minimum(open_price, close) * (1.0 - spread)
        volume = rng.lognormal(14.0, 0.25, size=args.dates)
        turnover = volume * close / 1e9
        rows.extend(
            {
                "date": date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
                "volume": volume_value,
                "turnover": turnover_value,
                "tradable": True,
            }
            for (
                date,
                open_value,
                high_value,
                low_value,
                close_value,
                volume_value,
                turnover_value,
            ) in zip(
                dates,
                open_price,
                high,
                low,
                close,
                volume,
                turnover,
                strict=True,
            )
        )
    memberships = [
        {"date": date.strftime("%Y-%m-%d"), "symbol": symbol, "is_member": True}
        for date in dates
        for symbol in symbols
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "prices.csv", index=False)
    pd.DataFrame(memberships).to_csv(output / "membership.csv", index=False)
    print(f"Wrote {len(rows)} price rows and {len(memberships)} membership rows to {output}")


if __name__ == "__main__":
    main()
