"""Prepare a local point-in-time dataset."""

from __future__ import annotations

import argparse
import json

from respecrank.preprocessing import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Preparation YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(args.config)
    summary = {split: len(files) for split, files in manifest["splits"].items()}
    print(json.dumps({"dates": summary, "metadata": manifest["metadata"]}, indent=2))


if __name__ == "__main__":
    main()

