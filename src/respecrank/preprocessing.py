"""Local-data preprocessing for point-in-time stock ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from respecrank.graph import positive_knn_adjacency, scaled_laplacian_edges
from respecrank.utils import write_json

FEATURE_NAMES = (
    "return_1d",
    "return_5d",
    "return_10d",
    "open_close_relative",
    "high_close_relative",
    "low_close_relative",
    "volume_change",
    "turnover",
    "realized_volatility_5d",
    "realized_volatility_20d",
    "momentum_5d",
    "ma5_ma20_ratio",
)


@dataclass
class PreparationConfig:
    prices_csv: str
    membership_csv: str
    output_dir: str
    columns: dict[str, str]
    lookback: int = 20
    graph_window: int = 60
    graph_min_observations: int = 55
    graph_neighbors: int = 10
    feature_winsor_lower: float = 0.01
    feature_winsor_upper: float = 0.99
    low_frequency_bins: tuple[int, ...] = (0, 1, 2)
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    purge_dates: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreparationConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        values["low_frequency_bins"] = tuple(values.get("low_frequency_bins", (0, 1, 2)))
        values["split_ratios"] = tuple(values.get("split_ratios", (0.6, 0.2, 0.2)))
        return cls(**values)

    def validate(self) -> None:
        if self.lookback < 20:
            raise ValueError("lookback must be at least 20 for feature computation")
        if self.graph_min_observations > self.graph_window:
            raise ValueError("graph_min_observations cannot exceed graph_window")
        if len(self.split_ratios) != 3 or abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must contain three values summing to one")
        if not 0 <= self.feature_winsor_lower < self.feature_winsor_upper <= 1:
            raise ValueError("invalid winsorization quantiles")


def _canonicalize_prices(config: PreparationConfig) -> pd.DataFrame:
    raw = pd.read_csv(config.prices_csv)
    required = ("date", "symbol", "open", "high", "low", "close", "volume", "turnover")
    missing_mapping = [name for name in required if name not in config.columns]
    if missing_mapping:
        raise ValueError(f"missing column mappings: {missing_mapping}")
    rename = {
        source: canonical for canonical, source in config.columns.items() if source in raw.columns
    }
    prices = raw.rename(columns=rename)
    missing = [name for name in required if name not in prices.columns]
    if missing:
        raise ValueError(f"prices CSV is missing columns: {missing}")
    prices["date"] = pd.to_datetime(prices["date"], utc=False).dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str)
    if "tradable" not in prices.columns:
        prices["tradable"] = True
    prices["tradable"] = prices["tradable"].fillna(False).astype(bool)
    numeric = ["open", "high", "low", "close", "volume", "turnover"]
    prices[numeric] = prices[numeric].apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"])
    return prices.set_index(["date", "symbol"]).sort_index()


def _canonicalize_membership(
    config: PreparationConfig,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    raw = pd.read_csv(config.membership_csv)
    rename = {
        source: canonical for canonical, source in config.columns.items() if source in raw.columns
    }
    membership = raw.rename(columns=rename)
    if "date" not in membership or "symbol" not in membership:
        raise ValueError("membership CSV must contain mapped date and symbol columns")
    membership["date"] = pd.to_datetime(membership["date"], utc=False).dt.normalize()
    membership["symbol"] = membership["symbol"].astype(str)
    if "membership_active" in membership:
        active = membership["membership_active"].fillna(False).astype(bool)
        membership = membership[active]
    return {
        date: tuple(sorted(group["symbol"].unique().tolist()))
        for date, group in membership.groupby("date", sort=True)
    }


def _compute_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_index(level="date").copy()
    close = group["close"]
    log_close = np.log(close.where(close > 0))
    daily_return = log_close.diff()
    log_volume = np.log(group["volume"].where(group["volume"] > 0))
    features = pd.DataFrame(index=group.index)
    features["return_1d"] = log_close.diff(1)
    features["return_5d"] = log_close.diff(5)
    features["return_10d"] = log_close.diff(10)
    features["open_close_relative"] = group["open"] / close - 1.0
    features["high_close_relative"] = group["high"] / close - 1.0
    features["low_close_relative"] = group["low"] / close - 1.0
    features["volume_change"] = log_volume.diff()
    features["turnover"] = group["turnover"]
    features["realized_volatility_5d"] = daily_return.rolling(5).std(ddof=0)
    features["realized_volatility_20d"] = daily_return.rolling(20).std(ddof=0)
    features["momentum_5d"] = close / close.shift(5) - 1.0
    features["ma5_ma20_ratio"] = close.rolling(5).mean() / close.rolling(20).mean()
    return features


def _cross_sectional_transform(
    values: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    low = values.quantile(lower)
    high = values.quantile(upper)
    clipped = values.clip(low, high, axis=1)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0, ddof=0).replace(0.0, np.nan)
    # Constant observed features map to zero; missing observations remain missing.
    return (clipped - mean) / std.fillna(1.0)


def _standardize_array(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return ((values - values.mean()) / max(values.std(ddof=0), epsilon)).astype(np.float32)


class PointInTimeBuilder:
    """Create date items from local adjusted OHLCV panels."""

    def __init__(self, config: PreparationConfig) -> None:
        config.validate()
        self.config = config
        self.prices = _canonicalize_prices(config)
        self.membership = _canonicalize_membership(config)
        self.calendar = tuple(sorted(self.prices.index.get_level_values("date").unique()))
        self.date_position = {date: index for index, date in enumerate(self.calendar)}
        self.features = self._build_feature_panel()
        self.daily_returns = self._build_return_panel()
        self._graph_cache: dict[pd.Timestamp, tuple[tuple[str, ...], np.ndarray]] = {}

    def _build_feature_panel(self) -> pd.DataFrame:
        pieces = []
        for _, group in self.prices.groupby(level="symbol", sort=False):
            symbol = group.index.get_level_values("symbol")[0]
            keys = pd.MultiIndex.from_product([self.calendar, [symbol]], names=["date", "symbol"])
            pieces.append(_compute_symbol_features(group.reindex(keys)))
        return pd.concat(pieces).sort_index()

    def _build_return_panel(self) -> pd.DataFrame:
        close = self.prices["close"].unstack("symbol").reindex(self.calendar)
        return np.log(close.where(close > 0)).diff()

    def _members(self, date: pd.Timestamp) -> tuple[str, ...]:
        return self.membership.get(date, ())

    def _graph_for_date(self, date: pd.Timestamp) -> tuple[tuple[str, ...], np.ndarray]:
        if date in self._graph_cache:
            return self._graph_cache[date]
        members = self._members(date)
        position = self.date_position.get(date, -1)
        start = position - self.config.graph_window + 1
        if not members or start < 0:
            result = ((), np.empty((0, 0), dtype=np.float32))
        else:
            history_dates = self.calendar[start : position + 1]
            available = [symbol for symbol in members if symbol in self.daily_returns.columns]
            history = self.daily_returns.loc[list(history_dates), available]
            available = [
                symbol
                for symbol in available
                if int(history[symbol].notna().sum()) >= self.config.graph_min_observations
            ]
            matrix = history.loc[:, available].to_numpy(dtype=np.float64)
            adjacency = positive_knn_adjacency(
                matrix,
                neighbors=self.config.graph_neighbors,
                min_observations=self.config.graph_min_observations,
            )
            result = (tuple(available), adjacency)
        self._graph_cache[date] = result
        return result

    def _market_statistics(self, date: pd.Timestamp) -> np.ndarray | None:
        position = self.date_position[date]
        if position < 19:
            return None
        state_dates = self.calendar[position - 19 : position + 1]
        market_returns = []
        dispersions = []
        graph_variations = []
        for state_date in state_dates:
            members, adjacency = self._graph_for_date(state_date)
            if not members:
                return None
            returns = self.daily_returns.loc[state_date, list(members)].to_numpy(dtype=np.float64)
            finite = np.isfinite(returns)
            if finite.sum() < 2:
                return None
            valid_returns = returns[finite]
            # Trend/dispersion use the date's observed PIT constituents, not
            # graph-window eligibility.
            observed = (
                self.daily_returns.loc[state_date]
                .reindex(self._members(state_date))
                .to_numpy(dtype=np.float64)
            )
            observed = observed[np.isfinite(observed)]
            if len(observed) < 2:
                return None
            market_returns.append(float(observed.mean()))
            dispersions.append(float(observed.std(ddof=0)))

            sub_adjacency = adjacency[np.ix_(finite, finite)]
            degree = sub_adjacency.sum(axis=1)
            inverse_sqrt = np.zeros_like(degree)
            positive = degree > 0
            inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
            normalized = inverse_sqrt[:, None] * sub_adjacency * inverse_sqrt[None, :]
            laplacian = np.eye(len(valid_returns)) - normalized
            numerator = float(valid_returns @ laplacian @ valid_returns)
            denominator = float(valid_returns @ valid_returns) + 1e-12
            graph_variations.append(numerator / denominator)

        market = np.asarray(market_returns)
        spectrum_energy = np.abs(np.fft.rfft(market)) ** 2
        selected = [
            index for index in self.config.low_frequency_bins if index < len(spectrum_energy)
        ]
        if not selected:
            raise ValueError("low_frequency_bins has no valid bins for the state window")
        low_frequency_ratio = spectrum_energy[selected].sum() / max(spectrum_energy.sum(), 1e-12)
        return np.asarray(
            [
                market.sum(),
                market.std(ddof=0),
                np.mean(dispersions),
                np.mean(graph_variations),
                low_frequency_ratio,
            ],
            dtype=np.float32,
        )

    def _build_item(self, date: pd.Timestamp) -> dict[str, Any] | None:
        position = self.date_position[date]
        if position < self.config.lookback - 1:
            return None
        graph_members, _ = self._graph_for_date(date)
        if not graph_members:
            return None
        history_dates = self.calendar[position - self.config.lookback + 1 : position + 1]
        label_available = position + 6 < len(self.calendar)

        valid_symbols = []
        histories = []
        targets = []
        for graph_index, symbol in enumerate(graph_members):
            keys = pd.MultiIndex.from_product([history_dates, [symbol]], names=["date", "symbol"])
            history = self.features.reindex(keys).loc[:, FEATURE_NAMES]
            current_index = pd.MultiIndex.from_tuples([(date, symbol)])
            current_quote = self.prices.reindex(current_index).iloc[0]
            correct_length = history.shape[0] == self.config.lookback
            finite_history = np.isfinite(history.to_numpy()).all()
            if not correct_length or not finite_history:
                continue
            if not bool(current_quote.get("tradable", False)):
                continue
            valid_symbols.append((graph_index, symbol))
            histories.append(history.to_numpy(dtype=np.float32))
            # Labels cannot determine candidates, normalization, topology or state.
            target = np.nan
            if label_available:
                target_index = pd.MultiIndex.from_tuples(
                    [
                        (self.calendar[position + 1], symbol),
                        (self.calendar[position + 6], symbol),
                    ]
                )
                target_quotes = self.prices.reindex(target_index)
                opens = target_quotes["open"].to_numpy(dtype=np.float64)
                tradable = target_quotes["tradable"].fillna(False).astype(bool).all()
                if np.isfinite(opens).all() and (opens > 0).all() and tradable:
                    target = np.log(opens[1] / opens[0])
            targets.append(target)

        if len(valid_symbols) < 2:
            return None
        symbols = [symbol for _, symbol in valid_symbols]
        # Select top-k within the final, past-only eligible vertex set.
        graph_dates = self.calendar[position - self.config.graph_window + 1 : position + 1]
        adjacency = positive_knn_adjacency(
            self.daily_returns.loc[list(graph_dates), symbols].to_numpy(dtype=np.float64),
            neighbors=self.config.graph_neighbors,
            min_observations=self.config.graph_min_observations,
        )
        edge_index, edge_weight = scaled_laplacian_edges(adjacency)

        feature_array = np.stack(histories)
        for time_index in range(feature_array.shape[1]):
            transformed = _cross_sectional_transform(
                pd.DataFrame(feature_array[:, time_index, :], columns=FEATURE_NAMES),
                self.config.feature_winsor_lower,
                self.config.feature_winsor_upper,
            )
            feature_array[:, time_index, :] = transformed.to_numpy(dtype=np.float32)

        raw_targets = np.asarray(targets, dtype=np.float64)
        labeled = np.isfinite(raw_targets)
        standardized_targets = np.full(raw_targets.shape, np.nan, dtype=np.float32)
        if labeled.any():
            relative_targets = raw_targets[labeled] - raw_targets[labeled].mean()
            standardized_targets[labeled] = _standardize_array(relative_targets)
        state = self._market_statistics(date)
        if state is None or not np.isfinite(state).all():
            return None
        return {
            "date": date.strftime("%Y-%m-%d"),
            "symbols": np.asarray(symbols),
            "features": feature_array.astype(np.float32),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "market_state": state,
            "targets": standardized_targets,
        }

    def build(self) -> dict[str, Any]:
        items = []
        for date in sorted(self.membership):
            if date not in self.date_position:
                continue
            # Offline supervised horizon; _build_item also supports live dates.
            if self.date_position[date] + 6 >= len(self.calendar):
                continue
            item = self._build_item(date)
            if item is not None:
                items.append(item)
        if len(items) < 3:
            raise RuntimeError("fewer than three valid prediction dates were produced")

        count = len(items)
        train_end = int(count * self.config.split_ratios[0])
        validation_end = train_end + int(count * self.config.split_ratios[1])
        purge = self.config.purge_dates
        split_items = {
            "train": items[: max(0, train_end - purge)],
            "validation": items[train_end : max(train_end, validation_end - purge)],
            "test": items[validation_end:],
        }
        if any(not values for values in split_items.values()):
            raise RuntimeError("one or more splits are empty after boundary purging")

        training_states = np.stack([item["market_state"] for item in split_items["train"]])
        state_mean = training_states.mean(axis=0)
        state_std = training_states.std(axis=0)
        state_std[state_std < 1e-8] = 1.0

        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest_splits: dict[str, list[str]] = {}
        for split, values in split_items.items():
            split_dir = output / split
            split_dir.mkdir(parents=True, exist_ok=True)
            manifest_splits[split] = []
            for item in values:
                item["market_state"] = ((item["market_state"] - state_mean) / state_std).astype(
                    np.float32
                )
                relative_path = f"{split}/{item['date']}.npz"
                np.savez_compressed(output / relative_path, **item)
                manifest_splits[split].append(relative_path)

        manifest = {
            "splits": manifest_splits,
            "metadata": {
                "implementation_version": 2,
                "feature_names": list(FEATURE_NAMES),
                "feature_dim": len(FEATURE_NAMES),
                "lookback": self.config.lookback,
                "state_names": [
                    "trend",
                    "volatility",
                    "cross_sectional_dispersion",
                    "graph_total_variation",
                    "low_frequency_energy_ratio",
                ],
                "state_mean": state_mean.tolist(),
                "state_std": state_std.tolist(),
                "low_frequency_bins": list(self.config.low_frequency_bins),
                "candidate_policy": "past-only eligibility; unavailable targets stay NaN",
                "label_policy": "fixed t+1/t+6 opens; finite labels only in loss and metrics",
                "purge_policy": "remove the final purge_dates from train and validation",
            },
        }
        write_json(output / "manifest.json", manifest)
        return manifest


def prepare_dataset(config_path: str | Path) -> dict[str, Any]:
    config = PreparationConfig.from_yaml(config_path)
    return PointInTimeBuilder(config).build()
