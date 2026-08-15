"""Daily cross-sectional metrics and paired time-series inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _valid_pair(scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(scores) & np.isfinite(targets)
    return scores[valid], targets[valid]


def pearson_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    scores, targets = _valid_pair(scores, targets)
    if len(scores) < 2 or np.std(scores) <= 1e-12 or np.std(targets) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(scores, targets)[0, 1])


def rank_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    scores, targets = _valid_pair(scores, targets)
    if len(scores) < 2:
        return float("nan")
    score_ranks = pd.Series(scores).rank(method="average").to_numpy()
    target_ranks = pd.Series(targets).rank(method="average").to_numpy()
    return pearson_ic(score_ranks, target_ranks)


@dataclass
class MetricSummary:
    rank_ic: float
    ic: float
    icir: float
    rank_ic_std: float
    ic_std: float
    num_dates: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rank_ic": self.rank_ic,
            "ic": self.ic,
            "icir": self.icir,
            "rank_ic_std": self.rank_ic_std,
            "ic_std": self.ic_std,
            "num_dates": self.num_dates,
        }


def summarize_daily_metrics(daily_rank_ic: list[float], daily_ic: list[float]) -> MetricSummary:
    rank_values = np.asarray(daily_rank_ic, dtype=np.float64)
    ic_values = np.asarray(daily_ic, dtype=np.float64)
    rank_mean = float(np.nanmean(rank_values))
    ic_mean = float(np.nanmean(ic_values))
    ic_std = float(np.nanstd(ic_values, ddof=0))
    return MetricSummary(
        rank_ic=rank_mean,
        ic=ic_mean,
        icir=ic_mean / max(ic_std, 1e-12),
        rank_ic_std=float(np.nanstd(rank_values, ddof=0)),
        ic_std=ic_std,
        num_dates=int(np.isfinite(rank_values).sum()),
    )


def moving_block_bootstrap_interval(
    paired_differences: np.ndarray,
    block_length: int = 10,
    repetitions: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Paired moving-block interval used for date-aligned model differences."""
    values = np.asarray(paired_differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < block_length:
        raise ValueError("the paired series is shorter than one bootstrap block")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values) - block_length + 1)
    estimates = np.empty(repetitions, dtype=np.float64)
    blocks_needed = int(np.ceil(len(values) / block_length))
    for repetition in range(repetitions):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [values[start : start + block_length] for start in sampled_starts]
        )[: len(values)]
        estimates[repetition] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1.0 - alpha]))


def blockwise_sign_flip_pvalue(
    paired_differences: np.ndarray,
    block_length: int = 10,
    repetitions: int = 1000,
    seed: int = 42,
) -> float:
    values = np.asarray(paired_differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("paired_differences is empty")
    blocks = [values[start : start + block_length] for start in range(0, len(values), block_length)]
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(blocks))
        randomized = np.concatenate(
            [sign * block for sign, block in zip(signs, blocks, strict=True)]
        )
        exceedances += abs(randomized.mean()) >= observed
    return (exceedances + 1.0) / (repetitions + 1.0)
