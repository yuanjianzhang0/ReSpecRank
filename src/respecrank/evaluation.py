"""Evaluation and state-response control routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from respecrank.diagnostics import response_diagnostics
from respecrank.metrics import MetricSummary, pearson_ic, rank_ic, summarize_daily_metrics
from respecrank.model import ReSpecRank
from respecrank.types import DateBatch


@dataclass
class DateEvaluation:
    date: str
    symbols: tuple[str, ...]
    scores: np.ndarray
    targets: np.ndarray
    rank_ic: float
    ic: float
    router_weights: np.ndarray
    market_state: np.ndarray
    temporal_centroid: float
    graph_centroid: float


@dataclass
class EvaluationResult:
    summary: MetricSummary
    dates: list[DateEvaluation]


@torch.no_grad()
def evaluate_model(
    model: ReSpecRank,
    loader: DataLoader,
    device: torch.device,
    shuffled_states: list[torch.Tensor] | None = None,
) -> EvaluationResult:
    model.eval()
    results = []
    state_index = 0
    for date_group in loader:
        for item in date_group:
            batch: DateBatch = item.to(device)
            if shuffled_states is not None:
                batch.market_state = shuffled_states[state_index].to(device)
            state_index += 1
            output = model(batch)
            scores = output.scores.detach().cpu().numpy()
            if batch.targets is None:
                raise ValueError("evaluation requires targets")
            targets = batch.targets.detach().cpu().numpy()
            diagnostics = response_diagnostics(output.coefficients, model.temporal_dilations)
            results.append(
                DateEvaluation(
                    date=batch.date,
                    symbols=batch.symbols,
                    scores=scores,
                    targets=targets,
                    rank_ic=rank_ic(scores, targets),
                    ic=pearson_ic(scores, targets),
                    router_weights=output.router_weights.detach().cpu().numpy(),
                    market_state=batch.market_state.detach().cpu().numpy(),
                    temporal_centroid=diagnostics.temporal_centroid,
                    graph_centroid=diagnostics.graph_centroid,
                )
            )
    summary = summarize_daily_metrics(
        [result.rank_ic for result in results],
        [result.ic for result in results],
    )
    return EvaluationResult(summary=summary, dates=results)


def collect_shuffled_states(
    loader: DataLoader,
    seed: int,
) -> list[torch.Tensor]:
    states = [item.market_state.clone() for group in loader for item in group]
    permutation = np.random.default_rng(seed).permutation(len(states))
    return [states[index] for index in permutation]
