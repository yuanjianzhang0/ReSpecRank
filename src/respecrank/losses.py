"""Ranking, regression, and spectral-diversity objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


def cross_sectional_standardize(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if values.numel() < 2:
        return values - values.mean()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(epsilon)


def sample_pairwise_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    num_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute sampled pairwise logistic loss within one date."""
    num_stocks = scores.numel()
    if num_stocks < 2:
        return scores.sum() * 0.0
    first = torch.randint(num_stocks, (num_pairs,), device=scores.device, generator=generator)
    second = torch.randint(num_stocks - 1, (num_pairs,), device=scores.device, generator=generator)
    second = second + (second >= first).to(second.dtype)
    direction = torch.sign(targets[first] - targets[second])
    valid = direction != 0
    if not torch.any(valid):
        return scores.sum() * 0.0
    score_difference = scores[first[valid]] - scores[second[valid]]
    return functional.softplus(-direction[valid] * score_difference).mean()


def huber_regression_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    standardized_scores = cross_sectional_standardize(scores)
    return functional.huber_loss(standardized_scores, targets, delta=delta)


def chebyshev_polynomials(values: torch.Tensor, order: int) -> torch.Tensor:
    polynomials = [torch.ones_like(values)]
    if order == 0:
        return torch.stack(polynomials)
    polynomials.append(values)
    for _ in range(2, order + 1):
        polynomials.append(2.0 * values * polynomials[-1] - polynomials[-2])
    return torch.stack(polynomials)


def joint_response(
    coefficients: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    lambdas: torch.Tensor,
    omegas: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the joint filter and return a complex [lambda, omega] response."""
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [graph_order, temporal_dilations]")
    graph_order = coefficients.shape[0] - 1
    graph_terms = chebyshev_polynomials(lambdas - 1.0, graph_order)
    dilation = torch.as_tensor(temporal_dilations, device=omegas.device, dtype=omegas.dtype)
    temporal_terms = torch.exp(-1j * omegas[:, None] * dilation[None, :])
    coefficients_complex = coefficients.to(temporal_terms.dtype)
    graph_terms_complex = graph_terms.to(temporal_terms.dtype)
    return torch.einsum(
        "pq,pl,wq->lw",
        coefficients_complex,
        graph_terms_complex,
        temporal_terms,
    )


def spectral_diversity_loss(
    coefficient_bases: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    grid_size: int = 32,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute basis-response diversity on a fixed lambda-omega grid."""
    if coefficient_bases.ndim == 4:
        return torch.stack(
            [
                spectral_diversity_loss(
                    layer_bases,
                    temporal_dilations,
                    grid_size,
                    epsilon,
                )
                for layer_bases in coefficient_bases
            ]
        ).mean()
    if coefficient_bases.ndim != 3:
        raise ValueError(
            "coefficient_bases must have shape [bases, graph, time] or "
            "[layers, bases, graph, time]"
        )
    num_bases = coefficient_bases.shape[0]
    if num_bases < 2:
        return coefficient_bases.sum() * 0.0
    lambdas = torch.linspace(0.0, 2.0, grid_size, device=coefficient_bases.device)
    omegas = torch.linspace(0.0, torch.pi, grid_size, device=coefficient_bases.device)
    magnitudes = torch.stack(
        [
            joint_response(surface, temporal_dilations, lambdas, omegas).abs().flatten()
            for surface in coefficient_bases
        ]
    )
    normalized = magnitudes / magnitudes.norm(dim=1, keepdim=True).clamp_min(epsilon)
    correlations = torch.abs(normalized @ normalized.T)
    upper = torch.triu(correlations.square(), diagonal=1).sum()
    return 2.0 * upper / (num_bases * (num_bases - 1))


@dataclass
class LossTerms:
    total: torch.Tensor
    ranking: torch.Tensor
    huber: torch.Tensor
    diversity: torch.Tensor


def respecrank_loss(
    daily_scores: list[torch.Tensor],
    daily_targets: list[torch.Tensor],
    coefficient_bases: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    num_pairs: int = 4096,
    huber_weight: float = 0.1,
    diversity_weight: float = 0.01,
    huber_delta: float = 1.0,
    response_grid_size: int = 32,
) -> LossTerms:
    ranking = torch.stack(
        [
            sample_pairwise_logistic_loss(scores, targets, num_pairs)
            for scores, targets in zip(daily_scores, daily_targets, strict=True)
        ]
    ).mean()
    huber = torch.stack(
        [
            huber_regression_loss(scores, targets, huber_delta)
            for scores, targets in zip(daily_scores, daily_targets, strict=True)
        ]
    ).mean()
    diversity = spectral_diversity_loss(
        coefficient_bases,
        temporal_dilations,
        response_grid_size,
    )
    total = ranking + huber_weight * huber + diversity_weight * diversity
    return LossTerms(total=total, ranking=ranking, huber=huber, diversity=diversity)
