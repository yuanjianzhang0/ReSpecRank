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
    score_difference = scores[first] - scores[second]
    return functional.softplus(-direction * score_difference).mean()


def huber_regression_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.sum() * 0
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


def matrix_response_energy(
    coefficients: torch.Tensor,
    channel_mixing: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    lambdas: torch.Tensor,
    omegas: torch.Tensor,
) -> torch.Tensor:
    """||sum_pq theta_pq T_p exp(-iw d_q) W_pq||_F^2 / d.

    Leading coefficient dimensions (e.g. anchors) are preserved. Contracting
    the real W Gram matrix avoids materializing grid_size^2 channel matrices.
    """
    graph = chebyshev_polynomials(lambdas - 1, coefficients.shape[-2] - 1)
    delays = omegas.new_tensor(temporal_dilations)
    phase = torch.exp(-1j * omegas[:, None] * delays)
    basis = torch.einsum("pl,wq->lwpq", graph.to(phase.dtype), phase).flatten(-2)
    amplitudes = coefficients.flatten(-2)[..., None, None, :] * basis
    maps = channel_mixing.flatten(0, 1).flatten(1)
    gram = (maps @ maps.T) / channel_mixing.shape[-1]
    real, imag = amplitudes.real, amplitudes.imag
    energy = ((real @ gram) * real + (imag @ gram) * imag).sum(-1)
    return energy.clamp_min(0)


def spectral_diversity_loss(
    coefficient_bases: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    grid_size: int = 32,
    epsilon: float = 1e-8,
    *,
    channel_mixing: torch.Tensor,
) -> torch.Tensor:
    """Mean squared pair cosine of W-aware anchor envelopes."""
    if coefficient_bases.ndim == 4:
        return torch.stack(
            [
                spectral_diversity_loss(
                    bases, temporal_dilations, grid_size, epsilon, channel_mixing=maps
                )
                for bases, maps in zip(coefficient_bases, channel_mixing, strict=True)
            ]
        ).mean()
    if coefficient_bases.ndim != 3:
        raise ValueError("expected [anchors, graph, delay] or [layers, anchors, graph, delay]")
    count = coefficient_bases.shape[0]
    if count < 2:
        return coefficient_bases.sum() * 0
    lambdas = torch.linspace(
        0, 2, grid_size, device=coefficient_bases.device, dtype=coefficient_bases.dtype
    )
    omegas = torch.linspace(
        0, torch.pi, grid_size, device=coefficient_bases.device, dtype=coefficient_bases.dtype
    )
    energy = matrix_response_energy(
        coefficient_bases, channel_mixing, temporal_dilations, lambdas, omegas
    )
    magnitudes = torch.where(energy > 0, energy.clamp_min(epsilon**2).sqrt(), 0).flatten(1)
    normalized = functional.normalize(magnitudes, dim=1, eps=epsilon)
    cosine = (normalized @ normalized.T).abs()
    return 2 * cosine.square().triu(diagonal=1).sum() / (count * (count - 1))


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
    channel_mixing: torch.Tensor | None = None,
) -> LossTerms:
    valid = [torch.isfinite(target) for target in daily_targets]
    daily_scores = [score[mask] for score, mask in zip(daily_scores, valid, strict=True)]
    daily_targets = [target[mask] for target, mask in zip(daily_targets, valid, strict=True)]
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
    if diversity_weight:
        if channel_mixing is None:
            raise ValueError("W-aware diversity requires channel_mixing")
        diversity = spectral_diversity_loss(
            coefficient_bases,
            temporal_dilations,
            response_grid_size,
            channel_mixing=channel_mixing,
        )
    else:
        diversity = coefficient_bases.new_zeros(())
    total = ranking + huber_weight * huber + diversity_weight * diversity
    return LossTerms(total=total, ranking=ranking, huber=huber, diversity=diversity)
