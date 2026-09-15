"""Permutation-invariant joint-response diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from respecrank.losses import matrix_response_energy


@dataclass
class ResponseDiagnostics:
    temporal_centroid: float
    graph_centroid: float
    response_magnitude: torch.Tensor


def response_diagnostics(
    coefficients: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    graph_grid_size: int = 8,
    temporal_grid_size: int = 12,
    *,
    channel_mixing: torch.Tensor,
) -> ResponseDiagnostics:
    lambdas = torch.linspace(0.0, 2.0, graph_grid_size, device=coefficients.device)
    omegas = torch.linspace(0.0, torch.pi, temporal_grid_size, device=coefficients.device)
    if coefficients.ndim == 2:
        energy = matrix_response_energy(
            coefficients, channel_mixing, temporal_dilations, lambdas, omegas
        )
    elif coefficients.ndim == 3:
        energy = torch.stack(
            [
                matrix_response_energy(surface, maps, temporal_dilations, lambdas, omegas)
                for surface, maps in zip(coefficients, channel_mixing, strict=True)
            ]
        ).mean(0)
    else:
        raise ValueError("expected [graph, time] or [layers, graph, time]")
    magnitude = energy.sqrt()
    denominator = energy.sum().clamp_min(1e-12)
    graph_centroid = (energy * lambdas[:, None]).sum() / denominator
    temporal_centroid = (energy * omegas[None, :]).sum() / denominator
    return ResponseDiagnostics(
        temporal_centroid=float(temporal_centroid.detach().cpu()),
        graph_centroid=float(graph_centroid.detach().cpu()),
        response_magnitude=magnitude.detach().cpu(),
    )
