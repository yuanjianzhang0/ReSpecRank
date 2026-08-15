"""Permutation-invariant joint-response diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from respecrank.losses import joint_response


@dataclass
class ResponseDiagnostics:
    temporal_centroid: float
    graph_centroid: float
    response_magnitude: torch.Tensor


def response_diagnostics(
    coefficients: torch.Tensor,
    temporal_dilations: tuple[int, ...],
    graph_grid_size: int = 32,
    temporal_grid_size: int = 32,
) -> ResponseDiagnostics:
    lambdas = torch.linspace(0.0, 2.0, graph_grid_size, device=coefficients.device)
    omegas = torch.linspace(0.0, torch.pi, temporal_grid_size, device=coefficients.device)
    if coefficients.ndim == 2:
        magnitude = joint_response(
            coefficients,
            temporal_dilations,
            lambdas,
            omegas,
        ).abs()
    elif coefficients.ndim == 3:
        layer_responses = torch.stack(
            [
                joint_response(surface, temporal_dilations, lambdas, omegas).abs()
                for surface in coefficients
            ]
        )
        magnitude = layer_responses.square().mean(dim=0).sqrt()
    else:
        raise ValueError("coefficients must have shape [graph, time] or [layers, graph, time]")
    energy = magnitude.square()
    denominator = energy.sum().clamp_min(1e-12)
    graph_centroid = (energy * lambdas[:, None]).sum() / denominator
    temporal_centroid = (energy * omegas[None, :]).sum() / denominator
    return ResponseDiagnostics(
        temporal_centroid=float(temporal_centroid.detach().cpu()),
        graph_centroid=float(graph_centroid.detach().cpu()),
        response_magnitude=magnitude.detach().cpu(),
    )
