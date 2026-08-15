"""Two-layer ReSpecRank model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from respecrank.graph import chebyshev_signals
from respecrank.types import DateBatch


@dataclass
class ReSpecRankOutput:
    scores: torch.Tensor
    router_weights: torch.Tensor
    coefficients: torch.Tensor
    filtered_features: torch.Tensor


class RegimeRouter(nn.Module):
    """Map five trailing market statistics to convex basis weights."""

    def __init__(self, state_dim: int = 5, hidden_dim: int = 16, num_bases: int = 4) -> None:
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_bases)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input_layer(state))
        return torch.softmax(self.output_layer(hidden), dim=-1)


class ReSpecRank(nn.Module):
    """Regime-adaptive joint filter with branch-specific channel mixing."""

    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dim: int = 128,
        joint_layers: int = 2,
        graph_order: int = 3,
        temporal_dilations: tuple[int, ...] = (0, 1, 2, 4, 8, 16),
        num_bases: int = 4,
        router_hidden_dim: int = 16,
        dropout: float = 0.1,
        coefficient_init_scale: float = 0.1,
        routing_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if graph_order < 0:
            raise ValueError("graph_order must be non-negative")
        if joint_layers < 1:
            raise ValueError("joint_layers must be positive")
        if not temporal_dilations or min(temporal_dilations) < 0:
            raise ValueError("temporal_dilations must be non-empty and non-negative")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.joint_layers = joint_layers
        self.graph_order = graph_order
        self.temporal_dilations = tuple(temporal_dilations)
        self.num_bases = num_bases
        if routing_mode not in {"adaptive", "uniform"}:
            raise ValueError("routing_mode must be 'adaptive' or 'uniform'")
        self.routing_mode = routing_mode

        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.router = RegimeRouter(5, router_hidden_dim, num_bases)
        self.coefficient_bases = nn.Parameter(
            torch.empty(
                joint_layers,
                num_bases,
                graph_order + 1,
                len(temporal_dilations),
            )
        )
        self.channel_mixing = nn.Parameter(
            torch.empty(
                joint_layers,
                graph_order + 1,
                len(temporal_dilations),
                hidden_dim,
                hidden_dim,
            )
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(joint_layers)]
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters(coefficient_init_scale)

    def reset_parameters(self, coefficient_init_scale: float = 0.1) -> None:
        for layer_bases in self.coefficient_bases.data:
            flattened = layer_bases.view(self.num_bases, -1)
            nn.init.orthogonal_(flattened)
            flattened.mul_(coefficient_init_scale)
        for layer_mixing in self.channel_mixing.data:
            for graph_mixing in layer_mixing:
                for branch_mixing in graph_mixing:
                    nn.init.xavier_uniform_(branch_mixing)

    def composite_coefficients(
        self,
        market_state: torch.Tensor,
        force_uniform_router: bool = False,
        router_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if router_weights is None:
            if force_uniform_router or self.routing_mode == "uniform":
                router_weights = torch.full(
                    (self.num_bases,),
                    1.0 / self.num_bases,
                    device=market_state.device,
                    dtype=market_state.dtype,
                )
            else:
                router_weights = self.router(market_state)
        coefficients = torch.einsum(
            "k,rkpq->rpq",
            router_weights,
            self.coefficient_bases,
        )
        return coefficients, router_weights

    def forward(
        self,
        batch: DateBatch,
        force_uniform_router: bool = False,
        router_weights: torch.Tensor | None = None,
    ) -> ReSpecRankOutput:
        if batch.features.ndim != 3:
            raise ValueError("features must have shape [stocks, time, features]")
        if batch.features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dimension {self.feature_dim}")
        if batch.features.shape[1] <= max(self.temporal_dilations):
            raise ValueError("input history is shorter than the largest temporal dilation")

        hidden = self.feature_projection(batch.features)
        coefficients, weights = self.composite_coefficients(
            batch.market_state,
            force_uniform_router=force_uniform_router,
            router_weights=router_weights,
        )

        num_stocks, sequence_length, _ = hidden.shape
        filtered = torch.zeros_like(hidden)
        for layer_index in range(self.joint_layers):
            # The current point-in-time graph acts on every temporal position.
            flattened = hidden.reshape(num_stocks, sequence_length * self.hidden_dim)
            graph_basis = chebyshev_signals(
                flattened,
                batch.edge_index,
                batch.edge_weight,
                self.graph_order,
            )
            filtered = torch.zeros_like(hidden)
            for graph_index, flattened_signal in enumerate(graph_basis):
                signal = flattened_signal.reshape(
                    num_stocks,
                    sequence_length,
                    self.hidden_dim,
                )
                for temporal_index, dilation in enumerate(self.temporal_dilations):
                    # Slicing implements a causal shift with a zero left boundary.
                    source = signal[:, : sequence_length - dilation, :]
                    mixed = source @ self.channel_mixing[
                        layer_index,
                        graph_index,
                        temporal_index,
                    ]
                    filtered[:, dilation:, :] = filtered[:, dilation:, :] + (
                        coefficients[layer_index, graph_index, temporal_index] * mixed
                    )
            hidden = self.layer_norms[layer_index](hidden + filtered)

        scores = self.ranking_head(hidden[:, -1, :]).squeeze(-1)
        return ReSpecRankOutput(
            scores=scores,
            router_weights=weights,
            coefficients=coefficients,
            filtered_features=filtered[:, -1, :],
        )

    @classmethod
    def from_config(cls, data_config: object, model_config: object) -> ReSpecRank:
        return cls(
            feature_dim=data_config.feature_dim,
            hidden_dim=model_config.hidden_dim,
            joint_layers=model_config.joint_layers,
            graph_order=model_config.graph_order,
            temporal_dilations=model_config.temporal_dilations,
            num_bases=model_config.num_bases,
            router_hidden_dim=model_config.router_hidden_dim,
            dropout=model_config.dropout,
            coefficient_init_scale=model_config.coefficient_init_scale,
            routing_mode=model_config.routing_mode,
        )
