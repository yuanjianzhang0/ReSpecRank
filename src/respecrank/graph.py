"""Point-in-time correlation graph and sparse Chebyshev operators."""

from __future__ import annotations

import numpy as np
import torch


def positive_knn_adjacency(
    returns: np.ndarray,
    neighbors: int = 10,
    min_observations: int = 55,
) -> np.ndarray:
    """Build a symmetrized positive-correlation k-NN adjacency.

    Args:
        returns: Array shaped [time, stocks], potentially containing NaNs.
        neighbors: Maximum outgoing positive neighbors per stock.
        min_observations: Minimum pairwise finite observations for correlation.
    """
    if returns.ndim != 2:
        raise ValueError("returns must have shape [time, stocks]")
    _, num_stocks = returns.shape
    directed = np.zeros((num_stocks, num_stocks), dtype=np.float32)
    for source in range(num_stocks):
        source_values = returns[:, source]
        candidates: list[tuple[float, int]] = []
        for target in range(num_stocks):
            if source == target:
                continue
            target_values = returns[:, target]
            valid = np.isfinite(source_values) & np.isfinite(target_values)
            if int(valid.sum()) < min_observations:
                continue
            x = source_values[valid]
            y = target_values[valid]
            if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            correlation = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(correlation) and correlation > 0.0:
                candidates.append((correlation, target))
        candidates.sort(reverse=True)
        for correlation, target in candidates[:neighbors]:
            directed[source, target] = correlation
    return (directed + directed.T) * 0.5


def scaled_laplacian_edges(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return edges of L_tilde = L_norm - I = -D^-1/2 A D^-1/2."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    rows, columns = np.nonzero(normalized)
    edge_index = np.stack([columns, rows], axis=0).astype(np.int64)
    edge_weight = (-normalized[rows, columns]).astype(np.float32)
    return edge_index, edge_weight


def sparse_shift(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a sparse graph operator stored as source-to-destination edges."""
    if values.ndim != 2:
        raise ValueError("values must have shape [nodes, channels]")
    source, destination = edge_index
    messages = values[source] * edge_weight.unsqueeze(-1)
    output = torch.zeros_like(values)
    output.index_add_(0, destination, messages)
    return output


def chebyshev_signals(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    order: int,
) -> list[torch.Tensor]:
    """Evaluate T_0 through T_order recursively on a graph signal."""
    if order < 0:
        raise ValueError("order must be non-negative")
    basis = [values]
    if order == 0:
        return basis
    basis.append(sparse_shift(values, edge_index, edge_weight))
    for _ in range(2, order + 1):
        basis.append(2.0 * sparse_shift(basis[-1], edge_index, edge_weight) - basis[-2])
    return basis
