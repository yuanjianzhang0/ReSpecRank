import numpy as np
import torch

from respecrank.graph import (
    chebyshev_signals,
    positive_knn_adjacency,
    scaled_laplacian_edges,
)


def test_positive_knn_graph_is_symmetric_and_nonnegative() -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(size=60)
    returns = np.column_stack([common, common + 0.01 * rng.normal(size=60), -common])
    adjacency = positive_knn_adjacency(returns, neighbors=1, min_observations=55)
    np.testing.assert_allclose(adjacency, adjacency.T)
    assert np.all(adjacency >= 0)
    assert adjacency[0, 1] > 0
    assert adjacency[0, 2] == 0


def test_sparse_chebyshev_recurrence() -> None:
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    edge_index, edge_weight = scaled_laplacian_edges(adjacency)
    values = torch.tensor([[1.0], [2.0]])
    basis = chebyshev_signals(
        values,
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_weight),
        order=2,
    )
    torch.testing.assert_close(basis[0], values)
    torch.testing.assert_close(basis[1], torch.tensor([[-2.0], [-1.0]]))
    torch.testing.assert_close(basis[2], values)

