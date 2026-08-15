import torch

from respecrank.model import ReSpecRank
from respecrank.types import DateBatch
from respecrank.utils import count_parameters


def make_batch() -> DateBatch:
    return DateBatch(
        date="2025-01-02",
        symbols=("A", "B", "C"),
        features=torch.randn(3, 20, 12),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_weight=torch.full((4,), -0.5),
        market_state=torch.randn(5),
        targets=torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_router_starts_uniform_and_model_backpropagates() -> None:
    model = ReSpecRank()
    batch = make_batch()
    output = model(batch)
    torch.testing.assert_close(output.router_weights, torch.full((4,), 0.25))
    assert output.scores.shape == (3,)
    output.scores.sum().backward()
    assert model.coefficient_bases.grad is not None


def test_default_model_has_target_parameter_count() -> None:
    model = ReSpecRank()
    assert count_parameters(model) == 805_605


def test_uniform_routing_mode_ignores_state() -> None:
    model = ReSpecRank(routing_mode="uniform")
    batch = make_batch()
    first = model(batch).coefficients
    batch.market_state = torch.randn(5) * 100
    second = model(batch).coefficients
    torch.testing.assert_close(first, second)
