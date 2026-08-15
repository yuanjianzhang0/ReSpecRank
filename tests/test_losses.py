import torch

from respecrank.losses import (
    joint_response,
    sample_pairwise_logistic_loss,
    spectral_diversity_loss,
)


def test_correct_order_has_lower_pairwise_loss() -> None:
    targets = torch.tensor([-1.0, 0.0, 1.0])
    correct = sample_pairwise_logistic_loss(targets * 5.0, targets, num_pairs=10000)
    reversed_loss = sample_pairwise_logistic_loss(-targets * 5.0, targets, num_pairs=10000)
    assert correct < reversed_loss


def test_joint_response_and_diversity_are_finite() -> None:
    coefficients = torch.randn(4, 4, 6, requires_grad=True)
    lambdas = torch.linspace(0.0, 2.0, 8)
    omegas = torch.linspace(0.0, torch.pi, 9)
    response = joint_response(coefficients[0], (0, 1, 2, 4, 8, 16), lambdas, omegas)
    assert response.shape == (8, 9)
    diversity = spectral_diversity_loss(coefficients, (0, 1, 2, 4, 8, 16), grid_size=8)
    assert torch.isfinite(diversity)
    diversity.backward()
    assert coefficients.grad is not None


def test_multilayer_diversity_is_finite() -> None:
    coefficients = torch.randn(2, 4, 4, 6, requires_grad=True)
    diversity = spectral_diversity_loss(
        coefficients,
        (0, 1, 2, 4, 8, 16),
        grid_size=8,
    )
    assert torch.isfinite(diversity)
    diversity.backward()
    assert coefficients.grad is not None
