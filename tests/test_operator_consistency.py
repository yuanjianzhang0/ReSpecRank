"""Operator checks and counterfactual tests for look-ahead leakage."""

import numpy as np
import pandas as pd
import pytest
import torch
from test_model import make_batch

from respecrank.diagnostics import response_diagnostics
from respecrank.losses import matrix_response_energy, respecrank_loss, spectral_diversity_loss
from respecrank.model import ReSpecRank
from respecrank.preprocessing import FEATURE_NAMES, PointInTimeBuilder, PreparationConfig


def test_shared_bank_matches_dense_operator_and_ignores_unused_history():
    torch.manual_seed(19)
    model = ReSpecRank(hidden_dim=4, dropout=0).double().eval()
    batch = make_batch()
    batch.features = batch.features.double()
    batch.market_state = batch.market_state.double()
    batch.edge_weight = batch.edge_weight.double()
    result = model(batch)
    n = batch.num_stocks
    shift = torch.zeros(n, n, dtype=torch.double)
    shift[batch.edge_index[1], batch.edge_index[0]] = batch.edge_weight
    polynomials = [torch.eye(n, dtype=torch.double), shift]
    for _ in range(2, 4):
        polynomials.append(2 * shift @ polynomials[-1] - polynomials[-2])
    bank = model.feature_projection(batch.features)
    expected = bank[:, -1]
    for layer in range(2):
        branch = sum(
            result.coefficients[layer, p, q]
            * (polynomials[p] @ bank[:, -1 - delay] @ model.channel_mixing[layer, p, q])
            for p in range(4)
            for q, delay in enumerate(model.temporal_dilations)
        )
        expected = model.layer_norms[layer](expected + branch)
    torch.testing.assert_close(result.scores, model.ranking_head(expected).squeeze(-1))
    # Stacking would read delay 3=1+2 or delay 17=16+1.
    batch.features[:, -4] += 100
    batch.features[:, -18] -= 100
    torch.testing.assert_close(result.scores, model(batch).scores)


def test_w_aware_response_matches_explicit_complex_matrices_and_gradients():
    torch.manual_seed(3)
    theta = torch.randn(4, 2, 2, dtype=torch.double, requires_grad=True)
    maps = torch.randn(2, 2, 3, 3, dtype=torch.double, requires_grad=True)
    lambdas = torch.linspace(0, 2, 5, dtype=torch.double)
    omegas = torch.linspace(0, torch.pi, 6, dtype=torch.double)
    explicit = torch.stack(
        [
            torch.stack(
                [
                    torch.stack(
                        [
                            sum(
                                theta[k, p, q]
                                * (1 if p == 0 else lam - 1)
                                * torch.exp(-1j * omega * delay)
                                * maps[p, q]
                                for p in range(2)
                                for q, delay in enumerate((0, 2))
                            )
                            for omega in omegas
                        ]
                    )
                    for lam in lambdas
                ]
            )
            for k in range(4)
        ]
    )
    expected = explicit.abs().square().sum((-1, -2)) / 3
    actual = matrix_response_energy(theta, maps, (0, 2), lambdas, omegas)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.sum(), (theta, maps), retain_graph=True)
    reference = torch.autograd.grad(expected.sum(), (theta, maps))
    for got, want in zip(gradients, reference, strict=True):
        torch.testing.assert_close(got, want)
    diversity = spectral_diversity_loss(theta, (0, 2), 5, channel_mixing=maps)
    diversity.backward()
    assert torch.isfinite(maps.grad).all() and maps.grad.abs().sum() > 0


def test_response_retains_channel_cancellation():
    theta = torch.ones(1, 2)
    maps = torch.stack([torch.eye(2), -torch.eye(2)]).unsqueeze(0)
    diag = response_diagnostics(theta, (0, 0), channel_mixing=maps)
    torch.testing.assert_close(diag.response_magnitude, torch.zeros(8, 12))


@pytest.mark.parametrize("mode", ["uniform", "feature", "direct"])
def test_conditioning_controls_and_warmup(mode):
    model = ReSpecRank(hidden_dim=8, dropout=0, routing_mode=mode).eval()
    batch = make_batch()
    first = model(batch, force_uniform_router=True)
    batch.market_state += 30
    second = model(batch, force_uniform_router=True)
    torch.testing.assert_close(first.scores, second.scores)
    if mode == "direct":
        first.scores.sum().backward()
        assert model.direct_routers[0][-1].bias.grad is not None
        assert model.direct_routers[0][0].weight.grad is None
        with torch.no_grad():
            model.direct_routers[0][-1].weight.normal_()
        assert not torch.allclose(model(batch).coefficients, second.coefficients)
    if mode == "feature":
        active = model(batch)
        torch.testing.assert_close(active.coefficients, second.coefficients)
        assert not torch.allclose(active.scores, second.scores)


def make_builder(tmp_path):
    rng = np.random.default_rng(22)
    dates = pd.bdate_range("2020-01-01", periods=115)
    records = []
    for symbol in ("A", "B", "C", "D"):
        close = 50 * np.exp(rng.normal(0, 0.01, len(dates)).cumsum())
        for date, value in zip(dates, close, strict=True):
            records.append(
                dict(
                    date=date,
                    symbol=symbol,
                    open=value * 0.99,
                    high=value * 1.01,
                    low=value * 0.98,
                    close=value,
                    volume=rng.uniform(100, 200),
                    turnover=rng.uniform(0.1, 0.5),
                    tradable=True,
                )
            )
    prices = pd.DataFrame(records)
    prices.to_csv(tmp_path / "prices.csv", index=False)
    prices[["date", "symbol"]].to_csv(tmp_path / "members.csv", index=False)
    config = PreparationConfig(
        str(tmp_path / "prices.csv"),
        str(tmp_path / "members.csv"),
        str(tmp_path / "processed"),
        {key: key for key in prices.columns},
    )
    return PointInTimeBuilder(config)


def test_future_quotes_and_tradability_cannot_change_model_inputs(tmp_path):
    builder = make_builder(tmp_path)
    date = builder.calendar[90]
    before = builder._build_item(date)
    builder.prices.loc[(builder.calendar[91], "A"), "open"] = np.nan
    builder.prices.loc[(builder.calendar[96], "B"), "tradable"] = False
    after = builder._build_item(date)
    for key in ("symbols", "features", "edge_index", "edge_weight", "market_state"):
        np.testing.assert_array_equal(before[key], after[key])
    assert np.isnan(after["targets"][:2]).all()
    assert np.isfinite(after["targets"][2:]).all()
    # Inference does not require six future dates to exist.
    assert builder._build_item(builder.calendar[-1]) is not None


def test_missing_calendar_date_is_not_a_shortened_rolling_window(tmp_path):
    builder = make_builder(tmp_path)
    missing_date = builder.calendar[85]
    builder.prices = builder.prices.drop((missing_date, "A"))
    panel = builder._build_feature_panel()
    assert (
        panel.loc[(builder.calendar[86], "A"), "return_1d"]
        != panel.loc[(builder.calendar[86], "A"), "return_1d"]
    )  # NaN
    assert panel.loc[(missing_date, "A"), list(FEATURE_NAMES)].isna().all()


def test_missing_targets_only_mask_supervision():
    scores = torch.tensor([1.0, 10.0, -1.0], requires_grad=True)
    terms = respecrank_loss(
        [scores],
        [torch.tensor([1.0, float("nan"), -1.0])],
        torch.zeros(2, 4, 4, 6),
        (0, 1, 2, 4, 8, 16),
        diversity_weight=0,
    )
    terms.total.backward()
    assert torch.isfinite(terms.total)
    assert scores.grad[1] == 0


def test_zero_operator_has_zero_diversity_and_finite_gradients():
    anchors = torch.zeros(4, 2, 2, requires_grad=True)
    maps = torch.randn(2, 2, 3, 3, requires_grad=True)
    loss = spectral_diversity_loss(anchors, (0, 1), channel_mixing=maps)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(anchors.grad).all()
    assert torch.isfinite(maps.grad).all()


def test_all_control_configs_use_expected_switches():
    from pathlib import Path

    from respecrank.config import load_config

    root = Path(__file__).resolve().parents[1]
    for name in ("fixedjoint", "featurecond", "directcond", "without_diversity"):
        config = load_config(root / "configs" / (name + ".yaml"))
        assert config.loss.diversity_weight == 0
        assert config.training.router_warmup_epochs == 5
        assert config.training.dates_per_batch == 8
