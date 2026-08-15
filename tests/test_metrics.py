import numpy as np

from respecrank.metrics import pearson_ic, rank_ic, summarize_daily_metrics


def test_rank_and_linear_ic() -> None:
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([10.0, 20.0, 30.0, 40.0])
    assert rank_ic(scores, targets) == 1.0
    assert pearson_ic(scores, targets) == 1.0


def test_metric_summary() -> None:
    summary = summarize_daily_metrics([0.1, 0.3], [0.2, 0.4])
    assert np.isclose(summary.rank_ic, 0.2)
    assert np.isclose(summary.ic, 0.3)
    assert summary.num_dates == 2

