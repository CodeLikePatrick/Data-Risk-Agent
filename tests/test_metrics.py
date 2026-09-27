"""Tests for the from-scratch evaluation metrics."""

from ecomrisk.metrics import ks_statistic, lift_table, regression_metrics, roc_auc


def test_roc_auc_is_perfect_for_separated_scores():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y, s) == 1.0


def test_roc_auc_is_half_for_random_scores():
    y = [0, 0, 1, 1]
    s = [0.5, 0.5, 0.5, 0.5]
    assert abs(roc_auc(y, s) - 0.5) < 1e-9


def test_ks_is_one_for_perfect_separation():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_lift_table_has_expected_columns_and_top_lift():
    y = [0] * 90 + [1] * 10
    s = list(range(100))
    table = lift_table(y, s, n_bins=10)
    assert set(table.columns) == {"decile", "size", "positives", "cum_positives", "cum_gain", "lift"}
    assert table.iloc[0]["lift"] == 10.0


def test_regression_metrics_perfect_prediction():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["r2"] == 1.0
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
