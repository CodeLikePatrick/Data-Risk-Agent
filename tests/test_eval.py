"""Tests for evaluation metrics."""

from ecomrisk.eval import _metrics


def test_metrics_are_computed_correctly():
    m = _metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "false_positive_rate": 0.5}


def test_perfect_classification_scores_one():
    m = _metrics([1, 1, 0, 0], [1, 1, 0, 0])
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "false_positive_rate": 0.0}


def test_no_positives_has_zero_precision_and_recall():
    m = _metrics([1, 0], [0, 0])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
