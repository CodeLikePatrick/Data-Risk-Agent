"""Tests for the model comparison harness."""

from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.models import compare_models


def test_compare_models_reports_baseline_auc():
    merchants, products = generate(seed=0, n_merchants=80)
    report = detect(merchants, products)
    result = compare_models(report, seed=42, k=5)
    assert set(result) >= {"n_samples", "n_features", "k_folds", "baseline_auc", "gradient_boosting_backend"}
    assert 0.0 <= result["baseline_auc"] <= 1.0
    assert result["k_folds"] == 5
    assert result["n_features"] == 6


def test_compare_models_handles_optional_backend_absence():
    merchants, products = generate(seed=1, n_merchants=60)
    report = detect(merchants, products)
    result = compare_models(report, seed=7, k=3)
    # gradient_boosting_auc is None when neither LightGBM nor scikit-learn is present.
    if result["gradient_boosting_backend"] is None:
        assert result["gradient_boosting_auc"] is None
