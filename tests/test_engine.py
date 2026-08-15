"""Tests for features, scoring, and the detection engine."""

import numpy as np

from ecomrisk.data import generate
from ecomrisk.engine import FEATURE_NAMES, LogisticRegression, build_features, detect, rule_score


def test_feature_matrix_has_expected_shape_and_dtypes():
    merchants, products = generate(seed=0, n_merchants=80)
    features = build_features(merchants, products)
    assert set(FEATURE_NAMES).issubset(set(features.columns))
    assert features["addr_fake"].isin([0.0, 1.0]).all()
    assert features["spam_ratio"].between(0.0, 1.0).all()


def test_address_contradiction_marks_faraway_coordinates():
    merchants, products = generate(seed=1, n_merchants=100)
    features = build_features(merchants, products)
    # Fake addresses are generated far from the claimed city.
    assert features.loc[features["addr_fake"] == 1.0, "addr_contradiction"].mean() > 0.9


def test_rule_score_is_bounded_and_sensitive():
    merchants, products = generate(seed=2, n_merchants=100)
    features = build_features(merchants, products)
    scores = rule_score(features)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert scores[features["addr_fake"] == 1.0].mean() > scores[features["addr_fake"] == 0.0].mean()


def test_logistic_regression_separates_simple_data():
    rng = np.random.default_rng(0)
    x0 = rng.normal(-1.0, 0.5, (100, 2))
    x1 = rng.normal(1.0, 0.5, (100, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 100 + [1] * 100)
    model = LogisticRegression(learning_rate=0.5, epochs=500).fit(x, y)
    proba = model.predict_proba(x)
    accuracy = ((proba >= 0.5).astype(int) == y).mean()
    assert accuracy > 0.95
    assert ((proba > 0.0) & (proba < 1.0)).all()


def test_detect_is_deterministic_and_returns_columns():
    merchants, products = generate(seed=3, n_merchants=80)
    report1 = detect(merchants, products, seed=5)
    report2 = detect(merchants, products, seed=5)
    assert report1["risk_score"].equals(report2["risk_score"])
    assert {"risk_score", "predicted_risk", "evidence"}.issubset(set(report1.columns))
