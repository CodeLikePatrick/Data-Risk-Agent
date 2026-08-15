"""Tests for the synthetic data generator."""

import pandas as pd

from ecomrisk.data import generate


def test_generation_is_deterministic():
    m1, p1 = generate(seed=42)
    m2, p2 = generate(seed=42)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(p1, p2)


def test_labels_are_consistent_with_injected_risks():
    merchants, _ = generate(seed=7, n_merchants=200)
    risk = merchants[merchants["true_risk"] == 1]
    normal = merchants[merchants["true_risk"] == 0]
    assert (risk["true_types"] != "").all()
    assert (normal["true_types"] == "").all()


def test_address_fake_flag_matches_violation_type():
    merchants, _ = generate(seed=8, n_merchants=200)
    fake = merchants[merchants["true_types"].str.contains("地址真实性", na=False)]
    assert (fake["address_fake"] == True).all()  # noqa: E712


def test_products_reference_only_known_merchants():
    merchants, products = generate(seed=9, n_merchants=100)
    assert set(products["merchant_id"]).issubset(set(merchants["merchant_id"]))
    assert not products.empty
