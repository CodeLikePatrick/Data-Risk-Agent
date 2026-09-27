"""Tests for Apriori and association rules."""

from ecomrisk.basket import apriori, association_rules


def test_apriori_finds_expected_support():
    tx = [{"a", "b"}, {"a", "b"}, {"a"}, {"b", "c"}]
    freq = apriori(tx, min_support=0.5, max_size=2)
    by_items = {frozenset(items): sup for items, sup in freq}
    assert by_items[frozenset({"a"})] == 0.75
    assert by_items[frozenset({"b"})] == 0.75
    assert by_items[frozenset({"a", "b"})] == 0.5
    assert frozenset({"c"}) not in by_items


def test_association_rules_compute_confidence_and_lift():
    tx = [{"a", "b"}, {"a", "b"}, {"a"}, {"b", "c"}]
    freq = apriori(tx, min_support=0.5, max_size=2)
    rules = association_rules(freq, tx, min_confidence=0.5)
    rule = next(r for r in rules if r["antecedent"] == ["a"] and r["consequent"] == ["b"])
    assert rule["support"] == 0.5
    assert round(rule["confidence"], 4) == 0.6667
    assert round(rule["lift"], 4) == 0.8889
