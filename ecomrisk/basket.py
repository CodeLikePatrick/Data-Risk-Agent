"""Market-basket analysis: Apriori and association rules (numpy/pandas only)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _support(itemset, transactions) -> float:
    if not itemset or not transactions:
        return 0.0
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)


def apriori(transactions, min_support: float = 0.05, max_size: int = 3) -> list[tuple[set, float]]:
    """Return frequent itemsets as ``(set(items), support)`` pairs."""
    transactions = [set(str(item) for item in t) for t in transactions]
    n = len(transactions)
    if n == 0:
        return []
    items = sorted({item for t in transactions for item in t})
    result: list[tuple[set, float]] = []
    candidates = [frozenset([i]) for i in items]
    k = 1
    while candidates and k <= max_size:
        freq = []
        for cand in candidates:
            sup = _support(cand, transactions)
            if sup >= min_support:
                freq.append(cand)
                result.append((set(cand), round(sup, 4)))
        k += 1
        freq_sets = [set(c) for c in freq]
        next_candidates = set()
        for a, b in itertools.combinations(freq_sets, 2):
            union = a | b
            if len(union) == k:
                next_candidates.add(frozenset(union))
        candidates = sorted(next_candidates, key=lambda x: (len(x), tuple(sorted(x))))
    return result


def association_rules(frequent_itemsets, transactions, min_confidence: float = 0.5) -> list[dict]:
    """Generate rules with support, confidence and lift."""
    support_map = {frozenset(items): sup for items, sup in frequent_itemsets}
    rules = []
    for items, sup in frequent_itemsets:
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ant in itertools.combinations(items, r):
                ant = frozenset(ant)
                con = frozenset(items - set(ant))
                ant_sup = support_map.get(ant, _support(ant, transactions))
                if not ant_sup:
                    continue
                confidence = sup / ant_sup
                if confidence < min_confidence:
                    continue
                con_sup = support_map.get(con, _support(con, transactions))
                lift = confidence / con_sup if con_sup else 0.0
                rules.append({
                    "antecedent": sorted(ant),
                    "consequent": sorted(con),
                    "support": round(sup, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                })
    return sorted(rules, key=lambda r: (-r["lift"], -r["confidence"]))


def sample_baskets(products: pd.DataFrame, seed: int = 7, n_orders: int = 400) -> list[set[str]]:
    """Synthesize order baskets from product categories with one injected pattern."""
    rng = np.random.default_rng(seed)
    cats = [str(c) for c in products["category"].dropna().unique()]
    if not cats:
        return []
    pair = (cats[0], cats[1]) if len(cats) > 1 else (cats[0], cats[0])
    baskets = []
    for _ in range(n_orders):
        basket = set()
        if rng.random() < 0.35:
            basket.update(pair)
        for _ in range(int(rng.integers(1, 4))):
            basket.add(cats[int(rng.integers(0, len(cats)))])
        baskets.append(basket)
    return baskets
