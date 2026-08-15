"""Feature extraction, two-stage risk scoring, and detection orchestration."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "addr_fake",
    "addr_contradiction",
    "price_zmax",
    "price_low_ratio",
    "spam_ratio",
    "group_flag",
]


def _char_diversity(text: str) -> float:
    if not text:
        return 0.0
    return len(set(text)) / len(text)


def build_features(merchants: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-merchant risk features from merchants and products."""
    price_med = products.groupby("category")["price"].median()
    price_std = products.groupby("category")["price"].std().replace(0, 1)

    rows = []
    for _, m in merchants.iterrows():
        prod = products[products["merchant_id"] == m["merchant_id"]]
        if prod.empty:
            continue

        med = price_med[m["category"]]
        std = price_std[m["category"]]
        zscores = (prod["price"] - med) / std
        price_zmax = float(np.abs(zscores).max())
        price_low_ratio = float((prod["price"] < 0.1 * med).mean())
        spam_ratio = float(prod["title"].map(_char_diversity).lt(0.45).mean())

        # A merchant is in an organized group if any other merchant shares its device group.
        group_flag = 1.0 if (pd.notna(m["device_group"]) and (merchants["device_group"] == m["device_group"]).sum() > 1) else 0.0

        rows.append({
            "merchant_id": m["merchant_id"],
            "category": m["category"],
            "addr_fake": float(m["address_fake"]),
            "addr_contradiction": float(_address_contradiction(m)),
            "price_zmax": price_zmax,
            "price_low_ratio": price_low_ratio,
            "spam_ratio": spam_ratio,
            "group_flag": group_flag,
            "true_risk": int(m["true_risk"]),
            "true_types": m["true_types"],
            "city": m["city"],
            "address_raw": m["address_raw"],
        })
    return pd.DataFrame(rows)


def _address_contradiction(m) -> bool:
    """Coarse check: coordinates should fall inside the claimed city bbox."""
    # Reuse district centers from data.py to build rough city bounds.
    from .data import DISTRICTS
    pts = DISTRICTS.get(m["city"], {})
    if not pts:
        return False
    lats = [p[0] for p in pts.values()]
    lons = [p[1] for p in pts.values()]
    in_lat = min(lats) - 0.05 <= m["lat"] <= max(lats) + 0.05
    in_lon = min(lons) - 0.05 <= m["lon"] <= max(lons) + 0.05
    return not (in_lat and in_lon)


class LogisticRegression:
    """Minimal binary logistic regression (no sklearn dependency)."""

    def __init__(self, learning_rate: float = 0.4, epochs: int = 400):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xn = (X - self.mean) / self.std
        n, d = Xn.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = Xn @ self.w + self.b
            p = self._sigmoid(z)
            grad = (Xn.T @ (p - y)) / n
            self.w -= self.lr * grad
            self.b -= self.lr * float((p - y).mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / self.std
        return self._sigmoid(Xn @ self.w + self.b)


def rule_score(features: pd.DataFrame) -> np.ndarray:
    """Deterministic first-stage signal: weighted sum of interpretable rules."""
    weights = {
        "addr_fake": 0.35,
        "addr_contradiction": 0.30,
        "price_low_ratio": 0.15,
        "spam_ratio": 0.12,
        "group_flag": 0.20,
    }
    score = np.zeros(len(features))
    for key, w in weights.items():
        score += w * np.clip(features[key].to_numpy(), 0, 1)
    return np.clip(score, 0, 1)


def detect(merchants: pd.DataFrame, products: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    features = build_features(merchants, products)

    # Stage 1: rules.
    features["rule_score"] = rule_score(features)

    # Stage 2: model, trained on a held-out split to avoid leaking the eval split.
    rng = np.random.default_rng(seed)
    mask = rng.random(len(features)) < 0.7
    X = features[FEATURE_NAMES].to_numpy(dtype=float)
    y = features["true_risk"].to_numpy(dtype=float)
    model = LogisticRegression().fit(X[mask], y[mask])
    features["model_prob"] = model.predict_proba(X)

    features["risk_score"] = 0.55 * features["rule_score"] + 0.45 * features["model_prob"]
    features["predicted_risk"] = (features["risk_score"] >= 0.5).astype(int)

    features["evidence"] = features.apply(_evidence, axis=1)
    return features


def _evidence(row) -> str:
    parts = []
    if row["addr_fake"]:
        parts.append("地址真实性异常")
    if row["addr_contradiction"]:
        parts.append("地址与坐标不符")
    if row["price_low_ratio"] > 0.3:
        parts.append(f"异常低价商品占比 {row['price_low_ratio']:.0%}")
    if row["spam_ratio"] > 0.3:
        parts.append(f"灌水标题占比 {row['spam_ratio']:.0%}")
    if row["group_flag"]:
        parts.append("疑似团伙聚集")
    return "；".join(parts) if parts else "无明显异常"
