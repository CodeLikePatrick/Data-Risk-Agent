"""Merchant segmentation with numpy-only clustering implementations.

K-Means is the interpretable, scalable default; DBSCAN is included because it
does not force every point into a cluster and surfaces noise, which maps to
"unusual" merchants that deserve a second look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLUSTER_FEATURES = ["addr_contradiction", "price_zmax", "price_low_ratio", "spam_ratio", "group_flag"]


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels, centers


def dbscan(X: np.ndarray, eps: float, min_samples: int = 3) -> np.ndarray:
    """Density clustering; returns -1 for noise points."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    UNVISITED, NOISE = -2, -1
    labels = np.full(n, UNVISITED, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        neighbors = list(np.where(dist[i] <= eps)[0])
        if len(neighbors) < min_samples:
            labels[i] = NOISE
            continue
        labels[i] = cluster_id
        seeds = neighbors[:]
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            cursor += 1
            if labels[j] == NOISE:
                labels[j] = cluster_id
            if labels[j] != UNVISITED:
                continue
            labels[j] = cluster_id
            n_j = list(np.where(dist[j] <= eps)[0])
            if len(n_j) >= min_samples:
                seeds.extend(n_j)
        cluster_id += 1
    return labels


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = len(X)
    if len(np.unique(labels)) <= 1 or n <= 1:
        return 0.0
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    total = 0.0
    for i in range(n):
        same = labels == labels[i]
        a_i = dist[i][same].mean() if same.sum() > 1 else 0.0
        other = [c for c in np.unique(labels) if c != labels[i]]
        b_i = min(float(dist[i][labels == c].mean()) for c in other) if other else 0.0
        denom = max(a_i, b_i)
        total += (b_i - a_i) / denom if denom > 0 else 0.0
    return float(total / n)


def cluster_merchants(features: pd.DataFrame, k: int = 3, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    X = standardize(features[CLUSTER_FEATURES].to_numpy(dtype=float))
    labels, _ = kmeans(X, k, seed=seed)
    noise = dbscan(X, eps=0.9, min_samples=4)
    out = features[["merchant_id", "category", "true_risk"]].copy()
    out["cluster"] = labels
    out["dbscan_label"] = noise
    info = {
        "k": k,
        "kmeans_silhouette": round(silhouette_score(X, labels), 4),
        "dbscan_noise_count": int((noise == -1).sum()),
    }
    return out, info
