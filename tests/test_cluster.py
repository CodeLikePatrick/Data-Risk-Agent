"""Tests for the clustering module."""

import numpy as np

from ecomrisk.cluster import dbscan, kmeans, silhouette_score
from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.cluster import cluster_merchants


def test_kmeans_returns_requested_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
    labels, _ = kmeans(X, k=2, seed=0)
    assert len(labels) == 60
    assert len(set(labels)) == 2


def test_dbscan_marks_outliers_as_noise():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    labels = dbscan(X, eps=0.3, min_samples=2)
    assert list(labels).count(-1) == 1


def test_silhouette_is_bounded():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = np.array([0, 0, 1, 1])
    assert -1.0 <= silhouette_score(X, labels) <= 1.0


def test_cluster_merchants_adds_labels():
    merchants, products = generate(seed=2, n_merchants=60)
    report = detect(merchants, products)
    clustered, info = cluster_merchants(report, k=3, seed=0)
    assert {"cluster", "dbscan_label"}.issubset(set(clustered.columns))
    assert {"k", "kmeans_silhouette", "dbscan_noise_count"} <= set(info)
    assert clustered["cluster"].nunique() <= 3
