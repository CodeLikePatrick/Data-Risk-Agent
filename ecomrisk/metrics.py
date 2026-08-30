"""Model evaluation metrics implemented from scratch (numpy only).

Everything here is intentionally dependency-free so that every number is
auditable and every definition is explicit: ROC and AUC, KS, lift and gain
tables, and regression error metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_arrays(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same length")
    return y, s


def roc_points(y_true, scores) -> list[tuple[float, float]]:
    """Return (false positive rate, true positive rate) points in increasing fpr."""
    y, s = _check_arrays(y_true, scores)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    order = np.argsort(-s)
    y_sorted = y[order]
    tpr = np.concatenate([[0.0], np.cumsum(y_sorted) / pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_sorted) / neg])
    return list(zip(fpr.tolist(), tpr.tolist()))


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve, computed with average ranks to handle ties."""
    y, s = _check_arrays(y_true, scores)
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(sorted_s):
        j = i
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j

    sum_ranks_pos = float(ranks[pos_mask].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_statistic(y_true, scores) -> float:
    """Maximum separation between the score distributions of the two classes."""
    y, s = _check_arrays(y_true, scores)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tpr = np.cumsum(y_sorted) / pos
    fpr = np.cumsum(1 - y_sorted) / neg
    return float(np.max(np.abs(tpr - fpr)))


def lift_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    """Decile-level lift and cumulative gain for a ranked list of scores."""
    y, s = _check_arrays(y_true, scores)
    pos = int(y.sum())
    if pos == 0 or len(y) == 0:
        return pd.DataFrame(columns=["decile", "size", "positives", "cum_positives", "cum_gain", "lift"])
    order = np.argsort(-s)
    bins = [b for b in np.array_split(order, n_bins) if len(b) > 0]
    rows = []
    for i, idx in enumerate(bins, 1):
        decile_pos = int(y[idx].sum())
        expected = pos * len(idx) / len(y)
        rows.append({
            "decile": i,
            "size": len(idx),
            "positives": decile_pos,
            "cum_positives": int(y[np.concatenate(bins[:i])].sum()),
            "cum_gain": round(float(y[np.concatenate(bins[:i])].sum()) / pos, 4),
            "lift": round(float(decile_pos) / expected, 4) if expected else 0.0,
        })
    return pd.DataFrame(rows)


def gain_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    return lift_table(y_true, scores, n_bins=n_bins)[["decile", "size", "cum_positives", "cum_gain"]]


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    residual = y - p
    sse = float((residual ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst else 0.0
    mae = float(np.abs(residual).mean())
    mse = float((residual ** 2).mean())
    rmse = float(np.sqrt(mse))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = residual / np.where(y != 0, y, np.nan)
        mape = float(np.nanmean(np.abs(ratio)) * 100.0)
    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if np.isfinite(mape) else float("nan"),
    }
