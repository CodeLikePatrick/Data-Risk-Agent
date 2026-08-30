"""Model comparison: an interpretable baseline versus a stronger boosted model.

The baseline is a logistic regression implemented from scratch in numpy, kept
for explainability. The challenger is a gradient-boosted tree model loaded
lazily (LightGBM first, then scikit-learn), so the package still runs with no
heavy dependencies and falls back gracefully when neither backend is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import FEATURE_NAMES, LogisticRegression
from .metrics import roc_auc


def _gbm_backend() -> str | None:
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except ImportError:
        pass
    try:
        import sklearn  # noqa: F401
        return "sklearn"
    except ImportError:
        return None


def _fit_predict_gbm(X_train, y_train, X_val, backend: str, seed: int) -> np.ndarray:
    if backend == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3,
            random_state=seed, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_val)[:, 1]
    if backend == "sklearn":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3, random_state=seed,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_val)[:, 1]
    raise RuntimeError("no gradient boosting backend available")


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        yield train, val


def compare_models(features: pd.DataFrame, seed: int = 42, k: int = 5) -> dict:
    """Five-fold out-of-fold AUC for the baseline and the boosted challenger."""
    X = features[FEATURE_NAMES].to_numpy(dtype=float)
    y = features["true_risk"].to_numpy(dtype=float)
    backend = _gbm_backend()

    baseline_oof = np.zeros(len(X))
    boosted_oof = np.zeros(len(X))
    for train, val in kfold_indices(len(X), k, seed):
        base = LogisticRegression().fit(X[train], y[train])
        baseline_oof[val] = base.predict_proba(X[val])
        if backend is not None:
            boosted_oof[val] = _fit_predict_gbm(X[train], y[train], X[val], backend, seed)

    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "k_folds": int(k),
        "baseline": "logistic_regression_from_scratch",
        "baseline_auc": round(roc_auc(y, baseline_oof), 4),
        "gradient_boosting_backend": backend,
        "gradient_boosting_auc": round(roc_auc(y, boosted_oof), 4) if backend else None,
    }
