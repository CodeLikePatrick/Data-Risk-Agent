"""Benchmark for the detection engine and the review agent."""

from __future__ import annotations

import pandas as pd

from .agent import DeterministicPolicy, ToolRegistry


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    neg = sum(1 for a in y_true if a == 0)
    fpr = fp / neg if neg else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "false_positive_rate": round(fpr, 3)}


def evaluate_engine(report: pd.DataFrame) -> dict:
    return _metrics(report["true_risk"].tolist(), report["predicted_risk"].tolist())


def evaluate_agent(report: pd.DataFrame, tools: ToolRegistry, max_cases: int = 200) -> dict:
    rows = report.sample(n=min(max_cases, len(report)), random_state=0)
    policy = DeterministicPolicy()
    y_true, y_pred, evidence = [], [], []
    for _, row in rows.iterrows():
        verdict = policy.review(tools, row["merchant_id"])
        y_true.append(int(row["true_risk"]))
        y_pred.append(1 if verdict["risk_level"] in ("high", "medium") else 0)
        evidence.append(1 if verdict["evidence"] else 0)
    m = _metrics(y_true, y_pred)
    positive_idx = [i for i, p in enumerate(y_pred) if p == 1]
    m["evidence_coverage"] = round(sum(evidence[i] for i in positive_idx) / len(positive_idx), 3) if positive_idx else 0.0
    return m
