"""Tests for the human-in-the-loop review gate."""

import pandas as pd

from ecomrisk.review import HumanReviewGate, review_policy


def test_review_policy_thresholds():
    assert review_policy(0.8) == "mandatory"
    assert review_policy(0.5) == "advisory"
    assert review_policy(0.1) == "auto"


def _report():
    return pd.DataFrame([
        {"merchant_id": "M1", "risk_score": 0.9, "predicted_risk": 1, "evidence": "地址真实性异常"},
        {"merchant_id": "M2", "risk_score": 0.5, "predicted_risk": 1, "evidence": "疑似团伙聚集"},
        {"merchant_id": "M3", "risk_score": 0.1, "predicted_risk": 0, "evidence": "无明显异常"},
    ])


def test_queue_collects_only_review_items():
    gate = HumanReviewGate()
    pending = gate.queue(_report())
    assert {p["merchant_id"] for p in pending} == {"M1", "M2"}
    assert {p["policy"] for p in pending} == {"mandatory", "advisory"}


def test_apply_holds_undecided_and_honors_decisions():
    gate = HumanReviewGate()
    report = _report()
    gate.decide("M1", approve=True, note="ok")
    gate.decide("M2", approve=False, note="override to safe")
    out = gate.apply(report)
    final = dict(zip(out["merchant_id"], out["final_decision"]))
    assert final["M1"] == 1
    assert final["M2"] == 0
    assert final["M3"] == 0
    assert out.loc[out["merchant_id"] == "M1", "human_reviewed"].iloc[0] == 1
