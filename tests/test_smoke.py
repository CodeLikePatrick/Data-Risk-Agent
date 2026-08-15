"""End-to-end smoke tests that keep the repo reviewable and CI-friendly."""

from ecomrisk.agent import DeterministicPolicy, ToolRegistry
from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.eval import evaluate_agent, evaluate_engine


def test_pipeline_produces_expected_columns():
    merchants, products = generate(seed=0, n_merchants=60)
    report = detect(merchants, products)
    expected = {"risk_score", "predicted_risk", "evidence", "rule_score", "model_prob"}
    assert expected.issubset(set(report.columns))


def test_detection_finds_some_risks_without_false_positives():
    merchants, products = generate(seed=1, n_merchants=80)
    report = detect(merchants, products)
    assert report["true_risk"].sum() > 0
    metrics = evaluate_engine(report)
    assert metrics["precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert 0.0 < metrics["recall"] < 1.0


def test_agent_returns_structured_verdict():
    merchants, products = generate(seed=2, n_merchants=50)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)
    policy = DeterministicPolicy()
    verdict = policy.review(tools, report.iloc[0]["merchant_id"])
    required = {"merchant_id", "risk_level", "violation_types", "confidence", "evidence", "recommended_action", "explanation"}
    assert required.issubset(set(verdict))
    assert verdict["risk_level"] in {"high", "medium", "low"}
    assert isinstance(verdict["confidence"], float)


def test_agent_metrics_compute():
    merchants, products = generate(seed=3, n_merchants=60)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)
    metrics = evaluate_agent(report, tools, max_cases=40)
    assert {"precision", "recall", "f1", "false_positive_rate", "evidence_coverage"} == set(metrics)
