"""Tests for tools, the deterministic policy, and the ReAct loop."""

from ecomrisk.agent import DeterministicPolicy, ToolRegistry, run_react
from ecomrisk.data import generate
from ecomrisk.engine import detect


def _setup(seed=10, n=60):
    merchants, products = generate(seed=seed, n_merchants=n)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)
    return merchants, products, report, tools


def test_tools_return_expected_shapes():
    _, _, report, tools = _setup()
    mid = report.iloc[0]["merchant_id"]
    assert set(tools.query_risk_report(mid)) >= {"merchant_id", "risk_score", "predicted_risk", "evidence"}
    assert "sample" in tools.lookup_product(mid)
    assert "case_id" in tools.create_case(mid, "下架", "测试")


def test_search_rules_filters_by_keyword():
    _, _, _, tools = _setup()
    ids = [r["id"] for r in tools.search_rules("价格")]
    assert "PRICE-02" in ids


def test_rules_knowledge_base_is_enriched():
    from ecomrisk.data import RULES
    ids = {r["id"] for r in RULES}
    assert len(RULES) >= 20
    assert {"ADDR-01", "PRICE-01", "TEXT-01", "GROUP-01", "ORDER-01"} <= ids


def test_search_rules_maps_evidence_keywords():
    _, _, _, tools = _setup()
    assert any(r["id"].startswith("PRICE") for r in tools.search_rules("异常低价商品占比 70%"))
    assert any(r["id"].startswith("TEXT") for r in tools.search_rules("灌水标题占比 50%"))
    assert any(r["id"].startswith("GROUP") for r in tools.search_rules("疑似团伙聚集"))


def test_deterministic_policy_maps_score_to_level():
    _, _, report, tools = _setup(seed=11, n=100)
    policy = DeterministicPolicy()
    high_id = report.sort_values("risk_score", ascending=False).iloc[0]["merchant_id"]
    low_id = report.sort_values("risk_score", ascending=True).iloc[0]["merchant_id"]
    assert policy.review(tools, high_id)["risk_level"] in {"high", "medium"}
    assert policy.review(tools, low_id)["risk_level"] == "low"


def test_run_react_executes_tools_and_returns_final():
    _, _, report, tools = _setup(seed=12, n=30)
    mid = report.iloc[0]["merchant_id"]

    class ScriptedLLM:
        def __init__(self, actions):
            self._it = iter(actions)

        def chat(self, messages):
            return next(self._it)

    actions = [
        {"tool": "query_risk_report", "tool_input": {"merchant_id": mid}},
        {"tool": "search_rules", "tool_input": {"query": "价格"}},
        {"final": {"merchant_id": mid, "risk_level": "medium", "violation_types": [], "confidence": 0.7, "evidence": ["x"], "recommended_action": "下架", "explanation": "ok"}},
    ]
    verdict = run_react(ScriptedLLM(actions), tools, mid)
    assert verdict["merchant_id"] == mid
    assert verdict["risk_level"] == "medium"


def test_run_react_raises_without_final_answer():
    _, _, report, tools = _setup(seed=13, n=20)
    mid = report.iloc[0]["merchant_id"]

    class LoopingLLM:
        def chat(self, messages):
            return {"tool": "search_rules", "tool_input": {"query": "价格"}}

    try:
        run_react(LoopingLLM(), tools, mid, max_steps=2)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")
