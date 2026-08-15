"""End-to-end demo: generate data, detect risks, run the agent, and evaluate."""

from __future__ import annotations

from ecomrisk.agent import DeterministicPolicy, OpenAICompatLLM, ToolRegistry, run_react
from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.eval import evaluate_agent, evaluate_engine


def main() -> None:
    merchants, products = generate(seed=42)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)

    engine = evaluate_engine(report)
    agent = evaluate_agent(report, tools)

    print("=" * 60)
    print("检测引擎指标（rules + logistic regression 两阶段）")
    print(f"  precision={engine['precision']} recall={engine['recall']} f1={engine['f1']} fpr={engine['false_positive_rate']}")
    print("研判 Agent 指标（确定性策略，离线可复现）")
    print(f"  precision={agent['precision']} recall={agent['recall']} f1={agent['f1']} fpr={agent['false_positive_rate']} evidence_coverage={agent['evidence_coverage']}")
    print("=" * 60)

    policy = DeterministicPolicy()
    high = report[report["true_risk"] == 1].iloc[0]
    low = report[report["true_risk"] == 0].iloc[0]
    for name, row in [("高风险样例", high), ("低风险样例", low)]:
        verdict = policy.review(tools, row["merchant_id"])
        print(f"\n[{name}] {row['merchant_id']} 真实风险={row['true_risk']} 真实类型={row['true_types']}")
        print(f"  判定={verdict['risk_level']} 置信度={verdict['confidence']} 动作={verdict['recommended_action']}")
        print(f"  证据={verdict['evidence']}")
        print(f"  说明={verdict['explanation'][:160]}")

    print("\n如需使用真实大模型，设置 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL 后运行：")
    print("  llm = OpenAICompatLLM(); run_react(llm, tools, 'M0000')")


if __name__ == "__main__":
    main()
