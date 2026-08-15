"""Minimal example: review a single merchant."""

from ecomrisk.agent import DeterministicPolicy, OpenAICompatLLM, ToolRegistry, run_react
from ecomrisk.data import generate
from ecomrisk.engine import detect


def main() -> None:
    merchants, products = generate(seed=42, n_merchants=100)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)

    merchant_id = report.sort_values("risk_score", ascending=False).iloc[0]["merchant_id"]
    print("Offline deterministic review:")
    print(DeterministicPolicy().review(tools, merchant_id))

    try:
        llm = OpenAICompatLLM()
        if llm.api_key:
            print("\nLLM review:")
            print(run_react(llm, tools, merchant_id))
        else:
            print("\nSet OPENAI_API_KEY to run the real LLM path.")
    except Exception as exc:  # noqa: BLE001
        print("\nLLM path skipped:", exc)


if __name__ == "__main__":
    main()
