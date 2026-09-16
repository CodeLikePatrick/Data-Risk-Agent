"""End-to-end walkthrough.

Generate data, extract features, score risk, compare models, rank the results,
segment merchants, mine association rules, run the review agent, and close the
loop with a human-in-the-loop gate.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ecomrisk.agent import DeterministicPolicy, ToolRegistry
from ecomrisk.basket import apriori, association_rules, sample_baskets
from ecomrisk.cluster import cluster_merchants
from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.eval import evaluate_agent, evaluate_engine
from ecomrisk.metrics import ks_statistic, lift_table, roc_auc
from ecomrisk.models import compare_models
from ecomrisk.review import HumanReviewGate


def main() -> None:
    merchants, products = generate(seed=42)
    report = detect(merchants, products)
    tools = ToolRegistry(report, merchants, products)

    print("1) Two-stage detection engine")
    print("   ", evaluate_engine(report))

    print("2) Model comparison (5-fold, out-of-fold AUC)")
    print("   ", compare_models(report, seed=42))

    print("3) Ranking quality")
    y = report["true_risk"].to_numpy()
    scores = report["risk_score"].to_numpy()
    print("    AUC:", round(roc_auc(y, scores), 4), "KS:", round(ks_statistic(y, scores), 4))
    print(lift_table(y, scores, n_bins=10).head(6).to_string(index=False))

    print("4) Merchant segmentation")
    clustered, info = cluster_merchants(report, k=3, seed=0)
    print("   ", info)
    print(clustered.groupby("cluster")["merchant_id"].count().to_string())

    print("5) Market-basket analysis")
    baskets = sample_baskets(products, n_orders=400, seed=7)
    frequent = apriori(baskets, min_support=0.05, max_size=2)
    rules = association_rules(frequent, baskets, min_confidence=0.3)
    for rule in rules[:5]:
        print("   ", rule)

    print("6) Review agent and human-in-the-loop")
    print("   ", evaluate_agent(report, tools))
    gate = HumanReviewGate()
    pending = gate.queue(report)
    print(f"    pending human review: {len(pending)}")
    if pending:
        gate.decide(pending[0]["merchant_id"], approve=True, note="manual spot check")
    final = gate.apply(report)
    print("    final risk flags after review:", int(final["final_decision"].sum()))

    print("\nRun the real LLM path with examples/run_review.py after setting OPENAI_API_KEY.")


if __name__ == "__main__":
    main()
