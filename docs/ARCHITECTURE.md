# Architecture

The system is layered so that each stage can be replaced and tested on its own.

## Layers

1. **Data** (`ecomrisk/data.py`) generates a synthetic dataset with injected risk
   labels and a rule knowledge base. Synthetic labels make evaluation exact and
   reproducible.

2. **Features** (`ecomrisk/features.py`, `ecomrisk/engine.py`, `sql/features.sql`)
   aggregates six per-merchant signals: address validity, coordinate consistency,
   price deviation, low-price share, title spam and shared device groups.

3. **Detection** (`ecomrisk/engine.py`) runs two stages. A weighted rule score
   gives a precise, explainable first pass. A logistic regression trained from
   scratch adds recall on the weaker signals. The two are blended into one score.

4. **Models** (`ecomrisk/models.py`) compares the interpretable baseline against
   a gradient-boosted challenger using five-fold out-of-fold AUC.

5. **Segmentation and association** (`ecomrisk/cluster.py`, `ecomrisk/basket.py`)
   produce merchant clusters and product co-occurrence rules.

6. **Evaluation** (`ecomrisk/metrics.py`, `ecomrisk/eval.py`) measures both the
   detection engine and the review agent with precision, recall, F1,
   false-positive rate, ROC/AUC, KS, lift and gain.

7. **Agent** (`ecomrisk/agent.py`) reviews one merchant at a time through a ReAct
   loop and a fixed set of read-only tools. A deterministic policy walks the same
   tool sequence offline, which is what makes agent evaluation reproducible.

8. **Human-in-the-loop** (`ecomrisk/review.py`) routes high and borderline scores
   to a human for confirmation or override.

## Why the model is kept away from raw data

The agent never reads tables directly. It calls `query_risk_report`,
`search_rules`, `lookup_product` and `create_case`. The registry gives every tool
a fixed input/output contract, which means tools can be mocked in unit tests and
the model can be swapped without changing the rest of the system.

## Why the agent is auditable

The final verdict is structured: risk level, violation types, confidence,
evidence, recommended action and an explanation. A `DeterministicPolicy` produces
the same verdict from the same inputs, so the system can be evaluated offline and
the LLM path can be compared against a known baseline.
