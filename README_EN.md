# E-commerce Risk Intelligence

[中文](README.md)

An e-commerce **risk review agent plus anomaly detection engine**: the lower layer uses a two-stage "rules + logistic regression" pipeline to flag four risk types (address, price, text, and group), and the upper layer uses an agent to run a review loop of "check risk → look up rules → inspect a product → recommend an action → create a case", with a reproducible evaluation suite.

The point is not "calling an LLM", but bringing risk-control thinking — precision, recall, and false-positive rate — into an agent, so the result is measurable, explainable, and reversible.

## Background and motivation

E-commerce risk control has two lines of work: anomaly detection (finding risks) and policy handling (defining, recalling, identifying, and disposing of risks). Rules hard-coded into code are hard to explain and iterate on; large models understand semantics but are uncontrollable and hard to evaluate when used directly for decisions.

This project joins the two: the detection engine emits structured risk signals and evidence, and the agent turns those signals into actionable, evidence-backed verdicts.

## Architecture

```
                    ┌──────────────────────────────┐
  products / address │   Anomaly detection engine    │
  / merchants ─────► │   features → rules → model     │
                    │   output: risk score + evidence│
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   Risk review agent            │
                    │   ReAct: signals → rules →     │
                    │   product → action → case      │
                    └──────────────────────────────┘
                                   ▼
                     structured verdict + evidence + confidence
```

## Four risk types

- Address authenticity: the address cannot be resolved, or the city/district contradicts the coordinates
- Price anomaly: price deviates from the category median by more than 3 standard deviations, or is clearly below cost
- Text spam: templated titles, keyword stuffing, or low character entropy
- Group collusion: multiple merchants share devices, addresses, or contacts

## Project layout

```
ecomrisk/
  data.py      synthetic data generation + rule knowledge base
  engine.py    feature engineering + two-stage scoring + detection orchestration
  agent.py     tool registry + ReAct loop + deterministic policy + OpenAI-compatible client
  eval.py      evaluation metrics for the engine and the agent
main.py        end-to-end demo entry point
data/rules.md  retrievable risk-rule knowledge base
tests/         module-level test suites
examples/      single-merchant review example
docs/DESIGN.md design decisions
```

## Quick start

```bash
cd ecom-risk-agent
python main.py
```

Only `numpy` and `pandas` are required. `main.py` generates data, runs detection, runs the agent, prints evaluation metrics, and shows complete review results for two sample merchants.

## Approach

### Feature engineering

Six interpretable features are aggregated from raw products and addresses: fake address, address-coordinate contradiction, maximum price z-score, share of abnormally cheap products, share of spammy titles, and group flag.

### Two-stage detection

1. Rules first: produce an interpretable `rule_score` to keep precision high and decisions explainable.
2. Model second: a logistic regression implemented from scratch with numpy produces `model_prob`, catching weak signals rules miss.
3. Final `risk_score = 0.55 * rule_score + 0.45 * model_prob`.

The model is intentionally dependency-free and hand-written: gradient descent, feature standardization, and the sigmoid function are all explicit, which keeps the implementation reviewable and honest.

### Agent design

The agent uses a ReAct loop with clearly typed tools:

- `query_risk_report`: read a merchant's risk score and evidence
- `search_rules`: retrieve matching rules from the knowledge base
- `lookup_product`: sample a product as supporting context
- `create_case`: create a disposition case

The final output is a structured verdict: risk level, violation types, confidence, evidence, recommended action, and explanation.

A built-in `DeterministicPolicy` walks the exact same tool sequence for offline, reproducible evaluation. Setting `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` switches `run_react` to a real model.

### Evaluation

The synthetic data carries ground-truth labels, so precision, recall, F1, and false-positive rate can be reported rather than assumed.

| Metric | Detection engine | Review agent |
| --- | --- | --- |
| precision | 1.00 | 1.00 |
| recall | 0.59 | 0.59 |
| f1 | 0.74 | 0.74 |
| false positive rate | 0.00 | 0.00 |
| evidence coverage on positives | - | 1.00 |

This is a deliberately conservative setup: high precision and low false positives at the cost of recall. In practice, recall can be improved by lowering the threshold, increasing model weight, or adding more weak features — a useful set of trade-offs to make explicit.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

There are 21 test cases covering four layers:

- data generation: reproducibility, label consistency, merchant-product relations, and address-fake consistency
- detection engine: feature ranges, coordinate contradiction, rule-score bounds, logistic regression convergence, and detection reproducibility
- review agent: tool output shapes, rule retrieval, risk-level mapping, ReAct loop, and missing-final-answer handling
- evaluation: precision / recall / F1 / false-positive-rate computation and edge cases

GitHub Actions runs `pytest -q` on every push and pull request.

## Using a real LLM

```python
from ecomrisk.agent import OpenAICompatLLM, run_react

llm = OpenAICompatLLM()  # reads OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
verdict = run_react(llm, tools, "M0000")
```

Works with any OpenAI-compatible service, including DeepSeek and OpenRouter.

## Disclaimer

The data is synthetic and exists only to demonstrate methodology and engineering. The rules and metrics do not reflect any real platform's risk-control standard.
