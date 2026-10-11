# ecom-risk-agent

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An e-commerce risk intelligence pipeline: a two-stage detection engine that turns
raw merchant and product data into an interpretable risk score, and an LLM review
agent that turns the score into a defensible, human-reviewable decision.

The core is deliberately dependency-free (numpy and pandas only). Every model,
metric and clustering step is implemented from scratch so the numbers are
auditable and the trade-offs are explicit.

## Why this exists

Risk decisions usually fail in one of two expensive ways: blocking an innocent
merchant, or letting a genuinely risky one through. Rules are precise but miss
weak signals; a black-box model finds more cases but is hard to explain and even
harder to defend. An LLM alone can reason about evidence but is not deterministic
or easy to evaluate.

This project combines the three ideas instead of choosing between them: rules
provide a precise, explainable first pass; a statistical model generalises beyond
the rules; and an agent uses both, with a tool registry that keeps the model away
from raw data, a deterministic offline policy for reproducible evaluation, and a
human-in-the-loop gate for the decisions that matter.

## Architecture

```text
  merchants + products
          |
          v
  feature engineering            engine.py / features.py / sql/features.sql
          |
          v
  two-stage detection            rule_score -> logistic regression -> risk_score
          |
          v
  review agent (ReAct)           query report -> search rules -> inspect sample
          |                        -> recommend action -> create case
          v
  human-in-the-loop gate         review.py
```

## What is included

- **Feature engineering** for six interpretable risk signals, with an in-memory
  implementation and a matching Hive-style SQL version.
- **Two-stage detection**: a weighted rule score for precision plus a logistic
  regression trained from scratch for recall.
- **Model comparison**: five-fold out-of-fold AUC for the interpretable baseline
  versus a gradient-boosted challenger (LightGBM first, scikit-learn as fallback,
  both optional).
- **Clustering**: numpy K-Means and DBSCAN for merchant segmentation, with a
  silhouette score and a noise label that flags unusual merchants.
- **Market-basket analysis**: Apriori and association rules with support,
  confidence and lift.
- **Evaluation**: confusion-matrix metrics, ROC/AUC, KS, lift and gain tables,
  and regression error metrics, all implemented from scratch.
- **Review agent**: a tool registry, a ReAct loop, an OpenAI-compatible client,
  and a deterministic policy that runs the same tool sequence offline.
- **Human-in-the-loop**: a review gate that routes high and borderline scores to
  a human and merges the decisions back.
- **A sample report** and a **Streamlit dashboard** for presenting the result.

## Quick start

```bash
git clone <your-fork>
cd ecom-risk-agent
pip install -e ".[dev,models,viz]"
python examples/run_pipeline.py
```

To exercise the real LLM path, set `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`OPENAI_MODEL` (any OpenAI-compatible endpoint works) and run:

```bash
python examples/run_review.py
```

Run the test suite with pytest, or without any extra dependency:

```bash
pytest -q                 # if pytest is installed
python tests/run_tests.py # dependency-free runner
```

## Example output (synthetic benchmark)

The benchmark dataset has injected risk labels, so every number below is exact
and reproducible.

| Layer | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- |
| Detection engine | 1.00 | 0.59 | 0.74 | 0.00 |
| Review agent (deterministic) | 1.00 | 0.59 | 0.74 | 0.00 |

Ranking quality on the same run: AUC 0.95, KS 0.88, and the top decile alone
captures 36% of all risky merchants at a lift of 3.6.

## Repository layout

```text
ecomrisk/            package
sql/                 Hive-style profiling and feature SQL
notebooks/           EDA and model-evaluation notebooks
dashboard/           Streamlit app
reports/             generated sample report
examples/            runnable entry points
tests/               pytest-compatible tests + dependency-free runner
docs/                architecture, evaluation, decisions
data/                data dictionary and a small committed sample
```

## Human-in-the-loop

Clear predictions are auto-approved. Scores above the mandatory-review cutoff and
scores in the band around the decision threshold are queued for a human, who can
confirm or override. Un-reviewed borderline items are held safe until someone
acts. This makes the cost of review explicit: time spent reviewing versus the cost
of the two errors it prevents.

## Notes

The data is synthetic and exists only to make the benchmark reproducible and the
labels exact. It does not represent any real platform, customer or policy. The
rules and metrics illustrate a method, not a production risk standard.

## License

MIT. See [LICENSE](LICENSE).
