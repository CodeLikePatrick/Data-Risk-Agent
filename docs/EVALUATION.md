# Evaluation

The benchmark dataset has ground-truth risk labels, so the metrics below are
exact rather than estimated. Reproduce them with:

```bash
python examples/run_pipeline.py
```

## Metric definitions

- **Precision** - of the flagged cases, how many were actually risky.
- **Recall** - of the actually risky cases, how many were flagged.
- **F1** - harmonic mean of precision and recall.
- **False-positive rate** - of the normal cases, how many were wrongly flagged.
- **AUC** - probability that a random risky case scores above a random normal one.
- **KS** - maximum separation between the score distributions of the two classes.
- **Lift** - how many more risky cases a segment contains than a random sample of
  the same size.
- **Gain** - cumulative share of risky cases captured down the ranked list.

## Results (seed 42, 240 merchants)

| Layer | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- |
| Detection engine | 1.00 | 0.59 | 0.74 | 0.00 |
| Review agent | 1.00 | 0.59 | 0.74 | 0.00 |

Ranking quality: AUC 0.95, KS 0.88. The top decile captures 36% of risky
merchants at a lift of 3.6.

This is a deliberately conservative operating point: precision is maximised and
the false-positive rate is zero, at the cost of recall. Lowering the threshold,
increasing the model weight or adding weak features would raise recall while
accepting more false positives. The trade-off is a business decision, not a
model parameter.

## Model comparison

The from-scratch logistic regression reaches an out-of-fold AUC of 0.93 on this
benchmark. With `models` extras installed, the same harness also scores a
gradient-boosted challenger so the interpretability gain can be weighed against
any accuracy difference.
