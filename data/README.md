# Data

The pipeline runs on a small synthetic dataset with injected risk labels. Synthetic
data is used deliberately: every record has a known label, so precision, recall,
false-positive rate, lift and gain can be measured exactly and the benchmark is
fully reproducible.

- `sample/merchants.csv` and `sample/products.csv` are committed so the pipeline
  runs offline without needing to regenerate anything.
- `../ecomrisk/data.py` is the generator. Change the seed to get a different but
  deterministic sample.
- `data_dictionary.md` documents the columns, grain and definitions.

The data is illustrative and does not represent any real platform or customer.
