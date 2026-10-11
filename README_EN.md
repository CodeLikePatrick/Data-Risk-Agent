# Quick start

The full English README is in [README.md](README.md). This file is a condensed
installation and usage guide.

```bash
git clone <your-fork>
cd ecom-risk-agent
pip install -e ".[dev,models,viz]"
python examples/run_pipeline.py   # end-to-end demo
python examples/build_report.py   # write reports/sample_report.html
python tests/run_tests.py         # run tests without pytest
```

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` and run
`python examples/run_review.py` to use a real OpenAI-compatible model.

Optional extras:

- `models` installs LightGBM for the gradient-boosted challenger.
- `viz` installs matplotlib for the notebooks.
- `dev` installs pytest and scikit-learn for development.
