# Contributing

Thanks for your interest in contributing.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Guidelines

- Keep the engine deterministic and explainable; document any new feature in `engine.py`.
- Every new risk signal should map to a rule in `data/rules.md` when possible.
- Add or update tests when changing detection or agent behavior.
- Run the full smoke test before opening a pull request.

## Pull requests

Open a pull request against `main`, describe the problem and the approach, and
link any related issue. Keep unrelated changes out of a single PR.
