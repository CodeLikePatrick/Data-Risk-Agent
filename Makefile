.PHONY: install test demo report

install:
	pip install -e ".[dev,models]"

test:
	python tests/run_tests.py

demo:
	python examples/run_pipeline.py

report:
	python examples/build_report.py
