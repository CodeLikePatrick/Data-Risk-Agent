"""Minimal dependency-free test runner (pytest is preferred but optional)."""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    modules = [
        "tests.test_agent",
        "tests.test_basket",
        "tests.test_cluster",
        "tests.test_data",
        "tests.test_engine",
        "tests.test_eval",
        "tests.test_metrics",
        "tests.test_models",
        "tests.test_review",
        "tests.test_smoke",
    ]
    passed = failed = 0
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {module_name}.{name}")
                traceback.print_exc()
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
