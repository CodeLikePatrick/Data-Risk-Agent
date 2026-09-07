"""Feature engineering entry points.

The per-merchant risk features are computed in-memory by
:func:`ecomrisk.engine.build_features` and mirrored as SQL in
``sql/features.sql`` so the same logic can be executed inside a warehouse. This
module re-exports the canonical names and documents the mapping from each
feature to the business signal it captures.
"""

from __future__ import annotations

from .engine import FEATURE_NAMES, build_features, rule_score

FEATURE_DEFINITIONS = {
    "addr_fake": "Whether the registered address is unparsable or fictitious.",
    "addr_contradiction": "Whether the coordinates fall outside the claimed city.",
    "price_zmax": "Max absolute z-score of product prices against their category.",
    "price_low_ratio": "Share of products priced below 10% of the category median.",
    "spam_ratio": "Share of product titles with low character diversity.",
    "group_flag": "Whether the merchant shares a device group with another merchant.",
}

__all__ = ["FEATURE_NAMES", "FEATURE_DEFINITIONS", "build_features", "rule_score"]
