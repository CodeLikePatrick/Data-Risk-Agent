"""Human-in-the-loop review gate.

Clear-cut predictions are auto-approved. A small band around the decision
threshold, plus every score above the mandatory-review cutoff, is routed to a
human for confirmation or override. This keeps the agent auditable and makes the
cost/quality trade-off explicit: reviewing costs time, but it prevents two
expensive mistakes - blocking an innocent merchant and passing a high-risk one.
"""

from __future__ import annotations

import pandas as pd

MANDATORY_REVIEW_SCORE = 0.70
AMBIGUOUS_BAND = (0.45, 0.62)


def review_policy(score: float) -> str:
    if score >= MANDATORY_REVIEW_SCORE:
        return "mandatory"
    if AMBIGUOUS_BAND[0] <= score < AMBIGUOUS_BAND[1]:
        return "advisory"
    return "auto"


class HumanReviewGate:
    """Collects items that need a human and merges decisions back into a report."""

    def __init__(self, mandatory: float = MANDATORY_REVIEW_SCORE, band: tuple[float, float] = AMBIGUOUS_BAND):
        self.mandatory = mandatory
        self.band = band
        self.decisions: dict[str, dict] = {}

    def policy(self, score: float) -> str:
        if score >= self.mandatory:
            return "mandatory"
        if self.band[0] <= score < self.band[1]:
            return "advisory"
        return "auto"

    def queue(self, report: pd.DataFrame) -> list[dict]:
        pending = []
        for _, row in report.iterrows():
            pol = self.policy(float(row["risk_score"]))
            if pol == "auto":
                continue
            pending.append({
                "merchant_id": row["merchant_id"],
                "risk_score": round(float(row["risk_score"]), 4),
                "predicted_risk": int(row["predicted_risk"]),
                "evidence": row["evidence"],
                "policy": pol,
            })
        return pending

    def decide(self, merchant_id: str, approve: bool, note: str = "") -> None:
        self.decisions[merchant_id] = {"approved": approve, "note": note}

    def apply(self, report: pd.DataFrame) -> pd.DataFrame:
        """Return the report with ``final_decision`` and ``human_reviewed`` columns.

        Items that still require review and have not been decided are held safe
        (``final_decision = 0``) until a human acts.
        """
        out = report.copy()
        final = []
        reviewed = []
        for _, row in out.iterrows():
            mid = row["merchant_id"]
            pol = self.policy(float(row["risk_score"]))
            if mid in self.decisions:
                decision = self.decisions[mid]
                final.append(int(row["predicted_risk"]) if decision["approved"] else 0)
                reviewed.append(1)
            elif pol == "auto":
                final.append(int(row["predicted_risk"]))
                reviewed.append(0)
            else:
                final.append(0)
                reviewed.append(1)
        out["final_decision"] = final
        out["human_reviewed"] = reviewed
        return out
