"""Streamlit dashboard for the risk review pipeline.

Run with: ``pip install -e ".[models]" streamlit`` then
``streamlit run dashboard/app.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecomrisk.agent import DeterministicPolicy, ToolRegistry
from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.metrics import lift_table
from ecomrisk.review import HumanReviewGate


@st.cache_data
def load(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merchants, products = generate(seed=seed)
    report = detect(merchants, products)
    return merchants, products, report


st.set_page_config(page_title="Risk Review", layout="wide")
st.title("E-commerce risk review")
st.caption("Two-stage detection + review agent + human-in-the-loop")

seed = st.sidebar.number_input("Random seed", 0, 1000, 42)
merchants, products, report = load(seed)
tools = ToolRegistry(report, merchants, products)
policy = DeterministicPolicy()

top_n = st.sidebar.slider("Top risk merchants", 5, 50, 20)
tab_leaderboard, tab_lift, tab_review = st.tabs(["Leaderboard", "Lift", "Review"])

with tab_leaderboard:
    leaderboard = report.sort_values("risk_score", ascending=False).head(top_n)
    st.dataframe(leaderboard[["merchant_id", "category", "risk_score", "predicted_risk", "evidence"]], use_container_width=True)

with tab_lift:
    y = report["true_risk"].to_numpy()
    scores = report["risk_score"].to_numpy()
    st.dataframe(lift_table(y, scores, n_bins=10), use_container_width=True)

with tab_review:
    gate = HumanReviewGate()
    pending = pd.DataFrame(gate.queue(report))
    if pending.empty:
        st.write("No items require review.")
    else:
        st.dataframe(pending, use_container_width=True)
        merchant = st.selectbox("Review merchant", pending["merchant_id"].tolist())
        verdict = policy.review(tools, merchant)
        st.json(verdict)
