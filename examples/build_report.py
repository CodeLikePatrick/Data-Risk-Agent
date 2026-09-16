"""Render a self-contained HTML sample report (no plotting dependency)."""

from __future__ import annotations

import html
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ecomrisk.data import generate
from ecomrisk.engine import detect
from ecomrisk.eval import evaluate_engine
from ecomrisk.metrics import ks_statistic, lift_table, roc_auc


def _svg_lift(y_true, scores, width=560, height=260) -> str:
    table = lift_table(y_true, scores, n_bins=10)
    points = []
    max_x = float(table["cum_positives"].max() or 1)
    max_y = float(table["cum_gain"].max() or 1)
    for _, row in table.iterrows():
        x = 30 + (row["cum_positives"] / max_x) * (width - 70)
        y = height - 30 - (row["cum_gain"] / max_y) * (height - 70)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="cumulative gain curve" style="max-width:100%;height:auto">'
        f'<line x1="30" y1="{height-30}" x2="{width-40}" y2="{height-30}" stroke="#999"/>'
        f'<line x1="30" y1="{height-30}" x2="30" y2="30" stroke="#999"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#24425f" stroke-width="2"/>'
        "</svg>"
    )


def main() -> None:
    merchants, products = generate(seed=42)
    report = detect(merchants, products)
    y = report["true_risk"].to_numpy()
    scores = report["risk_score"].to_numpy()

    metrics = evaluate_engine(report)
    auc = roc_auc(y, scores)
    ks = ks_statistic(y, scores)
    lift = lift_table(y, scores, n_bins=10)
    top = report.sort_values("risk_score", ascending=False).head(10)

    top_rows = "".join(
        f"<tr><td>{html.escape(str(r['merchant_id']))}</td>"
        f"<td>{r['risk_score']:.3f}</td>"
        f"<td>{html.escape(str(r['evidence']))}</td></tr>"
        for _, r in top.iterrows()
    )
    lift_rows = "".join(
        f"<tr><td>{r['decile']}</td><td>{r['size']}</td>"
        f"<td>{r['positives']}</td><td>{r['cum_gain']}</td><td>{r['lift']}</td></tr>"
        for _, r in lift.iterrows()
    )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk Review Sample Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:40px;color:#1a1a1a;line-height:1.5}}
h1{{font-size:22px}} h2{{font-size:16px;margin-top:32px}}
table{{border-collapse:collapse;width:100%;max-width:820px;margin-top:12px}}
th,td{{border:1px solid #d9d9d9;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#24425f;color:#fff}}
.kpi{{display:inline-block;margin-right:28px;font-size:15px}}
.kpi b{{font-size:24px;font-weight:600}}
</style></head><body>
<h1>Risk Review Sample Report</h1>
<p>Generated from the synthetic benchmark dataset with injected labels. The two-stage
engine scores merchants, and the table below shows how well the ranked list separates
risk from non-risk.</p>
<div>
  <span class="kpi"><b>{metrics['precision']}</b> precision</span>
  <span class="kpi"><b>{metrics['recall']}</b> recall</span>
  <span class="kpi"><b>{metrics['f1']}</b> F1</span>
  <span class="kpi"><b>{metrics['false_positive_rate']}</b> false-positive rate</span>
  <span class="kpi"><b>{auc:.3f}</b> AUC</span>
  <span class="kpi"><b>{ks:.3f}</b> KS</span>
</div>
{_svg_lift(y, scores)}
<h2>Cumulative gain and lift by decile</h2>
<table><thead><tr><th>Decile</th><th>Size</th><th>Positives</th><th>Cum gain</th><th>Lift</th></tr></thead>
<tbody>{lift_rows}</tbody></table>
<h2>Top risk merchants</h2>
<table><thead><tr><th>Merchant</th><th>Score</th><th>Evidence</th></tr></thead>
<tbody>{top_rows}</tbody></table>
</body></html>"""

    out = Path(__file__).resolve().parents[1] / "reports" / "sample_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
