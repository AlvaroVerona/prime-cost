import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, SLATE, WINE, eur_short, explain, kpis, need_outputs, pct, setup

setup("Scenarios", "5,000 simulated futures for the next 12 months. Demand growth and noise and the volatility of ingredient prices are estimated from the bar's own history.")
need_outputs()
sc = read_json("scenarios_summary.json")
sq, rec = sc["status_quo"], sc["recommendations"]
kpis([
    ("Expected EBITDA, keep running as is", eur_short(sq["annual_ebitda_mean"]), f"P5 {eur_short(sq['annual_ebitda_p5'])}"),
    ("Expected EBITDA, with recommendations", eur_short(rec["annual_ebitda_mean"]), f"+{eur_short(sc['uplift_mean'])}"),
    ("Probability of a loss-making year", pct(sq["prob_annual_loss"]), None),
    ("Loss-making months in a year", f"{sq['expected_loss_months']:.1f}", f"{rec['expected_loss_months']:.1f} with recommendations"),
])
st.divider()
ann = read_csv("scenarios_annual_ebitda.csv")
fig = go.Figure()
fig.add_histogram(x=ann["status_quo"], name="Keep running as is", marker_color=SLATE, opacity=0.75, nbinsx=50)
fig.add_histogram(x=ann["recommendations"], name="With recommendations", marker_color=GOLD, opacity=0.75, nbinsx=50)
fig.update_layout(barmode="overlay", title="Distribution of next-12-month EBITDA (EUR)", height=380)
st.plotly_chart(fig, use_container_width=True)
explain("Both strategies are simulated on the **same random draws** of demand, ingredient prices and wage increases, so the gap between the two curves is the effect of the actions, not luck. "
        f"Only {pct(0.6, 0)} of the estimated benefit of the recommendations is assumed to materialise.")

mo = read_csv("scenarios_monthly.csv")
fig = go.Figure()
fig.add_scatter(x=mo["month"], y=mo["status_quo_p95"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
fig.add_scatter(x=mo["month"], y=mo["status_quo_p5"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(108,142,173,0.25)", name="Status quo: 5th to 95th percentile")
fig.add_scatter(x=mo["month"], y=mo["status_quo_p50"], mode="lines+markers", line={"color": SLATE}, name="Status quo, median")
fig.add_scatter(x=mo["month"], y=mo["recommendations_p50"], mode="lines+markers", line={"color": GOLD}, name="With recommendations, median")
fig.add_hline(y=0, line_color=WINE)
fig.update_layout(title="Monthly EBITDA, next 12 months (EUR)", height=380)
st.plotly_chart(fig, use_container_width=True)

st.subheader("What matters most? A 10% move in each driver")
se = read_csv("sensitivity.csv").sort_values("ebitda_change_up")
fig = go.Figure()
fig.add_bar(y=se["driver"], x=se["ebitda_change_up"], name="+10%", orientation="h", marker_color=GOLD)
fig.add_bar(y=se["driver"], x=se["ebitda_change_down"], name="-10%", orientation="h", marker_color=SLATE)
fig.update_layout(barmode="relative", height=340, xaxis_title="Change in annual EBITDA (EUR)")
st.plotly_chart(fig, use_container_width=True)
dr = sc["drivers"]
st.caption(f"Estimated from history: yearly demand growth {pct(dr['growth_yoy'])} (half of it is extrapolated), monthly demand noise {pct(dr['demand_noise_monthly'])}, "
           f"ingredient prices drift {pct(dr['cost_drift_monthly'])} per month with volatility {pct(dr['cost_vol_monthly'])}. Wages +4% (sd 1.5%) every January, rent +3%. "
           "Labor is treated as fixed within the year, which is why guests are the biggest lever.")
