"""Prime Cost: overview."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, WINE, eur, eur_short, kpis, need_outputs, pct, setup

setup("Overview", "La Cepa, a synthetic natural-wine and tapas bar in Madrid (Oct 2024 to Sep 2026). Where the money is made, where it leaks, and what to do.")
need_outputs()
h = read_json("headline.json")

kpis([
    ("Revenue, 12 months", eur_short(h["last12_revenue"]), None),
    ("EBITDA margin", pct(h["last12_ebitda_pct"]), eur_short(h["last12_ebitda"])),
    ("Prime cost", pct(h["last12_prime_cost_pct"]), "product + labor"),
    ("Stock cost gap", pct(h["actual_vs_theoretical_gap_pct"]), "actual vs recipes"),
    ("Data quality", f"{h['quality_score']:.1f}", "score out of 100"),
])
st.divider()

st.subheader("Three levers, measured out of sample")
c1, c2, c3 = st.columns(3)
c1.metric("Staff schedule built from the forecast", eur(h["staffing_saving_annualized"]) + " / year", f"{h['staffing_saving_pct'] * 100:.0f}% of the roles optimized")
c1.caption(f"Hours with a staffing gap: {h['staffing_gap_hours'][0]} to {h['staffing_gap_hours'][1]} over 12 weeks.")
c2.metric("Purchasing against the forecast", eur(h["purchasing_saving_annualized"]) + " / year", "waste plus lost sales")
c2.caption(f"Waste {eur(h['purchasing_waste_12w'][0])} to {eur(h['purchasing_waste_12w'][1])} and lost margin {eur(h['purchasing_lost_margin_12w'][0])} to {eur(h['purchasing_lost_margin_12w'][1])} in 12 weeks.")
c3.metric("Tested price rise on robust dishes", eur(read_json("menu_engineering_summary.json")["robust_price_gain_total"]) + " / year", "+5%, robust to elasticity")
c3.caption("Only dishes that still gain even under a pessimistic demand reaction.")
st.info(
    f"Applying the recommendations (with only 60% of the estimated benefit assumed) moves expected EBITDA over the next 12 months from "
    f"**{eur(h['scenario_status_quo_ebitda'])}** to **{eur(h['scenario_recommendations_ebitda'])}**. "
    f"Break-even is **{eur(h['break_even_monthly'])}** of monthly revenue, and in every simulated year at least one month is loss-making."
)

st.subheader("Monthly revenue and EBITDA")
pnl = read_csv("monthly_pnl.csv", ("month",))
fig = go.Figure()
fig.add_bar(x=pnl["month"], y=pnl["revenue"], name="Revenue", marker_color=WINE, opacity=0.75)
fig.add_scatter(x=pnl["month"], y=pnl["ebitda"], name="EBITDA", mode="lines+markers", line={"color": GOLD, "width": 3}, yaxis="y2")
fig.update_layout(height=380, yaxis={"title": "Revenue (EUR)"}, yaxis2={"title": "EBITDA (EUR)", "overlaying": "y", "side": "right", "showgrid": False})
st.plotly_chart(fig, use_container_width=True)
st.caption("August is the weak month in Madrid: demand falls about a third while rent and a fixed staff template do not.")

st.subheader("What is in this app")
st.markdown(
    "- **Data Quality**: validation with quarantine and lineage.\n"
    "- **Profitability**: by dish, channel, day and hour; unit economics; break-even.\n"
    "- **Food Cost and Waste**: recipes and sales versus what stock counts say was really used.\n"
    "- **Menu Engineering**: stars, plowhorses, puzzles and dogs; price elasticity; price scenarios.\n"
    "- **Demand Forecast**: guests per day, 7 days ahead, evaluated on weeks never used for fitting.\n"
    "- **Staffing** and **Purchasing**: the two optimizations, replayed against what really happened.\n"
    "- **Scenarios**: Monte Carlo of the next 12 months and which driver matters most."
)
st.caption(f"All data is synthetic and seeded. Forecast: {h['forecast_best_model'].upper()} with {pct(h['forecast_wape'])} error versus {pct(h['forecast_baseline_wape'])} for the naive weekday average. That is {pct(h['forecast_improvement_pct'], 0)} less error.")
