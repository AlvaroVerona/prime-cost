import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, SLATE, WINE, eur, eur_short, explain, kpis, need_outputs, pct, setup

setup("Food Cost and Waste", "Theoretical cost (recipes x sales) versus actual cost (opening stock + purchases - closing stock, from the Monday counts). The gap is money that left without being sold.")
need_outputs()
s = read_json("food_cost_summary.json")
kpis([
    ("Theoretical cost", eur_short(s["theoretical_cost"]), None),
    ("Actual cost", eur_short(s["actual_cost"]), f"+{pct(s['variance_pct_of_theoretical'])}"),
    ("Logged waste", eur_short(s["logged_waste_cost"]), "what the team wrote down"),
    ("Unexplained", eur_short(s["unexplained_cost"]), "not logged anywhere"),
])
explain(f"Over 24 months the bar used {eur(s['variance_cost'])} more product than its recipes and sales justify ({pct(s['variance_pct_of_theoretical'])} above theoretical). "
        f"The team logged {pct(s['logged_waste_cost'] / s['variance_cost'], 0)} of that gap; the rest is unlogged waste, over-portioning, over-pouring and counting noise.")
st.divider()

wk = read_csv("food_cost_weekly.csv", ("period_end",))
fig = go.Figure()
fig.add_scatter(x=wk["period_end"], y=wk["variance_pct"], mode="lines", name="Actual vs theoretical", line={"color": WINE, "width": 2})
fig.add_hline(y=0.05, line_dash="dot", line_color=GOLD, annotation_text="a common target: within 5%")
fig.update_layout(title="Weekly gap between actual and theoretical cost", yaxis_tickformat=".0%", height=340)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
ing = read_csv("food_cost_by_ingredient.csv")
top = ing.sort_values("variance_cost", ascending=False).head(12)
fig = go.Figure()
fig.add_bar(y=top["name"], x=top["logged_waste_cost"], name="Logged waste", orientation="h", marker_color=SLATE)
fig.add_bar(y=top["name"], x=top["unexplained_cost"], name="Unexplained", orientation="h", marker_color=WINE)
fig.update_layout(barmode="stack", title="Where the gap is (top 12 ingredients, EUR)", height=430, yaxis={"autorange": "reversed"}, legend={"orientation": "h", "y": 1.1})
c1.plotly_chart(fig, use_container_width=True)
reason = read_csv("waste_by_reason.csv")
c2.plotly_chart(px.bar(reason, x="cost", y="reason", orientation="h", title="Logged waste by reason (EUR)", color_discrete_sequence=[GOLD]).update_layout(height=430, yaxis={"autorange": "reversed"}), use_container_width=True)
explain("Most logged waste is **expiry**: products bought for days they could not be sold before going off. That is a purchasing problem, and it is what the Purchasing page addresses.")

st.subheader("Ingredient detail")
st.dataframe(ing[["name", "kind", "theoretical_cost", "actual_cost", "logged_waste_cost", "unexplained_cost", "variance_pct_of_theoretical"]].round(2),
             use_container_width=True, hide_index=True, column_config={"variance_pct_of_theoretical": st.column_config.NumberColumn("gap vs theoretical", format="percent")})
st.caption("Valuation uses the agreed supplier price of each week. Open wine bottles are not part of the Monday count, which adds a little noise to the wine lines.")
