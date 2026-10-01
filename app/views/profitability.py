import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import CATEGORY_COLORS, eur, eur_short, explain, kpis, need_outputs, pct, setup

setup("Profitability", "Never settle for global profitability: by product, channel, day and hour. Contribution = revenue minus recipe cost, delivery commissions and card fees.")
need_outputs()
summ = read_json("profitability_summary.json")
be = summ["break_even"]
kpis([
    ("Revenue (24 months)", eur_short(summ["revenue"]), None),
    ("Theoretical product cost", pct(summ["theoretical_food_cost_pct"]), "from recipes and sales"),
    ("Break-even revenue / month", eur_short(be["break_even_monthly_revenue"]), f"safety margin {pct(be['margin_of_safety_pct'])}"),
    ("Contribution margin", pct(be["contribution_margin_pct"]), None),
])
st.divider()

cat = read_csv("profit_by_category.csv").sort_values("revenue", ascending=False)
c1, c2 = st.columns(2)
c1.plotly_chart(px.bar(cat, x="category", y="revenue", color="category", color_discrete_map=CATEGORY_COLORS, title="Revenue by category").update_layout(showlegend=False), use_container_width=True)
c2.plotly_chart(px.bar(cat, x="category", y="contribution_pct", color="category", color_discrete_map=CATEGORY_COLORS, title="Contribution margin % by category").update_layout(showlegend=False).update_yaxes(tickformat=".0%"), use_container_width=True)

st.subheader("Unit economics by channel")
ue = pd.DataFrame(summ["unit_economics"]).T.reset_index(names="channel")
fig = px.bar(ue, x="channel", y=["avg_ticket", "avg_contribution"], barmode="group", title="Average ticket and contribution per ticket (EUR)")
st.plotly_chart(fig, use_container_width=True)
dl, si = summ["unit_economics"]["delivery"], summ["unit_economics"]["sala"]
explain(f"A delivery order brings in {eur(dl['avg_ticket'], 2)} and leaves {eur(dl['avg_contribution'], 2)} ({pct(dl['contribution_pct'])}) after the platform commission and packaging. "
        f"A table in the dining room brings {eur(si['avg_ticket'], 2)} and leaves {eur(si['avg_contribution'], 2)} ({pct(si['contribution_pct'])}). "
        f"One dining-room ticket is worth about {si['avg_contribution'] / dl['avg_contribution']:.1f} delivery orders.")

st.subheader("Which hours make money after paying the staff?")
hp = read_csv("hourly_profile.csv")
hp = hp[hp["hour"] <= 24]
pivot = hp.pivot(index="weekday_name", columns="hour", values="contribution_after_labor").reindex(["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=[f"{h}h" for h in pivot.columns], y=pivot.index, colorscale="RdYlGn", zmid=0, colorbar={"title": "EUR / hour"}))
fig.update_layout(height=340, title="Average contribution after labor, by weekday and hour")
st.plotly_chart(fig, use_container_width=True)
worst = hp.sort_values("contribution_after_labor").head(3)
explain("Red cells are hours where the contribution does not even cover the people on shift, with the fixed weekly template the bar uses today. "
        "Worst: " + ", ".join(f"{r.weekday_name} {int(r.hour)}h ({eur(r.contribution_after_labor)} per hour)" for r in worst.itertuples()) + ".")

st.subheader("By dish")
cats = ["All"] + sorted(read_csv("profit_by_item.csv")["category"].unique())
choice = st.selectbox("Category", cats)
items = read_csv("profit_by_item.csv")
if choice != "All":
    items = items[items["category"] == choice]
items = items.sort_values("contribution", ascending=False)
show = items[["name", "category", "units", "revenue", "food_cost_pct", "contribution", "contribution_pct"]].round(3)
st.dataframe(show, use_container_width=True, hide_index=True, column_config={
    "food_cost_pct": st.column_config.NumberColumn("food cost %", format="percent"), "contribution_pct": st.column_config.NumberColumn("contribution %", format="percent"),
    "revenue": st.column_config.NumberColumn(format="€%.0f"), "contribution": st.column_config.NumberColumn(format="€%.0f")})
