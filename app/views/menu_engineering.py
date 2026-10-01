import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import CLASS_COLORS, GOLD, eur, explain, need_outputs, setup

setup("Menu Engineering", "Each dish by popularity and margin (Kasavana and Smith), within its own family. Then what a price rise would do, tested against how customers might react.")
need_outputs()
m = read_csv("menu_engineering.csv")
summ = read_json("menu_engineering_summary.json")

fam = st.selectbox("Family", sorted(m["family"].unique()), index=sorted(m["family"].unique()).index("Tapas"))
d = m[m["family"] == fam].copy()
d["mix_pct_view"] = d["mix_pct"] * 100
fig = px.scatter(d, x="mix_pct_view", y="unit_margin", color="class", size="units", hover_name="name", color_discrete_map=CLASS_COLORS,
                 labels={"mix_pct_view": "Popularity: share of units sold (%)", "unit_margin": "Margin per unit (EUR)"}, height=480)
fig.add_hline(y=float(d["avg_unit_margin_family"].iloc[0]), line_dash="dot", line_color="#8A8078")
fig.add_vline(x=0.7 / len(d) * 100, line_dash="dot", line_color="#8A8078")
for r in d.itertuples():
    if r.units >= d["units"].quantile(0.8) or r.unit_margin >= d["unit_margin"].quantile(0.9):
        fig.add_annotation(x=r.mix_pct_view, y=r.unit_margin, text=r.name.split(" (")[0][:22], showarrow=False, yshift=14, font={"size": 10})
st.plotly_chart(fig, use_container_width=True)
explain("Right of the vertical line = popular (at least 70% of an equal share). Above the horizontal line = margin per unit above the family average, weighted by sales.")

counts = d["class"].value_counts().reindex(["Star", "Plowhorse", "Puzzle", "Dog"]).fillna(0).astype(int)
cols = st.columns(4)
actions = {"Star": "Protect and highlight", "Plowhorse": "Raise price a little, trim cost", "Puzzle": "Reposition and recommend", "Dog": "Redesign or drop"}
for col, (k, v) in zip(cols, counts.items(), strict=True):
    col.metric(k, int(v), actions[k])
st.dataframe(d.sort_values("total_margin", ascending=False)[["name", "class", "units", "unit_price", "food_cost_pct", "unit_margin", "total_margin", "action"]].round(2),
             use_container_width=True, hide_index=True, column_config={"food_cost_pct": st.column_config.NumberColumn("food cost %", format="percent")})

st.divider()
st.subheader("How much do customers react to price?")
el = read_csv("elasticity.csv")
fig = go.Figure()
fig.add_scatter(x=el["elasticity"], y=el["category"], mode="markers", marker={"size": 12, "color": GOLD}, name="Estimate",
                error_x={"type": "data", "symmetric": False, "array": el["ci90_high"] - el["elasticity"], "arrayminus": el["elasticity"] - el["ci90_low"]})
fig.add_scatter(x=el["elasticity_shrunk"], y=el["category"], mode="markers", marker={"size": 10, "symbol": "diamond", "color": "#EDE6DD"}, name="Used (shrunk to a prior)")
fig.add_vline(x=0, line_color="#8A8078")
fig.update_layout(title="Price elasticity from the two real price rounds, with 90% intervals", height=320, xaxis_title="Elasticity (change in units per % change in price)")
st.plotly_chart(fig, use_container_width=True)
explain("A 4-5% price move observed over a few weeks identifies elasticity only roughly: the intervals are wide and several include zero. "
        "So the point estimates are **shrunk toward a conservative prior**, and every price recommendation below is also tested against a pessimistic elasticity.")

st.subheader(f"What a +{summ['price_rise_tested'] * 100:.0f}% price rise would do (annual contribution)")
sc = read_csv("price_scenarios.csv")
robust = sc[sc["robust"]].sort_values("gain_central", ascending=False).head(15)
st.dataframe(robust[["name", "class", "unit_price", "new_price", "units_12m", "gain_pessimistic", "gain_central", "gain_optimistic"]].round(0),
             use_container_width=True, hide_index=True)
st.caption(f"{int(sc['robust'].sum())} of {len(sc)} items still gain under the pessimistic elasticity (-1.2). Together {eur(summ['robust_price_gain_total'])} per year at the central estimate. "
           "Cross-effects between dishes are ignored: a real test would raise a few prices first and measure.")
