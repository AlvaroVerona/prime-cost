import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, SLATE, WINE, explain, kpis, need_outputs, pct, setup

setup("Demand Forecast", "Guests per day for the next 7 days, evaluated out of time: the models never saw the weeks shown here, and the weather inputs are noisy like a real forecast.")
need_outputs()
mt = read_json("forecast_metrics.json")
best = mt["best_model"]
kpis([
    ("Best model", best.upper(), None),
    ("Error (WAPE)", pct(mt["models"][best]["wape"]), f"naive weekday average: {pct(mt['models']['baseline_dow4']['wape'])}"),
    ("Improvement", pct(mt["improvement_vs_baseline_pct"], 0), "less error than the baseline"),
    ("80% interval coverage", pct(mt["interval_coverage_out_of_sample"], 0), "measured on weeks not used to calibrate it"),
])
st.divider()
ev = read_csv("forecast_eval.csv", ("date", "origin"))
h = st.slider("Days ahead", 1, 7, 1)
d = ev[ev["h"] == h].sort_values("date")
fig = go.Figure()
fig.add_scatter(x=d["date"], y=d["upper"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
fig.add_scatter(x=d["date"], y=d["lower"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(201,162,39,0.18)", name="80% interval")
fig.add_scatter(x=d["date"], y=d["forecast"], mode="lines", name="Forecast", line={"color": GOLD, "width": 2})
fig.add_scatter(x=d["date"], y=d["covers"], mode="markers", name="Actual guests", marker={"color": WINE, "size": 7})
fig.update_layout(title=f"Guests per day, forecast {h} day(s) ahead", height=420, legend={"orientation": "h", "y": 1.12})
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
models = [(k, v["wape"]) for k, v in mt["models"].items()]
names = {"baseline_dow4": "Weekday average (naive)", "level_profile": "Level x weekday profile", "ridge": "Ridge regression", "gbm": "Gradient boosting"}
fig = px.bar(x=[names[k] for k, _ in models], y=[v for _, v in models], title="Error by model (WAPE, lower is better)", color_discrete_sequence=[SLATE])
fig.update_yaxes(tickformat=".0%")
c1.plotly_chart(fig, use_container_width=True)
rows = [(names[k], int(hh), w) for k, v in mt["wape_by_horizon"].items() for hh, w in v.items()]
hz = pd.DataFrame(rows, columns=["model", "days ahead", "wape"])
c2.plotly_chart(px.line(hz, x="days ahead", y="wape", color="model", markers=True, title="Error by forecast horizon").update_yaxes(tickformat=".0%"), use_container_width=True)
explain("The gradient-boosting model captures what a weekday average cannot: weather and its interaction with the season (terrace weather), holidays and the first-Wednesday tasting nights. "
        "A linear model with the same inputs does not beat the naive baseline, so the gain is not simply from having more variables.")
st.caption("Limitation: the data are synthetic, with 11% daily noise built in, so roughly 9% error is the floor nobody could beat. Real bars have more irregular demand.")
