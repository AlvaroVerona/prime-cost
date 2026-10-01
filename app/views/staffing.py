import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, SLATE, WINE, eur_short, explain, kpis, need_outputs, pct, setup

setup("Staffing", "A weekly schedule built from the demand forecast with a CP-SAT model, judged on the guests that really came, against the fixed template the bar uses today.")
need_outputs()
s = read_json("staffing_summary.json")
cur, opt = s["current"], s["optimized"]
kpis([
    ("Labor cost, cook + floor + bar", f"{eur_short(opt['cost'])}", f"{eur_short(cur['cost'])} today ({s['days_evaluated']} days)"),
    ("Saving", pct(s["labor_saving_pct"]), f"{eur_short(s['labor_saving_annualized_eur'])} per year"),
    ("Hours with too few people", f"{opt['hours_with_a_gap']}", f"{cur['hours_with_a_gap']} today"),
    ("Hours with too many", f"{opt['overstaffed_head_hours']:.0f}", f"{cur['overstaffed_head_hours']:.0f} today"),
])
explain("It is not only cheaper: it also leaves fewer hours understaffed, because it moves people from empty hours (Tuesday and Wednesday afternoons, Sunday evenings) to the peaks "
        "(Friday and Saturday nights) that today's identical-every-week template under-serves.")
st.divider()
cmp = read_csv("staffing_comparison.csv", ("date",))
c1, c2 = st.columns(2)
role = c1.selectbox("Role", ["sala", "cocina", "barra"])
wd = c2.selectbox("Weekday", ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=3)
names = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
d = cmp[(cmp["role"] == role) & (cmp["weekday"] == names[wd])].groupby("hour", as_index=False)[["required", "current", "optimized"]].mean()
fig = go.Figure()
fig.add_bar(x=d["hour"], y=d["current"], name="Today's template", marker_color=SLATE)
fig.add_bar(x=d["hour"], y=d["optimized"], name="Optimized", marker_color=GOLD)
fig.add_scatter(x=d["hour"], y=d["required"], name="Needed (actual guests)", mode="lines+markers", line={"color": WINE, "width": 3})
fig.update_layout(barmode="group", height=400, title=f"{role.capitalize()} on {wd}: average people per hour", xaxis_title="Hour of day (24 = after midnight)", yaxis_title="People")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Example week plan")
plan = read_csv("staffing_plan.csv", ("date",))
week = st.selectbox("Week starting", sorted((plan["date"] - plan["date"].dt.weekday * pd.Timedelta(days=1)).dt.date.unique()))
w = plan[(plan["date"].dt.date >= week) & (plan["date"].dt.date < week + pd.Timedelta(days=7))].sort_values(["date", "role", "start_hour"])
w = w.assign(day=w["date"].dt.strftime("%a %d %b"), shift=w["start_hour"].astype(str) + "h to " + w["end_hour"].astype(str) + "h")
st.dataframe(w[["day", "role", "shift", "n"]].rename(columns={"n": "people"}), use_container_width=True, hide_index=True)
st.caption("Assumptions: one person serves 14 guests (floor), 28 (kitchen) or 24 (bar) per hour; guests stay about 1.5 hours; shifts of 4 to 9 hours; at most 5 shifts and 40 hours per person per week. "
           "The model plans for the forecast plus half the gap to its upper bound. The need model is an assumption; the data do not link staffing to sales.")
