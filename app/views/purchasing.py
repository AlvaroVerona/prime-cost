import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st
from components.data import read_csv, read_json
from components.style import GOLD, SLATE, eur, eur_short, explain, kpis, need_outputs, setup

setup("Purchasing", "The last 12 weeks replayed with the same inventory simulator and the same guests, once with today's ordering rule and once ordering against the forecast.")
need_outputs()
s = read_json("purchasing_summary.json")
b, f = s["baseline"], s["forecast"]
kpis([
    ("Waste + lost sales margin", eur_short(f["leak"]), f"{eur_short(b['leak'])} with today's rule"),
    ("Saving in 12 weeks", eur_short(s["saving_mean"]), f"± {eur_short(s['saving_std'])} across {s['seeds']} random seeds"),
    ("Annualized", eur_short(s["saving_annualized"]), None),
    ("Seeds where it is better", f"{s['saving_share_of_seeds_positive'] * 100:.0f}%", f"worst case {eur_short(s['saving_min'])}"),
])
st.divider()
runs = read_csv("purchasing_runs.csv")
c1, c2 = st.columns(2)
fig = go.Figure()
for pol, color, label in [("baseline", SLATE, "Today's rule"), ("forecast", GOLD, "Forecast-based")]:
    d = runs[runs["policy"] == pol]
    fig.add_bar(x=["Waste (thrown away)", "Lost sales margin"], y=[d["waste_cost"].mean(), d["lost_margin"].mean()], name=label, marker_color=color,
                error_y={"type": "data", "array": [d["waste_cost"].std(), d["lost_margin"].std()]})
fig.update_layout(barmode="group", title="Cost of getting the order wrong, 12 weeks (EUR)", height=380, legend={"orientation": "h", "y": 1.12})
c1.plotly_chart(fig, use_container_width=True)
ex = read_csv("purchasing_expiry_by_ingredient.csv")
ex["gain"] = ex["baseline"] - ex["forecast"]
top = ex.sort_values("baseline", ascending=False).head(8)
fig = go.Figure()
fig.add_bar(y=top["ingredient_id"], x=top["baseline"], name="Today's rule", orientation="h", marker_color=SLATE)
fig.add_bar(y=top["ingredient_id"], x=top["forecast"], name="Forecast-based", orientation="h", marker_color=GOLD)
fig.update_layout(barmode="group", title="Expired stock by ingredient, 12 weeks (EUR)", height=380, yaxis={"autorange": "reversed"}, legend={"orientation": "h", "y": 1.12})
c2.plotly_chart(fig, use_container_width=True)

st.subheader("How the forecast policy decides how much to order")
st.markdown(
    "- **Expected use** of each ingredient over the days the next delivery must cover = forecast guests x ingredient used per guest on that weekday.\n"
    "- **Capped at the shelf life**: stock received today is gone after its shelf life, so ordering for later days only creates waste.\n"
    "- **Safety stock** = z x forecast error, where z comes from the newsvendor ratio *cost of running out / (cost of running out + cost of having too much)*. "
    "A dry good that cannot spoil gets a big buffer; fresh fish gets a small one.\n"
    "- Today's rule is the average use of the last 28 days times (cover days + safety days), which ignores that Friday is not Tuesday."
)
explain(f"Waste falls from {eur(b['waste_cost'])} to {eur(f['waste_cost'])} and the margin lost to stock-outs from {eur(b['lost_margin'])} to {eur(f['lost_margin'])}. "
        "The lost margin is an upper bound: it assumes a guest who finds a dish sold out orders nothing else.")
st.caption(f"The replay is repeated with {s['seeds']} random seeds because the simulator has chance events (prep errors, invoice mistakes). Today's rule is a deliberately simple rule of thumb; "
           "a careful manager who orders by weekday would already capture part of this gain.")
