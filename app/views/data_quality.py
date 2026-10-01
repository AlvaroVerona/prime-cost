import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st
from components.data import read_json
from components.style import GOLD, WINE, explain, kpis, need_outputs, setup

setup("Data Quality", "Every table is validated before it is used. Critical and high-severity records go to quarantine with the rule that caught them; nothing is edited or silently dropped.")
need_outputs()
q = read_json("quality_report.json")
rows, quar = q["rows"], q["quarantined"]
kpis([
    ("Overall score", f"{q['score']['overall']:.1f} / 100", None),
    ("Records checked", f"{sum(rows.values()):,}", None),
    ("Quarantined", f"{sum(quar.values()):,}", f"{sum(quar.values()) / sum(rows.values()) * 100:.2f}%"),
    ("Validated tickets", f"{q['validated']['tickets']:,}", f"of {rows['tickets']:,} raw"),
])
st.divider()
c1, c2 = st.columns(2)
score = pd.DataFrame({"dataset": list(q["score"]["by_dataset"]), "score": list(q["score"]["by_dataset"].values())})
fig = px.bar(score, x="dataset", y="score", title="Quality score by table", color_discrete_sequence=[WINE])
fig.update_yaxes(range=[90, 100.5])
c1.plotly_chart(fig, use_container_width=True)
sev = pd.DataFrame(q["issues_by_severity"].items(), columns=["severity", "records"])
c2.plotly_chart(px.bar(sev, x="severity", y="records", title="Issues by severity", color_discrete_sequence=[GOLD]), use_container_width=True)
st.subheader("What was found")
rules = pd.DataFrame(q["issues_by_rule"]).sort_values("records", ascending=False)
st.dataframe(rules, use_container_width=True, hide_index=True)
explain("**CRITICAL** and **HIGH** records are quarantined (RAW minus QUARANTINED equals VALIDATED, checked in the tests). **MEDIUM** records, such as a "
        "supplier invoicing 3% above the agreed price, are kept and reported: they are findings for the business, not corrupt rows.")
st.subheader("Raw, quarantined and validated")
tbl = pd.DataFrame({"table": list(rows), "raw": list(rows.values()), "quarantined": [quar[k] for k in rows]})
tbl["share quarantined"] = (tbl["quarantined"] / tbl["raw"]).map("{:.2%}".format)
st.dataframe(tbl, use_container_width=True, hide_index=True)
