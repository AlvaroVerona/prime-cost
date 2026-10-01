"""Prime Cost: entry point. A router with clean page names; each page lives in app/views/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from components.style import register_template

st.set_page_config(page_title="Prime Cost", page_icon="🍷", layout="wide")
register_template()

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/data_quality.py", title="Data Quality", url_path="data-quality"),
    st.Page("views/profitability.py", title="Profitability", url_path="profitability"),
    st.Page("views/food_cost_and_waste.py", title="Food Cost and Waste", url_path="food-cost"),
    st.Page("views/menu_engineering.py", title="Menu Engineering", url_path="menu-engineering"),
    st.Page("views/demand_forecast.py", title="Demand Forecast", url_path="forecast"),
    st.Page("views/staffing.py", title="Staffing", url_path="staffing"),
    st.Page("views/purchasing.py", title="Purchasing", url_path="purchasing"),
    st.Page("views/scenarios.py", title="Scenarios", url_path="scenarios"),
]
st.navigation(pages).run()
