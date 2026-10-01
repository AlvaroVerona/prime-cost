"""Look and feel shared by every page: palette, plotly template, KPI helpers and the page header."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

WINE, GOLD, SAGE, SLATE, CREAM, ROSE = "#B5495B", "#C9A227", "#7FB069", "#6C8EAD", "#EDE6DD", "#E8A0AA"
CLASS_COLORS = {"Star": GOLD, "Plowhorse": SLATE, "Puzzle": ROSE, "Dog": "#8A8078"}
CATEGORY_COLORS = {"Tapas": WINE, "Bebidas": SLATE, "Vino por copa": GOLD, "Vino botella": SAGE, "Postres": ROSE, "Cafe": "#8A8078"}


def register_template() -> None:
    t = go.layout.Template(pio.templates["plotly_dark"])
    t.layout.paper_bgcolor = "rgba(0,0,0,0)"
    t.layout.plot_bgcolor = "rgba(0,0,0,0)"
    t.layout.font = {"family": "sans-serif", "color": CREAM, "size": 13}
    t.layout.colorway = [WINE, GOLD, SLATE, SAGE, ROSE, "#8A8078"]
    t.layout.margin = {"l": 10, "r": 10, "t": 40, "b": 10}
    t.layout.xaxis.gridcolor = "rgba(237,230,221,0.08)"
    t.layout.yaxis.gridcolor = "rgba(237,230,221,0.08)"
    pio.templates["prime_cost"] = t
    pio.templates.default = "prime_cost"


def setup(title: str, subtitle: str = "") -> None:
    register_template()
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def eur(x: float, decimals: int = 0) -> str:
    return f"€{x:,.{decimals}f}"


def eur_short(x: float) -> str:
    """Compact money for KPI cards: 1,054,102 -> 1.05M, 143,344 -> 143k."""
    if abs(x) >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if abs(x) >= 10_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:,.0f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def kpis(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items, strict=True):
        col.metric(label, value, delta, delta_color="off")


def explain(text: str) -> None:
    st.markdown(f"<div style='opacity:.8;font-size:.95rem;margin:.2rem 0 1rem 0'>{text}</div>", unsafe_allow_html=True)


def need_outputs() -> None:
    from components.data import outputs_available

    if not outputs_available():
        st.error("No pipeline outputs found in reports/outputs/. Run `make all` first.")
        st.stop()
