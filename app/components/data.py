"""Cached readers for the pipeline outputs in reports/outputs/ (the dashboard computes nothing new)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from prime_cost.config import OUTPUT_DIR


def outputs_available() -> bool:
    return (OUTPUT_DIR / "headline.json").exists()


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text())


@st.cache_data
def read_csv(name: str, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, parse_dates=list(dates))
