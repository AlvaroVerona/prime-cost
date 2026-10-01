"""Configuration loading: `config/settings.yaml` (parameters) and `config/menu.yaml` (catalog)."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "outputs"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    with (CONFIG_DIR / "settings.yaml").open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_menu() -> dict:
    with (CONFIG_DIR / "menu.yaml").open() as f:
        return yaml.safe_load(f)


def small_settings(days: int = 150, warmup_days: int = 21, seed: int = 7, holdout_days: int = 42) -> dict:
    """A copy of the settings with a short period, for fast tests."""
    s = copy.deepcopy(load_settings())
    s["seed"] = seed
    start = pd.Timestamp("2025-04-07")  # a Monday
    s["period"] = {"start": start.strftime("%Y-%m-%d"), "end": (start + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d"), "warmup_days": warmup_days}
    s["holdout_days"] = holdout_days
    s["pricing"] = {"rounds": []}
    s["costs"]["shocks"] = {}
    return s
