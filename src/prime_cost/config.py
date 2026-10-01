"""Configuration loading: `config/settings.yaml` (parameters) and `config/menu.yaml` (catalog)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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


def small_settings(days: int = 70, warmup_days: int = 14, seed: int = 7) -> dict:
    """A copy of the settings with a short period, for fast tests."""
    import copy

    s = copy.deepcopy(load_settings())
    s["seed"] = seed
    s["period"] = {"start": "2025-04-01", "end": (__import__("pandas").Timestamp("2025-04-01") + __import__("pandas").Timedelta(days=days - 1)).strftime("%Y-%m-%d"), "warmup_days": warmup_days}
    s["holdout_days"] = 28
    return s
