"""Shared building blocks of the analytics: load the validated tables and build one fact row per ticket line."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prime_cost.config import PROCESSED_DIR, RAW_DIR, load_settings

HOUR_ORDER = list(range(13, 25))
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def load_tables(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Validated operational tables plus the (clean by construction) reference tables."""
    t = {n: pd.read_parquet(processed_dir / f"{n}_valid.parquet")
         for n in ["tickets", "ticket_lines", "purchases", "inventory_counts", "waste_log", "shifts"]}
    for n in ["calendar", "items", "ingredients", "recipes", "suppliers", "expenses", "cost_history", "price_history",
              "stockouts", "open_bottles", "employees"]:
        t[n] = pd.read_parquet(raw_dir / f"{n}.parquet")
    return t


def unit_cost_matrix(recipes: pd.DataFrame, cost_history: pd.DataFrame) -> pd.DataFrame:
    """Theoretical cost of ONE serving of each item on each day (recipe x agreed price that day). date x item_id."""
    cost = cost_history.pivot(index="date", columns="ingredient_id", values="agreed_price")
    r = recipes.pivot_table(index="ingredient_id", columns="item_id", values="qty_gross", aggfunc="sum", fill_value=0.0)
    cost = cost[r.index]
    return pd.DataFrame(cost.to_numpy() @ r.to_numpy(), index=cost.index, columns=r.columns)


def build_line_facts(t: dict[str, pd.DataFrame], settings: dict | None = None) -> pd.DataFrame:
    """One row per sold line with revenue, theoretical cost and the variable costs allocated to it."""
    s = settings or load_settings()
    lines = t["ticket_lines"]
    lines = lines[~lines["is_void"]].copy()
    tickets = t["tickets"].drop_duplicates("ticket_id")
    items = t["items"][["item_id", "name", "category"]]
    uc = unit_cost_matrix(t["recipes"], t["cost_history"]).stack().rename("unit_cost").reset_index()
    f = (lines.merge(tickets[["ticket_id", "hour", "daypart", "channel", "covers", "pay_method", "delivery_commission", "packaging_cost"]], on="ticket_id")
         .merge(items, on="item_id").merge(uc, on=["date", "item_id"], how="left"))
    f["net_revenue"] = f["qty"] * f["unit_price"] * (1 - f["discount_pct"])
    f["cost"] = f["qty"] * f["unit_cost"]
    # variable costs of the ticket spread over its lines in proportion to revenue
    tk_rev = f.groupby("ticket_id")["net_revenue"].transform("sum").replace(0, np.nan)
    share = (f["net_revenue"] / tk_rev).fillna(0.0)
    f["delivery_cost"] = (f["delivery_commission"] + f["packaging_cost"]) * share
    f["card_fee"] = np.where(f["pay_method"] == "card", f["net_revenue"] * s["payments"]["card_fee"], 0.0)
    f["contribution"] = f["net_revenue"] - f["cost"] - f["delivery_cost"] - f["card_fee"]
    f["weekday"] = f["date"].dt.weekday
    f["month"] = f["date"].dt.to_period("M").dt.to_timestamp()
    return f.drop(columns=["delivery_commission", "packaging_cost"])


def hourly_labor(shifts: pd.DataFrame) -> pd.DataFrame:
    """Labor cost and headcount per (date, hour, role), expanding each shift hour by hour."""
    rows = []
    for r in shifts.itertuples():
        for h in range(r.start_hour, r.end_hour):
            rows.append((r.date, h, r.role, r.hourly_cost))
    df = pd.DataFrame(rows, columns=["date", "hour", "role", "cost"])
    return df.groupby(["date", "hour", "role"], as_index=False).agg(cost=("cost", "sum"), heads=("cost", "size"))
