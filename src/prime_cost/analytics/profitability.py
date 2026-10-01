"""Profitability by product, category, channel, day and hour; unit economics; monthly P&L; break-even.

Follows the consulting method: never settle for global profitability, look at product, category, channel, daypart.
Run: `python -m prime_cost.analytics.profitability` (after the data quality step).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prime_cost.analytics.facts import HOUR_ORDER, WEEKDAY_NAMES, build_line_facts, hourly_labor, load_tables
from prime_cost.config import OUTPUT_DIR, load_settings


def by_group(f: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = f.groupby(keys).agg(units=("qty", "sum"), revenue=("net_revenue", "sum"), cost=("cost", "sum"),
                            delivery_cost=("delivery_cost", "sum"), card_fee=("card_fee", "sum"), contribution=("contribution", "sum")).reset_index()
    g["food_cost_pct"] = g["cost"] / g["revenue"]
    g["contribution_pct"] = g["contribution"] / g["revenue"]
    return g


def hourly_profile(f: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Average revenue, contribution and labor cost for each weekday-hour of the trading week."""
    labor = hourly_labor(shifts)
    labor = labor[labor["hour"].isin(HOUR_ORDER)]  # hours before opening (prep and set-up) are reported separately
    labor = labor.groupby(["date", "hour"], as_index=False).agg(labor_cost=("cost", "sum"), staff=("heads", "sum"))
    rev = f.groupby(["date", "hour"], as_index=False).agg(revenue=("net_revenue", "sum"), contribution=("contribution", "sum"))
    cov = f.drop_duplicates("ticket_id").groupby(["date", "hour"], as_index=False)["covers"].sum()
    d = labor.merge(rev, on=["date", "hour"], how="left").merge(cov, on=["date", "hour"], how="left").fillna(0.0)
    d["weekday"] = d["date"].dt.weekday
    g = d.groupby(["weekday", "hour"], as_index=False).agg(
        revenue=("revenue", "mean"), contribution=("contribution", "mean"), labor_cost=("labor_cost", "mean"), staff=("staff", "mean"), covers=("covers", "mean"))
    g["contribution_after_labor"] = g["contribution"] - g["labor_cost"]
    g["sales_per_labor_hour"] = g["revenue"] / g["staff"].replace(0, np.nan)
    g["weekday_name"] = g["weekday"].map(WEEKDAY_NAMES)
    return g


def monthly_pnl(f: pd.DataFrame, t: dict[str, pd.DataFrame], cogs_actual_month: pd.Series | None = None) -> pd.DataFrame:
    m = f.groupby("month").agg(revenue=("net_revenue", "sum"), cogs_theoretical=("cost", "sum"), delivery_cost=("delivery_cost", "sum"), card_fee=("card_fee", "sum"))
    shifts = t["shifts"].assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
    m["labor"] = shifts.groupby("month")["cost"].sum()
    m["fixed_expenses"] = t["expenses"].groupby("month")["amount"].sum()
    if cogs_actual_month is not None:
        m["cogs_actual"] = cogs_actual_month
    else:
        m["cogs_actual"] = m["cogs_theoretical"]
    m = m.dropna(subset=["revenue"])
    m["ebitda"] = m["revenue"] - m["cogs_actual"] - m["labor"] - m["fixed_expenses"] - m["delivery_cost"] - m["card_fee"]
    m["prime_cost_pct"] = (m["cogs_actual"] + m["labor"]) / m["revenue"]
    m["ebitda_pct"] = m["ebitda"] / m["revenue"]
    return m.reset_index()


def break_even(pnl: pd.DataFrame, last_months: int = 12) -> dict:
    """Break-even revenue = fixed costs / contribution margin %. Labor is fixed here: the template never changes."""
    p = pnl.tail(last_months)
    rev = p["revenue"].sum()
    variable = (p["cogs_actual"] + p["delivery_cost"] + p["card_fee"]).sum()
    cm_pct = 1 - variable / rev
    fixed = (p["labor"] + p["fixed_expenses"]).sum() / len(p)
    be = fixed / cm_pct
    return {"months": int(len(p)), "monthly_fixed_costs": round(float(fixed), 2), "contribution_margin_pct": round(float(cm_pct), 4),
            "break_even_monthly_revenue": round(float(be), 2), "avg_monthly_revenue": round(float(rev / len(p)), 2),
            "margin_of_safety_pct": round(float(1 - be / (rev / len(p))), 4)}


def unit_economics(f: pd.DataFrame) -> dict:
    tk = f.groupby(["ticket_id", "channel"]).agg(revenue=("net_revenue", "sum"), cost=("cost", "sum"), contribution=("contribution", "sum"), covers=("covers", "first")).reset_index()
    out = {}
    for ch, g in tk.groupby("channel"):
        out[ch] = {"tickets": int(len(g)), "avg_ticket": round(float(g["revenue"].mean()), 2), "avg_product_cost": round(float(g["cost"].mean()), 2),
                   "avg_contribution": round(float(g["contribution"].mean()), 2), "contribution_pct": round(float(g["contribution"].sum() / g["revenue"].sum()), 4),
                   "revenue_per_cover": round(float(g["revenue"].sum() / g["covers"].sum()), 2)}
    return out


def run(tables=None, output_dir=OUTPUT_DIR, cogs_actual_month=None) -> dict:
    t = tables or load_tables()
    f = build_line_facts(t, load_settings())
    output_dir.mkdir(parents=True, exist_ok=True)
    item = by_group(f, ["item_id", "name", "category"])
    cat = by_group(f, ["category"])
    chan = by_group(f, ["channel"])
    dp = by_group(f, ["daypart"])
    wd = by_group(f.assign(weekday_name=f["weekday"].map(WEEKDAY_NAMES)), ["weekday", "weekday_name"])
    hp = hourly_profile(f, t["shifts"])
    pnl = monthly_pnl(f, t, cogs_actual_month)
    be = break_even(pnl)
    ue = unit_economics(f)
    for name, df in {"profit_by_item": item, "profit_by_category": cat, "profit_by_channel": chan, "profit_by_daypart": dp,
                     "profit_by_weekday": wd, "hourly_profile": hp, "monthly_pnl": pnl}.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    summary = {"revenue": round(float(f["net_revenue"].sum()), 2), "contribution": round(float(f["contribution"].sum()), 2),
               "theoretical_food_cost_pct": round(float(f["cost"].sum() / f["net_revenue"].sum()), 4), "break_even": be, "unit_economics": ue}
    (output_dir / "profitability_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"revenue {summary['revenue']:,.0f}, theoretical food+bev cost {summary['theoretical_food_cost_pct']:.1%}, break-even {be['break_even_monthly_revenue']:,.0f}/month")
    return {"facts": f, "item": item, "category": cat, "channel": chan, "hourly": hp, "pnl": pnl, "break_even": be, "unit_economics": ue}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
