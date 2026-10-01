"""Food and beverage cost: theoretical (recipes x sales) versus actual (opening stock + purchases - closing stock).

The gap is the money that leaves the business without being sold: logged waste, unlogged waste, over-portioning,
over-pouring, theft and counting error. The method follows the consulting manual: a persistent gap must be investigated.
Run: `python -m prime_cost.analytics.food_cost`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prime_cost.analytics.facts import load_tables
from prime_cost.config import OUTPUT_DIR


def count_periods(counts: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(counts["count_date"].unique())
    return list(zip(dates[:-1], dates[1:], strict=True))


def reconcile(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (period, ingredient): theoretical use, actual use, logged waste and the unexplained remainder."""
    counts = t["inventory_counts"].pivot(index="count_date", columns="ingredient_id", values="counted_qty")
    lines = t["ticket_lines"]
    lines = lines[~lines["is_void"]]
    recipes = t["recipes"][["item_id", "ingredient_id", "qty_gross"]]
    use = lines.merge(recipes, on="item_id").assign(qty_theory=lambda d: d["qty"] * d["qty_gross"])
    use = use.groupby(["date", "ingredient_id"], as_index=False)["qty_theory"].sum()
    purchases = t["purchases"].groupby(["received_on", "ingredient_id"], as_index=False)["qty_received"].sum()
    waste = t["waste_log"].groupby(["date", "ingredient_id"], as_index=False).agg(qty_waste=("qty", "sum"), cost_waste=("cost", "sum"))
    price = t["cost_history"]
    rows = []
    for start, end in count_periods(t["inventory_counts"]):
        in_p = lambda s, a=start, b=end: (s > a) & (s <= b)  # noqa: E731
        u = use[in_p(use["date"])].groupby("ingredient_id")["qty_theory"].sum()
        p = purchases[in_p(purchases["received_on"])].groupby("ingredient_id")["qty_received"].sum()
        w = waste[in_p(waste["date"])].groupby("ingredient_id")["qty_waste"].sum()
        px = price[in_p(price["date"])].groupby("ingredient_id")["agreed_price"].mean()
        df = pd.DataFrame({"opening": counts.loc[start], "closing": counts.loc[end]})
        df["purchases"] = p
        df["theoretical"] = u
        df["logged_waste"] = w
        df["price"] = px
        df = df.fillna({"purchases": 0.0, "theoretical": 0.0, "logged_waste": 0.0}).reset_index(names="ingredient_id")
        df["period_start"], df["period_end"] = start, end
        rows.append(df)
    r = pd.concat(rows, ignore_index=True)
    r["actual"] = r["opening"] + r["purchases"] - r["closing"]
    r["variance"] = r["actual"] - r["theoretical"]
    r["unexplained"] = r["variance"] - r["logged_waste"]
    for c in ["theoretical", "actual", "logged_waste", "variance", "unexplained"]:
        r[f"{c}_cost"] = r[c] * r["price"]
    return r


def summarize(r: pd.DataFrame, ingredients: pd.DataFrame) -> dict:
    meta = ingredients.set_index("ingredient_id")[["name", "kind"]]
    by_ing = r.groupby("ingredient_id", as_index=False)[["theoretical_cost", "actual_cost", "logged_waste_cost", "variance_cost", "unexplained_cost"]].sum()
    by_ing = by_ing.merge(meta, left_on="ingredient_id", right_index=True)
    by_ing["variance_pct_of_theoretical"] = by_ing["variance_cost"] / by_ing["theoretical_cost"].replace(0, np.nan)
    by_kind = by_ing.groupby("kind", as_index=False)[["theoretical_cost", "actual_cost", "logged_waste_cost", "unexplained_cost"]].sum()
    tot = by_ing[["theoretical_cost", "actual_cost", "logged_waste_cost", "unexplained_cost", "variance_cost"]].sum()
    summary = {k: round(float(v), 2) for k, v in tot.items()}
    summary["variance_pct_of_theoretical"] = round(float(tot["variance_cost"] / tot["theoretical_cost"]), 4)
    return {"summary": summary, "by_ingredient": by_ing, "by_kind": by_kind}


def daily_actual_cogs(r: pd.DataFrame, open_dates: pd.Series) -> pd.Series:
    """Spread each period's actual stock cost over its open days, to get a monthly actual COGS series."""
    per = r.groupby(["period_start", "period_end"], as_index=False)["actual_cost"].sum()
    out = {}
    od = pd.DatetimeIndex(sorted(open_dates))
    for _, row in per.iterrows():
        days = od[(od > row["period_start"]) & (od <= row["period_end"])]
        for d in days:
            out[d] = row["actual_cost"] / max(len(days), 1)
    s = pd.Series(out).sort_index()
    return s.groupby(s.index.to_period("M").to_timestamp()).sum()


def run(tables=None, output_dir=OUTPUT_DIR) -> dict:
    t = tables or load_tables()
    r = reconcile(t)
    res = summarize(r, t["ingredients"])
    output_dir.mkdir(parents=True, exist_ok=True)
    r.to_csv(output_dir / "food_cost_reconciliation.csv", index=False)
    res["by_ingredient"].sort_values("unexplained_cost", ascending=False).to_csv(output_dir / "food_cost_by_ingredient.csv", index=False)
    weekly = r.groupby("period_end", as_index=False)[["theoretical_cost", "actual_cost", "logged_waste_cost", "unexplained_cost"]].sum()
    weekly["variance_pct"] = (weekly["actual_cost"] - weekly["theoretical_cost"]) / weekly["theoretical_cost"]
    weekly.to_csv(output_dir / "food_cost_weekly.csv", index=False)
    (output_dir / "food_cost_summary.json").write_text(json.dumps(res["summary"], indent=2))
    s = res["summary"]
    print(f"theoretical {s['theoretical_cost']:,.0f}  actual {s['actual_cost']:,.0f}  gap {s['variance_pct_of_theoretical']:.1%}  "
          f"of which logged waste {s['logged_waste_cost']:,.0f}, unexplained {s['unexplained_cost']:,.0f}")
    res["reconciliation"] = r
    return res


def main() -> None:
    run()


if __name__ == "__main__":
    main()
