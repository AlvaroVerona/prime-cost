"""Purchasing optimization: order against the demand forecast instead of last month's average.

The last weeks of the data are replayed with the SAME inventory engine that produced them, once with the current rule of thumb
(average daily use x cover days + safety days) and once with a forecast-based policy:

    order-up-to level = forecast use over the days the delivery must cover (capped at the shelf life) + newsvendor safety stock

The safety stock is z x forecast error, where z comes from the critical ratio  underage / (underage + overage):  an ingredient
that spoils is expensive to over-order, a dry good is not, a missing ingredient costs the margin of the dishes that cannot be sold.
Both policies see exactly the same guests; the replay is repeated with several random seeds (the engine has chance events such as
prep errors and invoice mistakes) so the result comes with a spread, not a single lucky number.
Run: `python -m prime_cost.optimization.purchasing` (after the forecast).
"""

from __future__ import annotations

import copy
import json
import pickle
from statistics import NormalDist

import numpy as np
import pandas as pd

from prime_cost.config import OUTPUT_DIR, PROCESSED_DIR, load_settings


class ForecastPolicy:
    name = "forecast"

    def __init__(self, forecast: pd.DataFrame, usage_rate: pd.DataFrame, covers_hist: pd.Series, rel_sd: float, z: dict, shelf: dict,
                 perishable_days: int = 14):
        self.fc = forecast  # columns: origin, date, forecast
        self.rate = usage_rate  # index ingredient, columns weekday 0..6 (ingredient units per guest)
        self.covers_hist = covers_hist
        self.rel_sd = rel_sd
        self.z = z
        self.shelf = shelf
        self.perishable_days = perishable_days
        self._by_origin = {o: g.set_index("date")["forecast"] for o, g in forecast.groupby("origin")}
        self._origins = sorted(self._by_origin)

    def covers_forecast(self, today: pd.Timestamp, day: pd.Timestamp) -> float:
        for o in reversed([o for o in self._origins if o <= today]):
            s = self._by_origin[o]
            if day in s.index:
                return float(s.loc[day])
        same = self.covers_hist[(self.covers_hist.index < today) & (self.covers_hist.index.weekday == day.weekday())].tail(4)
        return float(same.mean()) if len(same) else 0.0

    def _use(self, engine, ing: str, days: list[pd.Timestamp]) -> np.ndarray:
        if ing not in self.rate.index:
            return np.zeros(len(days))
        return np.array([self.rate.at[ing, d.weekday()] * self.covers_forecast(engine.today, d) for d in days])

    def expected_use(self, engine, ing: str, days: list[pd.Timestamp]) -> float:
        return float(self._use(engine, ing, self._within_shelf(ing, days)).sum())

    def safety_stock(self, engine, ing: str, days: list[pd.Timestamp]) -> float:
        u = self._use(engine, ing, self._within_shelf(ing, days))
        return float(self.z.get(ing, 0.0) * self.rel_sd * np.sqrt((u**2).sum()))

    def _within_shelf(self, ing: str, days: list[pd.Timestamp]) -> list[pd.Timestamp]:
        """Stock received on days[0] is gone after `shelf` days: ordering for later days only creates waste."""
        shelf = self.shelf[ing]
        if shelf > self.perishable_days or not days:
            return days
        return [d for d in days if (d - days[0]).days < shelf]


def newsvendor_z(engine, catalog, gross_margin_multiple: float) -> dict:
    """z for each ingredient from the critical ratio underage / (underage + overage)."""
    ing = catalog.ingredients.set_index("ingredient_id")
    nd = NormalDist()
    z = {}
    for i, r in ing.iterrows():
        underage = r["base_cost"] * gross_margin_multiple
        overage = r["base_cost"] if r["shelf_days"] <= 14 else 0.05 * r["base_cost"]
        cr = underage / (underage + overage)
        z[i] = nd.inv_cdf(min(max(cr, 0.5), 0.995))
    return z


def usage_rates(engine, cal: pd.DataFrame, catalog, holdout_start: pd.Timestamp, weeks: int = 12) -> pd.DataFrame:
    """Ingredient used per guest by weekday over the last weeks before the holdout, counting the demand that went unserved."""
    u = pd.DataFrame(engine.usage_theory)
    lost = pd.DataFrame(engine.stockouts)
    if len(lost):
        rec = catalog.recipes[["item_id", "ingredient_id", "qty_gross"]]
        lost = lost.merge(rec, on="item_id").assign(qty=lambda d: d["lost_qty"] * d["qty_gross"])[["date", "ingredient_id", "qty"]]
        u = pd.concat([u, lost], ignore_index=True)
    lo = holdout_start - pd.Timedelta(weeks=weeks)
    u = u[(u["date"] >= lo) & (u["date"] < holdout_start)]
    cov = cal.set_index("date")["covers"]
    daily = u.groupby(["ingredient_id", "date"], as_index=False)["qty"].sum()
    daily["weekday"] = daily["date"].dt.weekday
    daily["covers"] = daily["date"].map(cov)
    g = daily.groupby(["ingredient_id", "weekday"]).agg(q=("qty", "sum"), c=("covers", "sum"))
    # divide by the covers of ALL open days of that weekday, not only days the ingredient was used
    days = cal[(cal["date"] >= lo) & (cal["date"] < holdout_start) & cal["open"]].copy()
    days["weekday"] = days["date"].dt.weekday
    cov_wd = days.groupby("weekday")["covers"].sum()
    rate = g["q"].unstack("weekday").fillna(0.0)
    rate = rate.div(cov_wd.reindex(rate.columns), axis=1)
    return rate.reindex(columns=range(7), fill_value=0.0).fillna(0.0)


def replay(engine_snapshot, policy, lines_by_day: dict, days: list[pd.Timestamp], seed: int):
    eng = copy.deepcopy(engine_snapshot)
    eng.rng = np.random.default_rng(seed)
    eng.policy = policy
    n_loss, n_pur, n_so = len(eng.physical_loss), len(eng.purchases), len(eng.stockouts)
    for d in days:
        eng.run_day(d, lines_by_day.get(d), record=True)
    loss = pd.DataFrame(eng.physical_loss[n_loss:])
    pur = pd.DataFrame(eng.purchases[n_pur:])
    so = pd.DataFrame(eng.stockouts[n_so:])
    return eng, loss, pur, so


def run(processed_dir=PROCESSED_DIR, output_dir=OUTPUT_DIR, settings: dict | None = None, n_seeds: int | None = None) -> dict:
    s = settings or load_settings()
    cfg = s["analysis"]["purchasing"]
    n_seeds = n_seeds or cfg["n_seeds"]
    with (processed_dir / "engine_snapshot.pkl").open("rb") as f:
        snap = pickle.load(f)
    engine, catalog, cal, holdout_start = snap["engine"], snap["catalog"], snap["cal"], snap["holdout_start"]
    hold = pd.read_parquet(processed_dir / "holdout_desired_lines.parquet")
    lines_by_day = {d: g.reset_index(drop=True) for d, g in hold.groupby("date")}
    days = sorted(cal.loc[cal["date"] >= holdout_start, "date"])
    ev = pd.read_parquet(processed_dir / "forecast_holdout.parquet")
    fc = ev[["origin", "date", "forecast"]]
    rel_sd = float(np.std(ev["covers"] / ev["forecast"] - 1))
    rate = usage_rates(engine, cal, catalog, holdout_start)
    z = newsvendor_z(engine, catalog, cfg["gross_margin_multiple"])
    shelf = catalog.ingredients.set_index("ingredient_id")["shelf_days"].to_dict()
    covers_hist = cal.set_index("date")["covers"]
    base_policy = engine.policy
    fc_policy = ForecastPolicy(fc, rate, covers_hist, rel_sd, z, shelf)

    # unit margin of each item: used to value the sales lost to stock-outs
    items = catalog.items.set_index("item_id")
    cost = catalog.recipes.merge(catalog.ingredients[["ingredient_id", "base_cost"]], on="ingredient_id").assign(c=lambda d: d["qty_gross"] * d["base_cost"])
    item_cost = cost.groupby("item_id")["c"].sum()
    unit_margin = (items["base_price"] - item_cost.reindex(items.index).fillna(0.0)).to_dict()

    runs, by_ing = [], []
    for k in range(n_seeds):
        for pol in (base_policy, fc_policy):
            eng, loss, pur, so = replay(engine, pol, lines_by_day, days, seed=1000 + k)
            waste = float(loss["cost"].sum()) if len(loss) else 0.0
            expiry = float(loss.loc[loss["cause"] == "expiry", "cost"].sum()) if len(loss) else 0.0
            lost_margin = float((so["lost_qty"] * so["item_id"].map(unit_margin)).sum()) if len(so) else 0.0
            spend = float((pur["qty_received"] * pur["invoiced_price"]).sum()) if len(pur) else 0.0
            end_value = sum(eng.stock[i] * float(eng.costs.at[days[-1], i]) for i in eng.stock)
            runs.append({"seed": k, "policy": pol.name, "waste_cost": waste, "expiry_cost": expiry, "lost_margin": lost_margin,
                         "lost_units": int(so["lost_qty"].sum()) if len(so) else 0, "purchases": spend, "end_stock_value": end_value})
            if len(loss):
                g = loss[loss["cause"] == "expiry"].groupby("ingredient_id")["cost"].sum().rename("expiry_cost").reset_index().assign(policy=pol.name, seed=k)
                by_ing.append(g)
    r = pd.DataFrame(runs)
    r["leak"] = r["waste_cost"] + r["lost_margin"]
    agg = r.groupby("policy").agg(["mean", "std"]).round(2)
    pairs = r.pivot(index="seed", columns="policy", values="leak")
    saving = pairs["baseline"] - pairs["forecast"]
    summary = {
        "weeks": len(days) // 7, "seeds": n_seeds,
        "baseline": {k: round(float(r.loc[r["policy"] == "baseline", k].mean()), 2) for k in ["waste_cost", "expiry_cost", "lost_margin", "lost_units", "purchases", "end_stock_value", "leak"]},
        "forecast": {k: round(float(r.loc[r["policy"] == "forecast", k].mean()), 2) for k in ["waste_cost", "expiry_cost", "lost_margin", "lost_units", "purchases", "end_stock_value", "leak"]},
        "saving_mean": round(float(saving.mean()), 2), "saving_std": round(float(saving.std()), 2),
        "saving_min": round(float(saving.min()), 2), "saving_share_of_seeds_positive": round(float((saving > 0).mean()), 2),
        "saving_annualized": round(float(saving.mean() / len(days) * 313), 2),
        "rel_forecast_sd": round(rel_sd, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    r.to_csv(output_dir / "purchasing_runs.csv", index=False)
    if by_ing:
        bi = pd.concat(by_ing).groupby(["policy", "ingredient_id"], as_index=False)["expiry_cost"].mean()
        bi.pivot(index="ingredient_id", columns="policy", values="expiry_cost").fillna(0.0).round(2).reset_index().to_csv(output_dir / "purchasing_expiry_by_ingredient.csv", index=False)
    (output_dir / "purchasing_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"waste + lost margin: baseline {summary['baseline']['leak']:,.0f} -> forecast {summary['forecast']['leak']:,.0f} "
          f"(saving {summary['saving_mean']:,.0f} +/- {summary['saving_std']:,.0f}; {summary['saving_share_of_seeds_positive']:.0%} of seeds better)")
    return {"summary": summary, "runs": r, "agg": agg}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
