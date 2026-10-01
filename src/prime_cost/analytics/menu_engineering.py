"""Menu engineering (popularity x contribution margin) and price elasticity from the real price rounds.

Classification follows Kasavana & Smith, as in the consulting manual (Stars, Plowhorses, Puzzles, Dogs), within each
family of products: comparing a croqueta with a bottle of Priorat says nothing, so each family has its own matrix.

Elasticity is estimated with a difference-in-differences around each price round (treated categories versus the
categories whose price did not move). The honest finding is that a 4-5% price move over a few weeks identifies
elasticity only roughly, so every price recommendation is also tested against a range of elasticities.
Run: `python -m prime_cost.analytics.menu_engineering`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prime_cost.analytics.facts import build_line_facts, load_tables
from prime_cost.config import OUTPUT_DIR, load_settings

FAMILIES = {"Tapas": "Tapas", "Postres": "Postres y cafe", "Cafe": "Postres y cafe", "Bebidas": "Bebidas",
            "Vino por copa": "Vino por copa", "Vino botella": "Vino botella"}
ACTIONS = {
    "Star": "Highlight, protect quality, do not discount",
    "Plowhorse": "Popular but low margin: raise price a little, trim portion or cost, improve the recipe",
    "Puzzle": "High margin but under-ordered: better name and photo, move up the menu, staff recommendation",
    "Dog": "Low margin and low popularity: redesign, replace or drop",
}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Kasavana-Smith within one family: popular if mix >= 70% of an equal share, high margin if >= weighted average."""
    n = len(df)
    out = df.copy()
    out["mix_pct"] = out["units"] / out["units"].sum()
    pop_threshold = 0.7 / n
    avg_cm = (out["unit_margin"] * out["units"]).sum() / out["units"].sum()
    out["popular"] = out["mix_pct"] >= pop_threshold
    out["high_margin"] = out["unit_margin"] >= avg_cm
    out["avg_unit_margin_family"] = avg_cm
    out["class"] = np.select(
        [out["popular"] & out["high_margin"], out["popular"] & ~out["high_margin"], ~out["popular"] & out["high_margin"]],
        ["Star", "Plowhorse", "Puzzle"], default="Dog")
    out["action"] = out["class"].map(ACTIONS)
    return out


def menu_matrix(f: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    last = f["date"].max()
    g = f[f["date"] > last - pd.DateOffset(months=months)]
    it = g.groupby(["item_id", "name", "category"], as_index=False).agg(units=("qty", "sum"), revenue=("net_revenue", "sum"), cost=("cost", "sum"))
    it["unit_price"] = it["revenue"] / it["units"]
    it["unit_cost"] = it["cost"] / it["units"]
    it["unit_margin"] = it["unit_price"] - it["unit_cost"]
    it["food_cost_pct"] = it["unit_cost"] / it["unit_price"]
    it["family"] = it["category"].map(FAMILIES)
    parts = [classify(grp) for _, grp in it.groupby("family")]
    out = pd.concat(parts, ignore_index=True)
    out["total_margin"] = out["unit_margin"] * out["units"]
    return out.sort_values(["family", "total_margin"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- elasticity
def _window_rate(daily: pd.DataFrame, cats: list[str], lo: pd.Timestamp, hi: pd.Timestamp, days: np.ndarray | None = None) -> float:
    d = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
    if days is not None:
        d = d.set_index("date").loc[days].reset_index()
    return d[cats].sum().sum() / d["covers"].sum()


def estimate_elasticity(f: pd.DataFrame, calendar: pd.DataFrame, rnd: dict, weeks: int = 12, n_boot: int = 400, seed: int = 0) -> list[dict]:
    """Difference-in-differences of units per cover around one price round, with a day-level bootstrap CI."""
    covers = f.drop_duplicates("ticket_id").groupby("date")["covers"].sum().rename("covers")
    units = f.pivot_table(index="date", columns="category", values="qty", aggfunc="sum", fill_value=0)
    daily = units.join(covers).reset_index().dropna(subset=["covers"])
    all_cats = [c for c in units.columns]
    treated_all = [c for c in rnd["categories"] if c in all_cats]
    control = [c for c in all_cats if c not in rnd["categories"] and c not in ("Cafe",)]
    start = pd.Timestamp(rnd["date"])
    pre = (start - pd.Timedelta(weeks=weeks), start - pd.Timedelta(days=8))
    post = (start + pd.Timedelta(days=7), start + pd.Timedelta(weeks=weeks))
    if pre[0] < daily["date"].min() or post[1] > daily["date"].max():
        return []
    log_change = np.log(1 + rnd["change"])
    rng = np.random.default_rng(seed)
    pre_days = daily.loc[(daily["date"] >= pre[0]) & (daily["date"] <= pre[1]), "date"].to_numpy()
    post_days = daily.loc[(daily["date"] >= post[0]) & (daily["date"] <= post[1]), "date"].to_numpy()
    out = []
    for cat in treated_all:
        def did(pd_days=None, po_days=None, cat=cat):
            tr = np.log(_window_rate(daily, [cat], *post, po_days) / _window_rate(daily, [cat], *pre, pd_days))
            ct = np.log(_window_rate(daily, control, *post, po_days) / _window_rate(daily, control, *pre, pd_days))
            return (tr - ct) / log_change
        est = did()
        boots = [did(rng.choice(pre_days, len(pre_days)), rng.choice(post_days, len(post_days))) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [5, 95])
        out.append({"category": cat, "round_date": rnd["date"], "price_change": rnd["change"], "elasticity": round(float(est), 2),
                    "ci90_low": round(float(lo), 2), "ci90_high": round(float(hi), 2), "se": round(float((hi - lo) / (2 * 1.645)), 3),
                    "significant_90": bool(hi < 0 or lo > 0)})
    return out


# ----------------------------------------------------------------------------------------------- price simulation
def shrink(estimate: float, se: float, prior_mean: float, prior_sd: float) -> float:
    """Precision-weighted average of a noisy estimate and a prior (empirical-Bayes style shrinkage)."""
    w_est, w_prior = 1.0 / se**2, 1.0 / prior_sd**2
    return (estimate * w_est + prior_mean * w_prior) / (w_est + w_prior)


def price_scenarios(matrix: pd.DataFrame, elasticities: dict[str, float], change: float = 0.05,
                    e_range: tuple[float, float, float] = (-1.2, -0.8, -0.4)) -> pd.DataFrame:
    """Annual contribution change if an item's price rises by `change`, under three elasticities (low/central/high)."""
    rows = []
    for _, r in matrix.iterrows():
        e_central = elasticities.get(r["category"], -0.7)
        p0, c, q = r["unit_price"], r["unit_cost"], r["units"]
        p1 = p0 * (1 + change)
        gains = {}
        for label, e in {"pessimistic": e_range[0], "central": e_central, "optimistic": e_range[2]}.items():
            gains[label] = (p1 - c) * q * (1 + change) ** e - (p0 - c) * q
        rows.append({"item_id": r["item_id"], "name": r["name"], "category": r["category"], "family": r["family"], "class": r["class"],
                     "units_12m": q, "unit_price": round(p0, 2), "new_price": round(p1, 2), "unit_cost": round(c, 2),
                     "elasticity_central": e_central, "gain_pessimistic": gains["pessimistic"], "gain_central": gains["central"],
                     "gain_optimistic": gains["optimistic"]})
    out = pd.DataFrame(rows)
    out["robust"] = out["gain_pessimistic"] > 0
    return out


def run(tables=None, output_dir=OUTPUT_DIR) -> dict:
    t = tables or load_tables()
    s = load_settings()
    f = build_line_facts(t, s)
    matrix = menu_matrix(f, months=s["analysis"]["menu"]["months"])
    cfg = s["analysis"]["menu"]
    est = []
    for rnd in s["pricing"]["rounds"]:
        est += estimate_elasticity(f, t["calendar"], rnd, weeks=cfg["window_weeks"], n_boot=cfg["n_boot"])
    el = pd.DataFrame(est)
    if len(el):
        el["elasticity_shrunk"] = [round(shrink(r.elasticity, r.se, cfg["elasticity_prior_mean"], cfg["elasticity_prior_sd"]), 2) for r in el.itertuples()]
    # elasticities used in the price simulation: the shrunk estimates (a noisy point estimate is never used raw)
    central = dict(zip(el["category"], el["elasticity_shrunk"], strict=True)) if len(el) else {}
    lo, hi = cfg["elasticity_range"]
    scen = price_scenarios(matrix, central, change=cfg["price_change_tested"], e_range=(lo, (lo + hi) / 2, hi))
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "menu_engineering.csv", index=False)
    el.to_csv(output_dir / "elasticity.csv", index=False)
    scen.to_csv(output_dir / "price_scenarios.csv", index=False)
    summary = {
        "classes": matrix.groupby(["family", "class"]).size().unstack(fill_value=0).to_dict("index"),
        "margin_share_by_class": (matrix.groupby("class")["total_margin"].sum() / matrix["total_margin"].sum()).round(4).to_dict(),
        "robust_price_gain_total": round(float(scen.loc[scen["robust"], "gain_central"].sum()), 2),
        "price_rise_tested": cfg["price_change_tested"],
    }
    (output_dir / "menu_engineering_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("classes:", matrix["class"].value_counts().to_dict(), "| robust +5% price gain/yr:", f"{summary['robust_price_gain_total']:,.0f}")
    return {"matrix": matrix, "elasticity": el, "scenarios": scen, "summary": summary}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
