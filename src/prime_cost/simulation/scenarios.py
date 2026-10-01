"""Next-12-months profit under uncertainty (Monte Carlo) and a sensitivity analysis of the key drivers.

Everything random is estimated from the bar's own history: year-on-year demand growth and its monthly noise, and the volatility
of ingredient prices. Two strategies are simulated on the same random draws: keep running the bar as today, or apply the
recommendations of the analysis (an optimized staff schedule, forecast-based purchasing, a tested price rise on robust items).
Run: `python -m prime_cost.simulation.scenarios` (after the analytics and optimizers).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prime_cost.analytics.facts import build_line_facts, load_tables
from prime_cost.analytics.food_cost import daily_actual_cogs, reconcile
from prime_cost.analytics.profitability import monthly_pnl
from prime_cost.config import OUTPUT_DIR, load_settings


def estimate_drivers(pnl: pd.DataFrame, cost_history: pd.DataFrame, recipes: pd.DataFrame) -> dict:
    """Demand growth/noise from year-on-year revenue, and a monthly volatility of the ingredient cost index."""
    p = pnl.sort_values("month").reset_index(drop=True)
    yoy = np.log(p["revenue"].iloc[12:].to_numpy() / p["revenue"].iloc[:-12].to_numpy())
    n = min(len(yoy), 12)
    yoy = yoy[-n:]
    growth, noise = float(yoy.mean()), float(yoy.std(ddof=1) / np.sqrt(2))
    # cost index: ingredients weighted by their theoretical use in the recipes
    w = recipes.groupby("ingredient_id")["qty_gross"].sum()
    px = cost_history.pivot(index="date", columns="ingredient_id", values="agreed_price")
    monthly = px.groupby(px.index.to_period("M")).mean()
    idx = (monthly[w.index] * w).sum(axis=1)
    chg = np.log(idx).diff().dropna()
    return {"growth_yoy": growth, "growth_se": float(yoy.std(ddof=1) / np.sqrt(len(yoy))), "demand_noise_monthly": noise,
            "cost_drift_monthly": float(chg.mean()), "cost_vol_monthly": float(chg.std(ddof=1))}


def simulate(base: pd.DataFrame, drivers: dict, cfg: dict, strategy: dict, n: int, rng: np.random.Generator,
             draws: dict | None = None) -> dict:
    """base: last 12 months of P&L (one row per calendar month, in order). Returns monthly EBITDA paths (n x 12)."""
    months = len(base)
    d = draws or {}
    g = d.get("g", rng.normal(cfg["growth_haircut"] * drivers["growth_yoy"], np.sqrt(drivers["growth_se"] ** 2 + cfg["growth_extra_sd"] ** 2), n))
    eps = d.get("eps", rng.normal(0, drivers["demand_noise_monthly"], (n, months)))
    cost_steps = d.get("cost", rng.normal(drivers["cost_drift_monthly"], drivers["cost_vol_monthly"], (n, months)))
    wage = d.get("wage", rng.normal(cfg["wage_rise_mean"], cfg["wage_rise_sd"], n))
    t = np.arange(1, months + 1)
    growth = np.exp(g[:, None] * t / 12.0)
    revenue = base["revenue"].to_numpy()[None, :] * growth * np.exp(eps - drivers["demand_noise_monthly"] ** 2 / 2)
    cost_idx = np.exp(np.cumsum(cost_steps, axis=1))
    cogs_pct = (base["cogs_actual"] / base["revenue"]).to_numpy()[None, :]
    cogs = revenue * cogs_pct * cost_idx * (1 - strategy.get("waste_saving_pct_of_cogs", 0.0))
    first_year = base["month"].dt.month.to_numpy()
    jan = (first_year == 1)[None, :]
    wage_mult = np.where(np.cumsum(jan, axis=1) > 0, 1 + wage[:, None], 1.0)
    labor = base["labor"].to_numpy()[None, :] * wage_mult * (1 - strategy.get("labor_saving_pct", 0.0))
    rent_mult = np.where(np.cumsum(jan, axis=1) > 0, 1 + cfg["rent_rise"], 1.0)
    fixed = base["fixed_expenses"].to_numpy()[None, :] * rent_mult
    var_pct = ((base["delivery_cost"] + base["card_fee"]) / base["revenue"]).to_numpy()[None, :]
    revenue = revenue * (1 + strategy.get("price_effect_pct_of_revenue", 0.0))
    variable = revenue * var_pct
    ebitda = revenue - cogs - labor - fixed - variable
    return {"ebitda": ebitda, "revenue": revenue, "draws": {"g": g, "eps": eps, "cost": cost_steps, "wage": wage}}


def summarize(path: np.ndarray, revenue: np.ndarray) -> dict:
    annual = path.sum(axis=1)
    return {
        "annual_ebitda_mean": round(float(annual.mean()), 0), "annual_ebitda_p5": round(float(np.percentile(annual, 5)), 0),
        "annual_ebitda_p50": round(float(np.percentile(annual, 50)), 0), "annual_ebitda_p95": round(float(np.percentile(annual, 95)), 0),
        "annual_ebitda_margin_mean": round(float((annual / revenue.sum(axis=1)).mean()), 4),
        "prob_annual_loss": round(float((annual < 0).mean()), 4),
        "prob_any_monthly_loss": round(float((path < 0).any(axis=1).mean()), 4),
        "expected_loss_months": round(float((path < 0).sum(axis=1).mean()), 2),
    }


def sensitivity(base: pd.DataFrame, shock: float = 0.10, elasticity: float = -0.8) -> pd.DataFrame:
    """Change in annual EBITDA for a +/-10% move in each driver, everything else equal (labor and rent are fixed in the short run)."""
    rev, cogs = base["revenue"].sum(), base["cogs_actual"].sum()
    labor, fixed = base["labor"].sum(), base["fixed_expenses"].sum()
    var = (base["delivery_cost"] + base["card_fee"]).sum()
    contrib = rev - cogs - var

    def price(s: float, e: float) -> float:
        rev_f, vol_f = (1 + s) ** (1 + e) - 1, (1 + s) ** e - 1
        return rev * rev_f - cogs * vol_f - var * rev_f

    rows = [
        ("Guests (volume)", contrib * shock, -contrib * shock),
        ("Ingredient prices", -cogs * shock, cogs * shock),
        ("Wages", -labor * shock, labor * shock),
        ("Rent and fixed costs", -fixed * shock, fixed * shock),
        ("Menu prices (elasticity -0.8)", price(shock, elasticity), price(-shock, elasticity)),
        ("Menu prices (no volume reaction)", price(shock, 0.0), price(-shock, 0.0)),
    ]
    return pd.DataFrame([{"driver": n, "ebitda_change_up": round(float(u), 0), "ebitda_change_down": round(float(d), 0)} for n, u, d in rows])


def run(output_dir=OUTPUT_DIR, settings: dict | None = None) -> dict:
    s = settings or load_settings()
    cfg = s["analysis"]["scenarios"]
    t = load_tables()
    f = build_line_facts(t, s)
    r = reconcile(t)
    open_days = t["calendar"].loc[t["calendar"]["open"], "date"]
    pnl = monthly_pnl(f, t, daily_actual_cogs(r, open_days))
    pnl = pnl.dropna(subset=["revenue", "cogs_actual"]).reset_index(drop=True)
    drivers = estimate_drivers(pnl, t["cost_history"], t["recipes"])
    base = pnl.tail(12).reset_index(drop=True)

    st = json.loads((output_dir / "staffing_summary.json").read_text())
    pu = json.loads((output_dir / "purchasing_summary.json").read_text())
    me = json.loads((output_dir / "menu_engineering_summary.json").read_text())
    cogs_12w = pu["baseline"]["purchases"]
    waste_cash_saving = pu["baseline"]["waste_cost"] - pu["forecast"]["waste_cost"]
    strategies = {
        "status_quo": {},
        "recommendations": {
            "labor_saving_pct": st["labor_saving_pct_of_total_labor"] * cfg["recommendation_realisation"],
            "waste_saving_pct_of_cogs": waste_cash_saving / cogs_12w * cfg["recommendation_realisation"],
            "price_effect_pct_of_revenue": me["robust_price_gain_total"] / base["revenue"].sum() * cfg["recommendation_realisation"],
        },
    }
    rng = np.random.default_rng(s["seed"])
    n = cfg["n_sims"]
    sq = simulate(base, drivers, cfg, strategies["status_quo"], n, rng)
    rec = simulate(base, drivers, cfg, strategies["recommendations"], n, rng, draws=sq["draws"])  # same random draws
    results = {"drivers": drivers, "strategies": strategies,
               "status_quo": summarize(sq["ebitda"], sq["revenue"]), "recommendations": summarize(rec["ebitda"], rec["revenue"]),
               "n_sims": n, "base_year_revenue": round(float(base["revenue"].sum()), 0), "base_year_ebitda": round(float(base["ebitda"].sum()), 0)}
    results["uplift_mean"] = round(results["recommendations"]["annual_ebitda_mean"] - results["status_quo"]["annual_ebitda_mean"], 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenarios_summary.json").write_text(json.dumps(results, indent=2))
    months = base["month"].dt.strftime("%b").tolist()
    pd.DataFrame({"month": months, "status_quo_p5": np.percentile(sq["ebitda"], 5, axis=0), "status_quo_p50": np.percentile(sq["ebitda"], 50, axis=0),
                  "status_quo_p95": np.percentile(sq["ebitda"], 95, axis=0), "recommendations_p50": np.percentile(rec["ebitda"], 50, axis=0),
                  "prob_loss_status_quo": (sq["ebitda"] < 0).mean(axis=0), "prob_loss_recommendations": (rec["ebitda"] < 0).mean(axis=0)}).round(3).to_csv(output_dir / "scenarios_monthly.csv", index=False)
    np.savetxt(output_dir / "scenarios_annual_ebitda.csv", np.column_stack([sq["ebitda"].sum(axis=1), rec["ebitda"].sum(axis=1)])[:2000], delimiter=",", header="status_quo,recommendations", comments="", fmt="%.0f")
    sens = sensitivity(base)
    sens.to_csv(output_dir / "sensitivity.csv", index=False)
    pnl.to_csv(output_dir / "monthly_pnl.csv", index=False)
    print(f"EBITDA next 12m: status quo {results['status_quo']['annual_ebitda_mean']:,.0f} (P5 {results['status_quo']['annual_ebitda_p5']:,.0f}) -> "
          f"recommendations {results['recommendations']['annual_ebitda_mean']:,.0f}; P(loss year) {results['status_quo']['prob_annual_loss']:.1%}")
    return results


def main() -> None:
    run()


if __name__ == "__main__":
    main()
