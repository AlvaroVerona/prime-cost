"""Collects the headline numbers of every stage into one file used by the README and the dashboard overview."""

from __future__ import annotations

import json

import pandas as pd

from prime_cost.config import OUTPUT_DIR


def build(output_dir=OUTPUT_DIR) -> dict:
    j = lambda n: json.loads((output_dir / n).read_text())
    q, prof, fc = j("quality_report.json"), j("profitability_summary.json"), j("food_cost_summary.json")
    me, fo, st, pu, sc = j("menu_engineering_summary.json"), j("forecast_metrics.json"), j("staffing_summary.json"), j("purchasing_summary.json"), j("scenarios_summary.json")
    pnl = pd.read_csv(output_dir / "monthly_pnl.csv", parse_dates=["month"])
    last12 = pnl.tail(12)
    delivery = prof["unit_economics"].get("delivery", {})
    dine_in = prof["unit_economics"].get("sala", {})
    h = {
        "quality_score": q["score"]["overall"],
        "records_quarantined": int(sum(q["quarantined"].values())),
        "revenue_two_years": prof["revenue"],
        "last12_revenue": round(float(last12["revenue"].sum()), 0),
        "last12_ebitda": round(float(last12["ebitda"].sum()), 0),
        "last12_ebitda_pct": round(float(last12["ebitda"].sum() / last12["revenue"].sum()), 4),
        "last12_prime_cost_pct": round(float((last12["cogs_actual"] + last12["labor"]).sum() / last12["revenue"].sum()), 4),
        "theoretical_food_cost_pct": prof["theoretical_food_cost_pct"],
        "actual_vs_theoretical_gap_pct": fc["variance_pct_of_theoretical"],
        "gap_cost_two_years": fc["variance_cost"],
        "break_even_monthly": prof["break_even"]["break_even_monthly_revenue"],
        "delivery_contribution_per_order": delivery.get("avg_contribution"),
        "dine_in_contribution_per_ticket": dine_in.get("avg_contribution"),
        "menu_classes": me["margin_share_by_class"],
        "forecast_best_model": fo["best_model"],
        "forecast_wape": fo["models"][fo["best_model"]]["wape"],
        "forecast_baseline_wape": fo["models"]["baseline_dow4"]["wape"],
        "forecast_improvement_pct": fo["improvement_vs_baseline_pct"],
        "interval_coverage": fo["interval_coverage_out_of_sample"],
        "staffing_saving_pct": st["labor_saving_pct"],
        "staffing_saving_annualized": st["labor_saving_annualized_eur"],
        "staffing_gap_hours": [st["current"]["hours_with_a_gap"], st["optimized"]["hours_with_a_gap"]],
        "purchasing_saving_12w": pu["saving_mean"],
        "purchasing_saving_annualized": pu["saving_annualized"],
        "purchasing_waste_12w": [pu["baseline"]["waste_cost"], pu["forecast"]["waste_cost"]],
        "purchasing_lost_margin_12w": [pu["baseline"]["lost_margin"], pu["forecast"]["lost_margin"]],
        "scenario_status_quo_ebitda": sc["status_quo"]["annual_ebitda_mean"],
        "scenario_recommendations_ebitda": sc["recommendations"]["annual_ebitda_mean"],
        "scenario_uplift": sc["uplift_mean"],
        "scenario_prob_loss_month_status_quo": sc["status_quo"]["prob_any_monthly_loss"],
    }
    (output_dir / "headline.json").write_text(json.dumps(h, indent=2))
    return h


def main() -> None:
    h = build()
    print(json.dumps(h, indent=2))


if __name__ == "__main__":
    main()
