import numpy as np
import pandas as pd
import pytest

from prime_cost.analytics import food_cost, menu_engineering, profitability
from prime_cost.analytics.facts import build_line_facts, unit_cost_matrix


@pytest.fixture(scope="module")
def facts(pipeline):
    return build_line_facts(pipeline.tables, pipeline.settings)


def test_theoretical_cost_is_recipe_times_price(pipeline, facts):
    t = pipeline.tables
    uc = unit_cost_matrix(t["recipes"], t["cost_history"])
    row = facts.iloc[0]
    assert row["unit_cost"] == pytest.approx(uc.at[row["date"], row["item_id"]])
    assert row["cost"] == pytest.approx(row["qty"] * row["unit_cost"])


def test_contribution_is_revenue_minus_all_variable_costs(facts):
    resid = facts["net_revenue"] - facts["cost"] - facts["delivery_cost"] - facts["card_fee"] - facts["contribution"]
    assert resid.abs().max() < 1e-6


def test_delivery_is_worth_much_less_per_order_than_dining_in(pipeline):
    res = profitability.run(pipeline.tables, pipeline.out)
    ue = res["unit_economics"]
    assert ue["delivery"]["avg_contribution"] < 0.5 * ue["sala"]["avg_contribution"]
    assert ue["delivery"]["contribution_pct"] < ue["sala"]["contribution_pct"]


def test_break_even_is_fixed_costs_over_contribution_margin(pipeline):
    res = profitability.run(pipeline.tables, pipeline.out)
    be = res["break_even"]
    assert be["break_even_monthly_revenue"] == pytest.approx(be["monthly_fixed_costs"] / be["contribution_margin_pct"], rel=1e-3)


def test_food_cost_gap_identity_and_recovers_the_hidden_losses(pipeline):
    r = food_cost.reconcile(pipeline.tables)
    # the accounting identity: actual = theoretical + logged waste + what is unexplained
    assert (r["actual"] - r["theoretical"] - r["logged_waste"] - r["unexplained"]).abs().max() < 1e-6
    summ = food_cost.summarize(r, pipeline.tables["ingredients"])["summary"]
    truth = sum(pipeline.gen["truth"]["physical_loss_by_cause"].values())
    # the measured gap should be close to the physical losses the simulator really produced (plus over-pouring, which it does not label)
    assert 0.85 * truth < summ["variance_cost"] < 1.4 * truth


def test_unexplained_loss_is_dominated_by_the_ingredients_with_hidden_overuse(pipeline):
    r = food_cost.reconcile(pipeline.tables)
    by = food_cost.summarize(r, pipeline.tables["ingredients"])["by_ingredient"].set_index("ingredient_id")
    top = by.sort_values("unexplained_cost", ascending=False).head(8).index
    assert "jamon" in top  # 9% over-portioning on the most expensive ingredient


# ----------------------------------------------------------------------------------- menu engineering
def _toy():
    return pd.DataFrame({
        "item_id": list("abcd"), "units": [400, 400, 100, 100], "unit_margin": [10.0, 4.0, 10.0, 4.0],
    })


def test_kasavana_smith_classification():
    out = menu_engineering.classify(_toy()).set_index("item_id")["class"]
    assert out["a"] == "Star" and out["b"] == "Plowhorse" and out["c"] == "Puzzle" and out["d"] == "Dog"


def test_every_item_gets_exactly_one_class(pipeline, facts):
    m = menu_engineering.menu_matrix(facts, months=12)
    assert m["class"].isin(["Star", "Plowhorse", "Puzzle", "Dog"]).all()
    assert m["item_id"].is_unique


def test_shrinkage_moves_towards_the_prior_when_the_estimate_is_noisy():
    noisy = menu_engineering.shrink(-2.0, se=2.0, prior_mean=-0.8, prior_sd=0.5)
    precise = menu_engineering.shrink(-2.0, se=0.05, prior_mean=-0.8, prior_sd=0.5)
    assert -0.9 < noisy < -0.8
    assert precise == pytest.approx(-2.0, abs=0.02)


def test_price_scenario_gain_depends_on_elasticity_and_margin():
    m = pd.DataFrame({"item_id": ["x", "y"], "name": ["x", "y"], "category": ["Tapas", "Tapas"], "family": ["Tapas"] * 2, "class": ["Star", "Dog"],
                      "units": [1000, 1000], "unit_price": [10.0, 10.0], "unit_cost": [2.0, 8.0]})
    s = menu_engineering.price_scenarios(m, {"Tapas": -0.8}, change=0.05).set_index("item_id")
    assert s.at["x", "gain_central"] > 0 and s.at["y", "gain_central"] > 0  # inelastic enough: a price rise pays on both
    assert s.at["y", "gain_central"] > s.at["x", "gain_central"]  # the same +5% weighs more on a thin margin
    assert s.at["x", "gain_pessimistic"] < s.at["x", "gain_optimistic"]
    harsh = menu_engineering.price_scenarios(m, {"Tapas": -4.0}, change=0.05).set_index("item_id")
    assert harsh.at["x", "gain_central"] < 0  # with a very elastic demand the same rise loses money
    assert np.isfinite(s["gain_central"]).all()
