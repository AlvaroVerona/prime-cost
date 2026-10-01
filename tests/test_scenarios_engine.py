import numpy as np
import pandas as pd
import pytest

from prime_cost.simulation import scenarios
from prime_cost.simulation.inventory_engine import InventoryEngine


@pytest.fixture(scope="module")
def base():
    months = pd.date_range("2025-10-01", periods=12, freq="MS")
    return pd.DataFrame({"month": months, "revenue": 90_000.0, "cogs_actual": 28_000.0, "labor": 26_000.0, "fixed_expenses": 20_000.0,
                         "delivery_cost": 1_500.0, "card_fee": 700.0})


CFG = {"growth_haircut": 0.5, "growth_extra_sd": 0.04, "wage_rise_mean": 0.04, "wage_rise_sd": 0.015, "rent_rise": 0.03}
DRV = {"growth_yoy": 0.08, "growth_se": 0.01, "demand_noise_monthly": 0.03, "cost_drift_monthly": 0.002, "cost_vol_monthly": 0.01}


def test_monte_carlo_shapes_and_reproducibility(base):
    a = scenarios.simulate(base, DRV, CFG, {}, 500, np.random.default_rng(1))
    b = scenarios.simulate(base, DRV, CFG, {}, 500, np.random.default_rng(1))
    assert a["ebitda"].shape == (500, 12)
    np.testing.assert_allclose(a["ebitda"], b["ebitda"])


def test_recommendations_help_on_the_very_same_random_draws(base):
    sq = scenarios.simulate(base, DRV, CFG, {}, 800, np.random.default_rng(2))
    rec = scenarios.simulate(base, DRV, CFG, {"labor_saving_pct": 0.05, "waste_saving_pct_of_cogs": 0.03}, 800, np.random.default_rng(99), draws=sq["draws"])
    assert (rec["ebitda"].sum(axis=1) > sq["ebitda"].sum(axis=1)).all()


def test_higher_cost_volatility_widens_the_profit_distribution(base):
    calm = scenarios.simulate(base, DRV, CFG, {}, 2000, np.random.default_rng(3))["ebitda"].sum(axis=1)
    wild = scenarios.simulate(base, {**DRV, "cost_vol_monthly": 0.04}, CFG, {}, 2000, np.random.default_rng(3))["ebitda"].sum(axis=1)
    assert wild.std() > calm.std()


def test_sensitivity_signs(base):
    s = scenarios.sensitivity(base).set_index("driver")
    assert s.at["Guests (volume)", "ebitda_change_up"] > 0
    assert s.at["Wages", "ebitda_change_up"] < 0 and s.at["Ingredient prices", "ebitda_change_up"] < 0
    assert s.at["Menu prices (no volume reaction)", "ebitda_change_up"] > s.at["Menu prices (elasticity -0.8)", "ebitda_change_up"] > 0


# ------------------------------------------------------------------------------------------ inventory engine
def _engine(pipeline):
    snap = pipeline.gen["snapshot"]
    import copy

    return copy.deepcopy(snap)


def test_lots_expire_after_their_shelf_life(pipeline):
    eng = _engine(pipeline)
    day = sorted(eng.open_days)[3]
    for lots in eng.lots.values():
        lots.clear()
    eng.stock = dict.fromkeys(eng.stock, 0.0)
    eng._add_lot("gambas", day, 5.0)  # shelf life 2 days
    eng._expire(day + pd.Timedelta(days=1), record=False)
    assert eng.stock["gambas"] == pytest.approx(5.0)
    eng._expire(day + pd.Timedelta(days=2), record=False)
    assert eng.stock["gambas"] == 0.0


def test_fifo_consumption_takes_the_oldest_lot_first(pipeline):
    eng = _engine(pipeline)
    d = sorted(eng.open_days)[3]
    eng.lots["patata"].clear()
    eng.stock["patata"] = 0.0
    eng._add_lot("patata", d, 2.0)
    eng._add_lot("patata", d + pd.Timedelta(days=1), 3.0)
    eng._take("patata", 2.5)
    assert eng.lots["patata"][0][1] == pytest.approx(2.5)
    assert eng.stock["patata"] == pytest.approx(2.5)


def test_an_order_is_lost_when_an_ingredient_is_missing(pipeline):
    eng = _engine(pipeline)
    d = sorted(eng.open_days)[3]
    eng.today = d
    eng.stock["jamon"] = 0.0
    eng.lots["jamon"].clear()
    lines = pd.DataFrame({"item_id": ["jamon_racion"], "qty": [3], "line_time": [d + pd.Timedelta(hours=20)], "is_void": [False]})
    out = eng._serve(d, lines, record=False)
    assert out["sold_qty"].iat[0] == 0


def test_open_wine_bottle_is_discarded_after_its_glass_life(pipeline):
    eng = _engine(pipeline)
    d = sorted(eng.open_days)[3]
    bottle = "bottle_w08"  # sparkling: 1 day
    eng.stock[bottle] = 10.0
    eng.lots[bottle].clear()
    eng._add_lot(bottle, d, 10.0)
    assert eng._glass(bottle, d)
    before = len(eng.physical_loss)
    eng._close_bottles(d + pd.Timedelta(days=1), record=False)
    assert len(eng.physical_loss) == before + 1 and eng.physical_loss[-1]["cause"] == "open_bottle"
    assert isinstance(eng, InventoryEngine)
