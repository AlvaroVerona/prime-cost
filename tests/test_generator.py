import numpy as np
import pandas as pd

from prime_cost.config import load_menu, small_settings
from prime_cost.data.catalog import build_catalog, cost_matrix, price_matrix
from prime_cost.data.generator import generate


def test_same_seed_gives_the_same_data(tmp_path):
    s = small_settings(days=40, warmup_days=14)
    a = generate(s, load_menu(), write=False)["tables"]["tickets"]
    b = generate(s, load_menu(), write=False)["tables"]["tickets"]
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate(small_settings(days=40, warmup_days=14, seed=1), load_menu(), write=False)["tables"]["tickets"]
    b = generate(small_settings(days=40, warmup_days=14, seed=2), load_menu(), write=False)["tables"]["tickets"]
    assert len(a) != len(b) or not a["opened_at"].equals(b["opened_at"])


def test_monday_is_closed_and_every_ticket_has_a_known_item(pipeline):
    t = pipeline.gen["tables"]
    assert (t["tickets"]["date"].dt.weekday != 0).all()
    assert set(t["ticket_lines"]["item_id"]) <= set(t["items"]["item_id"])


def test_injected_duplicates_match_the_truth_log(pipeline):
    t = pipeline.gen["tables"]["tickets"]
    injected = pipeline.gen["truth"]["injected_quality_issues"]["duplicate_ticket"]
    assert t.duplicated("ticket_id").sum() == injected


def test_stock_never_goes_negative_and_stockouts_are_recorded(pipeline):
    eng = pipeline.gen["engine"]
    assert min(eng.stock.values()) >= -1e-9
    assert len(eng.stockouts) > 0  # a lean ordering rule does run out of things sometimes


def test_purchases_are_received_after_they_are_ordered(pipeline):
    p = pipeline.gen["tables"]["purchases"]
    assert (p["received_on"] >= p["ordered_on"]).all()
    assert (p["invoiced_price"] >= p["agreed_price"] - 1e-9).all()


def test_stock_counts_are_taken_on_the_closed_day(pipeline):
    c = pipeline.gen["tables"]["inventory_counts"]
    assert (c["count_date"].dt.weekday == 0).all()


def test_prices_follow_the_price_rounds():
    s = small_settings()
    s["pricing"] = {"rounds": [{"date": "2025-05-01", "categories": ["Tapas"], "change": 0.10}]}
    cat = build_catalog(load_menu())
    dates = pd.date_range("2025-04-01", "2025-06-01")
    p = price_matrix(cat, s, dates)
    before, after = p.loc["2025-04-15", "croquetas"], p.loc["2025-05-15", "croquetas"]
    assert after > before and abs(after / before - 1.10) < 0.02
    assert p.loc["2025-04-15", "cana"] == p.loc["2025-05-15", "cana"]  # drinks did not move


def test_ingredient_cost_shock_ramps_up_and_holds():
    s = small_settings()
    s["costs"]["shocks"] = {"aceite": {"start": "2025-04-01", "end": "2025-06-01", "change": 0.30}}
    cat = build_catalog(load_menu())
    dates = pd.date_range("2025-03-01", "2025-09-01")
    c = cost_matrix(cat, s, dates, np.random.default_rng(0))["aceite"]
    assert c.loc["2025-08-15"] > c.loc["2025-03-15"] * 1.15
