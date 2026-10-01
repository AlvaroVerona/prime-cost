import numpy as np
import pandas as pd
import pytest

from prime_cost.config import load_menu, small_settings
from prime_cost.optimization import purchasing, staffing
from prime_cost.optimization.staffing import OPT_ROLES, required_heads, scheduled_heads, solve_week

SHARES = {k: pd.Series({h: 1 / 12 for h in range(13, 25)}) for k in ("weekday", "fri_sat", "sun")}
_peak_w = pd.Series([1, 1, 1, 1, 1, 1, 3, 10, 10, 6, 1, 1], index=range(13, 25), dtype=float)
PEAKY = dict.fromkeys(("weekday", "fri_sat", "sun"), _peak_w / _peak_w.sum())


def test_required_heads_grow_with_demand_and_never_drop_below_one():
    s = small_settings()
    lo = required_heads(40, "weekday", SHARES, s)
    hi = required_heads(400, "weekday", SHARES, s)
    for r in OPT_ROLES:
        assert all(v >= 1 for v in lo[r].values())
        assert max(hi[r].values()) > max(lo[r].values())
    assert lo["cocina"][s["labor"]["prep_hour"]] == 1  # mise en place before opening


def test_solver_covers_every_required_hour_and_respects_pool_limits():
    s = small_settings()
    days = [{"date": pd.Timestamp("2025-04-08") + pd.Timedelta(days=i), "dtype": "weekday" if i < 3 else "fri_sat",
             "req": required_heads(60 + 40 * i, "weekday" if i < 3 else "fri_sat", SHARES, s)} for i in range(5)]
    plan = solve_week(days, s)
    heads = scheduled_heads(plan)
    for d in days:
        for r in OPT_ROLES:
            for h, need in d["req"][r].items():
                assert heads.get((d["date"], r, h), 0) >= need
    df = pd.DataFrame(plan)
    from prime_cost.data.labor import ROLE_POOL

    for r in OPT_ROLES:
        sub = df[df["role"] == r]
        assert (sub["n"] * (sub["end_hour"] - sub["start_hour"])).sum() <= ROLE_POOL[r] * 40
        assert sub["n"].sum() <= ROLE_POOL[r] * 5


def test_optimized_plan_costs_less_than_staffing_every_hour_for_the_peak():
    s = small_settings()
    d = {"date": pd.Timestamp("2025-04-08"), "dtype": "weekday", "req": required_heads(160, "weekday", PEAKY, s)}
    plan = pd.DataFrame(solve_week([d], s))
    cost = sum(r.n * (r.end_hour - r.start_hour) * s["labor"]["wages"][r.role] for r in plan.itertuples())
    peak = {r: max(d["req"][r].values()) for r in OPT_ROLES}
    flat = sum(peak[r] * 12 * s["labor"]["wages"][r] for r in OPT_ROLES)  # peak crew for every hour incl. the prep hour
    assert cost < flat


def test_forecast_policy_does_not_order_for_days_after_the_stock_expires():
    pol = purchasing.ForecastPolicy(pd.DataFrame({"origin": [], "date": [], "forecast": []}), pd.DataFrame(), pd.Series(dtype=float), 0.15,
                                    {"gambas": 0.5}, {"gambas": 2, "aceite": 365})
    days = list(pd.date_range("2025-05-06", periods=5))
    assert len(pol._within_shelf("gambas", days)) == 2  # 2-day shelf life
    assert len(pol._within_shelf("aceite", days)) == 5  # dry goods keep


def test_newsvendor_z_is_higher_for_dry_goods_than_for_perishables():
    from prime_cost.data.catalog import build_catalog

    z = purchasing.newsvendor_z(None, build_catalog(load_menu()), 2.3)
    assert z["aceite"] > z["gambas"] > 0  # spoiling is costly, so keep less buffer; a missing dry good costs more than holding it


def test_replay_is_reproducible_for_a_given_seed(pipeline):
    import pickle

    with (pipeline.proc / "engine_snapshot.pkl").open("rb") as f:
        snap = pickle.load(f)
    hold = pd.read_parquet(pipeline.proc / "holdout_desired_lines.parquet")
    by_day = {d: g.reset_index(drop=True) for d, g in hold.groupby("date")}
    days = sorted(snap["cal"].loc[snap["cal"]["date"] >= snap["holdout_start"], "date"])[:14]
    a = purchasing.replay(snap["engine"], snap["engine"].policy, by_day, days, seed=5)
    b = purchasing.replay(snap["engine"], snap["engine"].policy, by_day, days, seed=5)
    assert a[1]["cost"].sum() == pytest.approx(b[1]["cost"].sum())
    assert len(a[3]) == len(b[3])


def test_end_to_end_purchasing_run_produces_a_summary(pipeline):
    from prime_cost.forecasting import demand

    s = pipeline.settings
    s["analysis"]["forecast"] = {**s["analysis"]["forecast"], "min_train_days": 60}
    demand.run(pipeline.raw, pipeline.out, s, processed_dir=pipeline.proc)
    res = purchasing.run(pipeline.proc, pipeline.out, s, n_seeds=2)
    sm = res["summary"]
    assert sm["seeds"] == 2 and np.isfinite(sm["saving_mean"])
    assert sm["baseline"]["leak"] > 0 and sm["forecast"]["leak"] > 0


def test_end_to_end_staffing_run_covers_demand_at_lower_cost(pipeline):
    s = pipeline.settings
    res = staffing.run(pipeline.proc, pipeline.out, s)
    sm = res["summary"]
    assert sm["optimized"]["cost"] < sm["current"]["cost"]
    assert sm["optimized"]["overstaffed_head_hours"] < sm["current"]["overstaffed_head_hours"]
