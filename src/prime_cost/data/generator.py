"""Orchestrates the synthetic data generation and writes the raw tables.

Run: `python -m prime_cost.data.generator`. Fully seeded: same settings, same data.
"""

from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np
import pandas as pd

from prime_cost.config import PROCESSED_DIR, RAW_DIR, load_menu, load_settings
from prime_cost.data.catalog import build_catalog, cost_matrix, price_matrix
from prime_cost.data.demand import build_calendar, generate_demand
from prime_cost.data.labor import build_expenses, build_shifts, employees_table
from prime_cost.simulation.inventory_engine import BaselinePolicy, InventoryEngine


def inject_quality_issues(tickets: pd.DataFrame, lines: pd.DataFrame, settings: dict, rng: np.random.Generator):
    """Corrupt a small share of rows the way real POS exports are corrupted. Returns (tickets, lines, log)."""
    q = settings["quality"]
    log: dict[str, int] = {}
    tickets = tickets.copy()
    lines = lines.copy()

    n = int(len(tickets) * q["duplicate_ticket_rate"])
    dup = tickets.sample(n, random_state=int(rng.integers(1_000_000)))
    tickets = pd.concat([tickets, dup], ignore_index=True)
    log["duplicate_ticket"] = n

    idx = tickets.sample(int(len(tickets) * q["missing_channel_rate"]), random_state=int(rng.integers(1_000_000))).index
    tickets.loc[idx, "channel"] = None
    log["missing_channel"] = len(idx)

    idx = lines.sample(int(len(lines) * q["negative_qty_rate"]), random_state=int(rng.integers(1_000_000))).index
    lines.loc[idx, "qty"] = -lines.loc[idx, "qty"]
    log["negative_qty"] = len(idx)

    idx = lines.sample(int(len(lines) * q["price_mismatch_rate"]), random_state=int(rng.integers(1_000_000))).index
    lines.loc[idx, "unit_price"] = (lines.loc[idx, "unit_price"] * rng.choice([0.1, 10.0], size=len(idx))).round(2)
    log["price_mismatch"] = len(idx)

    idx = tickets.sample(int(len(tickets) * q["bad_timestamp_rate"]), random_state=int(rng.integers(1_000_000))).index
    tickets.loc[idx, "opened_at"] = tickets.loc[idx, "opened_at"] - pd.Timedelta(hours=9)
    log["bad_timestamp"] = len(idx)
    return tickets, lines, log


def generate(settings: dict | None = None, menu: dict | None = None, write: bool = True) -> dict:
    settings = settings or load_settings()
    menu = menu or load_menu()
    t0 = time.time()
    ss = np.random.SeedSequence(settings["seed"])
    r_cal, r_cost, r_dem, r_eng, r_lab, r_q = (np.random.default_rng(s) for s in ss.spawn(6))

    catalog = build_catalog(menu)
    cal = build_calendar(settings, r_cal)
    dates = pd.DatetimeIndex(cal["date"])
    prices = price_matrix(catalog, settings, dates)
    costs = cost_matrix(catalog, settings, dates, r_cost)
    tickets, desired = generate_demand(cal, catalog, prices, settings, r_dem)

    policy = BaselinePolicy(settings["inventory"]["baseline_policy"]["safety_days"])
    engine = InventoryEngine(catalog, menu, settings, cal, costs, r_eng, policy)
    warm = settings["period"]["warmup_days"]
    start = pd.Timestamp(settings["period"]["start"])
    by_day = {d: g.reset_index(drop=True) for d, g in desired.groupby("date")}
    first = pd.concat([by_day[d] for d in sorted(by_day)[:14]])
    engine.seed_initial_stock(first, n_days=14)

    holdout_start = pd.Timestamp(settings["period"]["end"]) - pd.Timedelta(days=settings["holdout_days"] - 1)
    snapshot = None
    sold_parts = []
    for date in cal["date"]:
        if date == holdout_start:
            snapshot = copy.deepcopy(engine)
        lines = by_day.get(date)
        out = engine.run_day(date, lines, record=date >= start)
        if out is not None:
            sold_parts.append(out)
    sold = pd.concat(sold_parts, ignore_index=True)

    # ---- raw tables (period only) -------------------------------------------------------------------------
    in_period = lambda s: s >= start  # noqa: E731
    sold_lines = sold[in_period(sold["date"]) & (sold["sold_qty"] > 0)].copy()
    sold_lines["qty"] = sold_lines["sold_qty"]
    sold_lines = sold_lines.drop(columns=["sold_qty"]).reset_index(drop=True)
    sold_lines.insert(0, "line_id", [f"L{i + 1:09d}" for i in range(len(sold_lines))])
    tk = tickets[in_period(tickets["date"])].copy()
    tk = tk[tk["ticket_id"].isin(sold_lines["ticket_id"])].reset_index(drop=True)
    com = settings["demand"]["delivery"]
    net = sold_lines.assign(rev=lambda d: d["qty"] * d["unit_price"] * (1 - d["discount_pct"]) * (~d["is_void"])).groupby("ticket_id")["rev"].sum()
    tk["delivery_commission"] = np.where(tk["channel"] == "delivery", (tk["ticket_id"].map(net) * com["commission"]).round(2), 0.0)
    tk["packaging_cost"] = np.where(tk["channel"] == "delivery", com["packaging_cost"], 0.0)

    shifts = build_shifts(cal[in_period(cal["date"])], settings, r_lab)
    expenses = build_expenses(settings)
    employees = employees_table(settings)

    counts = pd.DataFrame(engine.counts)
    truth = {
        "overuse": engine.overuse,
        "physical_loss_by_cause": pd.DataFrame(engine.physical_loss).query("date >= @start").groupby("cause")["cost"].sum().round(2).to_dict(),
        "counts_true_qty": counts[["count_date", "ingredient_id", "_true_qty"]].assign(count_date=lambda d: d["count_date"].astype(str)).to_dict("records"),
        "expected_covers": cal.loc[cal["date"] >= start, ["date", "expected_covers"]].assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }
    counts = counts.drop(columns="_true_qty")

    tk_dirty, lines_dirty, qlog = inject_quality_issues(tk, sold_lines, settings, r_q)
    truth["injected_quality_issues"] = qlog

    cal_out = cal.loc[cal["date"] >= start, ["date", "weekday", "open", "holiday", "tasting", "temp_c", "rain", "covers"]].reset_index(drop=True)
    tables = {
        "calendar": cal_out, "tickets": tk_dirty, "ticket_lines": lines_dirty,
        "stockouts": pd.DataFrame(engine.stockouts), "purchases": pd.DataFrame(engine.purchases),
        "inventory_counts": counts, "waste_log": pd.DataFrame(engine.waste),
        "open_bottles": pd.DataFrame(engine.bottles), "shifts": shifts, "employees": employees, "expenses": expenses,
        "items": catalog.items.drop(columns=["daypart"]), "ingredients": catalog.ingredients,
        "recipes": catalog.recipes, "suppliers": catalog.suppliers,
        "price_history": prices.loc[prices.index >= start].rename_axis("date").reset_index().melt(id_vars="date", var_name="item_id", value_name="price"),
        "cost_history": costs.loc[costs.index >= start].rename_axis("date").reset_index().melt(id_vars="date", var_name="ingredient_id", value_name="agreed_price"),
    }
    tables["purchases"] = tables["purchases"][tables["purchases"]["received_on"] >= start].reset_index(drop=True)

    if write:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(RAW_DIR / f"{name}.parquet", index=False)
        (RAW_DIR / "_truth.json").write_text(json.dumps(truth, default=str))
        # artifacts for the purchasing backtest: engine state at the start of the holdout + the demand to replay
        hold = sold[sold["date"] >= holdout_start].copy()
        hold.to_parquet(PROCESSED_DIR / "holdout_desired_lines.parquet", index=False)
        with (PROCESSED_DIR / "engine_snapshot.pkl").open("wb") as f:
            pickle.dump({"engine": snapshot, "catalog": catalog, "costs": costs, "cal": cal, "holdout_start": holdout_start}, f)
    print(f"generated {len(tk)} tickets, {len(sold_lines)} lines, {len(shifts)} shifts in {time.time() - t0:.1f}s")
    return {"tables": tables, "truth": truth, "engine": engine, "snapshot": snapshot, "catalog": catalog, "cal": cal}


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
