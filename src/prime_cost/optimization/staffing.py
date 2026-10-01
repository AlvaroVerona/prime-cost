"""Weekly staffing plan: turn the demand forecast into the cheapest shifts that cover the hourly need.

The need per hour and role comes from the guests expected (and still being served) divided by what one person can serve.
A CP-SAT model chooses how many shifts of each length start at each hour, with a limit on hours and shifts per person.
The result is judged on the demand that REALLY happened, against the fixed weekly template the bar uses today.
Run: `python -m prime_cost.optimization.staffing` (after the forecast).
"""

from __future__ import annotations

import json
import math

import pandas as pd
from ortools.sat.python import cp_model

from prime_cost.config import OUTPUT_DIR, PROCESSED_DIR, load_settings
from prime_cost.data.labor import ROLE_POOL, day_type

OPT_ROLES = ["cocina", "sala", "barra"]


def hour_shares(tickets: pd.DataFrame, before: pd.Timestamp) -> dict[str, pd.Series]:
    """Share of the day's guests arriving in each hour, by type of day, from history only."""
    t = tickets[(tickets["date"] < before) & (tickets["channel"] != "delivery")].copy()
    t["dtype"] = t["date"].dt.weekday.map(day_type)
    out = {}
    for dt, g in t.groupby("dtype"):
        s = g.groupby("hour")["covers"].sum()
        out[dt] = (s / s.sum()).sort_index()
    return out


def required_heads(covers: float, dtype: str, shares: dict[str, pd.Series], settings: dict) -> dict[str, dict[int, int]]:
    """Required people per role and hour for a day with `covers` guests."""
    lab = settings["labor"]
    first, last = lab["service_hours"][dtype]
    share = shares[dtype]
    arrivals = {h: covers * float(share.get(h, 0.0)) for h in range(first - 1, last + 1)}
    req: dict[str, dict[int, int]] = {r: {} for r in OPT_ROLES}
    for h in range(first, last):
        present = arrivals.get(h, 0.0) + lab["stay_carryover"] * arrivals.get(h - 1, 0.0)
        for r in OPT_ROLES:
            req[r][h] = max(1, math.ceil(present / lab["guests_per_staff"][r] - 1e-9))
    req["cocina"][lab["prep_hour"]] = 1
    return req


def solve_week(days: list[dict], settings: dict) -> list[dict]:
    """days: [{date, dtype, req}]. Returns shifts [{date, role, start, end, n}]."""
    cfg = settings["analysis"]["staffing"]
    wages = settings["labor"]["wages"]
    model = cp_model.CpModel()
    xs, cover = {}, {}
    for d in days:
        first, last = settings["labor"]["service_hours"][d["dtype"]]
        for r in OPT_ROLES:
            lo = settings["labor"]["prep_hour"] if r == "cocina" else first
            for start in range(lo, last):
                for length in cfg["shift_lengths"]:
                    end = start + length
                    if end > last:
                        continue
                    v = model.NewIntVar(0, 6, f"x_{d['date']:%m%d}_{r}_{start}_{length}")
                    xs[(d["date"], r, start, length)] = v
                    for h in range(start, end):
                        cover.setdefault((d["date"], r, h), []).append(v)
    for d in days:
        for r in OPT_ROLES:
            for h, need in d["req"][r].items():
                model.Add(sum(cover.get((d["date"], r, h), [])) >= need)
    for r in OPT_ROLES:
        shifts_r = [(k, v) for k, v in xs.items() if k[1] == r]
        model.Add(sum(v for _, v in shifts_r) <= ROLE_POOL[r] * cfg["max_shifts_per_person_week"])
        model.Add(sum(k[3] * v for k, v in shifts_r) <= ROLE_POOL[r] * cfg["max_hours_per_person_week"])
    model.Minimize(sum(round(100 * wages[k[1]] * k[3]) * v + v for k, v in xs.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg["solver_seconds"]
    solver.parameters.num_workers = 4
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"staffing model infeasible (status {status})")
    out = []
    for (date, r, start, length), v in xs.items():
        n = solver.Value(v)
        if n > 0:
            out.append({"date": date, "role": r, "start_hour": start, "end_hour": start + length, "n": n})
    return out


def scheduled_heads(shifts: list[dict] | pd.DataFrame) -> dict[tuple, int]:
    df = pd.DataFrame(shifts)
    out: dict[tuple, int] = {}
    for r in df.itertuples():
        n = getattr(r, "n", 1)
        for h in range(r.start_hour, r.end_hour):
            out[(r.date, r.role, h)] = out.get((r.date, r.role, h), 0) + n
    return out


def evaluate(plan_heads: dict, actual_req: dict[pd.Timestamp, dict], settings: dict) -> dict:
    wages = settings["labor"]["wages"]
    under = over = cost = 0.0
    short_hours = total_hours = 0
    for date, req in actual_req.items():
        for r in OPT_ROLES:
            for h, need in req[r].items():
                have = plan_heads.get((date, r, h), 0)
                under += max(0, need - have)
                over += max(0, have - need)
                short_hours += int(have < need)
                total_hours += 1
    for (date, r, h), n in plan_heads.items():
        cost += n * wages[r]
    return {"cost": round(cost, 2), "understaffed_head_hours": round(under, 1), "overstaffed_head_hours": round(over, 1),
            "hours_with_a_gap": short_hours, "role_hours_checked": total_hours}


def run(processed_dir=PROCESSED_DIR, output_dir=OUTPUT_DIR, settings: dict | None = None, raw_dir=None) -> dict:
    s = settings or load_settings()
    from prime_cost.config import RAW_DIR
    raw_dir = raw_dir or RAW_DIR
    cfg = s["analysis"]["staffing"]
    ev = pd.read_parquet(processed_dir / "forecast_holdout.parquet")
    tickets = pd.read_parquet(processed_dir / "tickets_valid.parquet")
    shifts = pd.read_parquet(processed_dir / "shifts_valid.parquet")
    holdout_start = pd.Timestamp(s["period"]["end"]) - pd.Timedelta(days=s["holdout_days"] - 1)
    shares = hour_shares(tickets, holdout_start)
    # weekly plans published on Monday (closed day) with that Monday's forecast
    plan_fc = ev[ev["origin"].dt.weekday == 0].copy()
    plan_fc["week"] = plan_fc["origin"]
    plan_fc["covers_plan"] = plan_fc["forecast"] + cfg["buffer_share"] * (plan_fc["upper"] - plan_fc["forecast"])
    actual_by_day = ev.drop_duplicates("date").set_index("date")["covers"]

    all_plan = []
    cur_template = shifts[(shifts["date"] >= holdout_start) & (shifts["role"].isin(OPT_ROLES))]
    for week, g in plan_fc.groupby("week"):
        days = []
        for _, r in g.iterrows():
            dt = day_type(r["date"].weekday())
            days.append({"date": r["date"], "dtype": dt, "req": required_heads(r["covers_plan"], dt, shares, s)})
        all_plan += solve_week(days, s)
    plan_df = pd.DataFrame(all_plan)

    covered_days = sorted(plan_df["date"].unique())
    actual_req = {d: required_heads(actual_by_day[d], day_type(d.weekday()), shares, s) for d in covered_days}
    cur_heads = scheduled_heads(cur_template[cur_template["date"].isin(covered_days)].rename(columns={"start_hour": "start_hour"})[["date", "role", "start_hour", "end_hour"]])
    opt_heads = scheduled_heads(plan_df)
    res_cur = evaluate(cur_heads, actual_req, s)
    res_opt = evaluate(opt_heads, actual_req, s)

    n_days = len(covered_days)
    saving = res_cur["cost"] - res_opt["cost"]
    total_labor = float(shifts.loc[shifts["date"].isin(covered_days), "cost"].sum())  # all roles, incl. the manager that is not optimized
    summary = {"days_evaluated": n_days, "current": res_cur, "optimized": res_opt, "labor_saving_eur": round(saving, 2),
               "labor_saving_pct": round(saving / res_cur["cost"], 4), "labor_saving_pct_of_total_labor": round(saving / total_labor, 4), "labor_saving_annualized_eur": round(saving / n_days * 313, 2),
               "gap_hours_reduction_pct": round(1 - res_opt["hours_with_a_gap"] / max(res_cur["hours_with_a_gap"], 1), 4),
               "buffer_share": cfg["buffer_share"]}
    # sensitivity to how much safety the plan carries
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(output_dir / "staffing_plan.csv", index=False)
    # hour-by-hour comparison (average week) for the dashboard
    rows = []
    for date in covered_days:
        for r in OPT_ROLES:
            for h in sorted(set(actual_req[date][r]) | {k[2] for k in cur_heads if k[0] == date and k[1] == r}):
                rows.append({"date": date, "weekday": date.weekday(), "role": r, "hour": h, "required": actual_req[date][r].get(h, 0),
                             "current": cur_heads.get((date, r, h), 0), "optimized": opt_heads.get((date, r, h), 0)})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(output_dir / "staffing_comparison.csv", index=False)
    (output_dir / "staffing_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"labor cost {res_cur['cost']:,.0f} -> {res_opt['cost']:,.0f} ({summary['labor_saving_pct']:.1%}); "
          f"hours with a gap {res_cur['hours_with_a_gap']} -> {res_opt['hours_with_a_gap']}")
    return {"summary": summary, "plan": plan_df, "comparison": cmp_df}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
