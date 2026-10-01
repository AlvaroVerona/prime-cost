"""Staff roster, the current fixed weekly shift template, and monthly fixed expenses."""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_POOL = {"encargado": 2, "cocina": 5, "sala": 7, "barra": 4}


def day_type(weekday: int) -> str:
    return "sun" if weekday == 6 else ("fri_sat" if weekday in (4, 5) else "weekday")


def employees_table(settings: dict) -> pd.DataFrame:
    rows = []
    for role, n in ROLE_POOL.items():
        for k in range(n):
            rows.append({"employee_id": f"E_{role[:3].upper()}{k + 1}", "role": role, "hourly_cost": settings["labor"]["wages"][role]})
    return pd.DataFrame(rows)


def build_shifts(cal: pd.DataFrame, settings: dict, rng: np.random.Generator) -> pd.DataFrame:
    """The shifts actually worked: the fixed template applied to every open day (the owner never adapts it)."""
    emp = employees_table(settings)
    pools = {r: emp.loc[emp["role"] == r, "employee_id"].tolist() for r in ROLE_POOL}
    cursor = dict.fromkeys(pools, 0)
    wages = settings["labor"]["wages"]
    rows = []
    for _, day in cal[cal["open"]].iterrows():
        tpl = settings["labor"]["template"][day_type(int(day["weekday"]))]
        for role, start, end, n in tpl:
            for _ in range(n):
                pool = pools[role]
                eid = pool[cursor[role] % len(pool)]
                cursor[role] += 1
                hours = end - start
                rows.append({"date": day["date"], "employee_id": eid, "role": role, "start_hour": start, "end_hour": end,
                             "hours": hours, "hourly_cost": wages[role], "cost": round(hours * wages[role], 2)})
    shifts = pd.DataFrame(rows)
    shifts.insert(0, "shift_id", [f"SH{i + 1:07d}" for i in range(len(shifts))])
    return shifts


def build_expenses(settings: dict) -> pd.DataFrame:
    p = settings["period"]
    months = pd.date_range(pd.Timestamp(p["start"]).replace(day=1), p["end"], freq="MS")
    e = settings["expenses_monthly"]
    rows = []
    for m in months:
        years_indexed = (m.year - months[0].year) + (1 if m.month >= e["rent_indexation"]["month"] and m.year > months[0].year else 0)
        rent = e["rent"] * (1 + e["rent_indexation"]["rate"]) ** max(years_indexed - (1 if months[0].month >= e["rent_indexation"]["month"] else 0), 0)
        util = e["utilities_base"] + (e["utilities_summer_extra"] if m.month in (6, 7, 8, 9) else 0)
        for concept, amount in [("rent", rent), ("utilities", util), ("insurance", e["insurance"]), ("software_pos", e["software_pos"]),
                                ("accounting", e["accounting"]), ("marketing", e["marketing"]), ("maintenance", e["maintenance"]),
                                ("cleaning", e["cleaning"]), ("licences_other", e["licences_other"]),
                                ("consumables_breakage", e["consumables_breakage"])]:
            rows.append({"month": m, "concept": concept, "amount": round(float(amount), 2)})
    return pd.DataFrame(rows)
