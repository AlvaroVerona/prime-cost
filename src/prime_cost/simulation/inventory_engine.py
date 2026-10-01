"""Day-by-day inventory simulation of the wine bar.

The same engine produces the synthetic operational data (purchases, waste, stock counts, stock-outs) and is
reused by the purchasing optimizer to re-run the last weeks under a different ordering policy on the SAME
realized demand, which makes the comparison fair.

What it models, per day:
  * deliveries arrive and are put in FIFO lots with an expiry date
  * lots that expired are thrown away (physical waste, only partly logged)
  * every guest order consumes ingredients; if any ingredient is missing, the order is lost (stock-out)
  * wine by the glass comes from open bottles that go off after a few days
  * operational waste happens (prep errors, returned dishes, spoiled storage)
  * the ordering policy places orders for the next delivery
  * on count days the physical stock is counted, with measurement noise

Hidden truth (physical use above the recipe, overpour, unlogged waste) is what the analytics later has to find.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from prime_cost.data.catalog import WINE_GLASS_CAT, Catalog


class BaselinePolicy:
    """The current rule of thumb: average daily use over the last weeks times (cover days + safety days)."""

    name = "baseline"

    def __init__(self, safety_days: dict):
        self.safety = safety_days

    def target(self, engine: InventoryEngine, ing: str, horizon_days: int, arrival: pd.Timestamp) -> float:
        kind = engine.ing_kind[ing]
        key = "drink" if kind in ("drink", "wine") else kind
        return engine.avg_use(ing) * (horizon_days + self.safety[key])


class InventoryEngine:
    def __init__(self, catalog: Catalog, menu: dict, settings: dict, cal: pd.DataFrame, costs: pd.DataFrame,
                 rng: np.random.Generator, policy):
        self.settings = settings
        self.inv = settings["inventory"]
        self.rng = rng
        self.policy = policy
        self.costs = costs
        ing = catalog.ingredients.set_index("ingredient_id")
        self.ing_kind = ing["kind"].to_dict()
        self.ing_unit = ing["unit"].to_dict()
        self.ing_supplier = ing["supplier_id"].to_dict()
        self.ing_pack = ing["pack"].to_dict()
        self.ing_shelf = ing["shelf_days"].to_dict()
        self.wine_type = ing["wine_type"].to_dict() if "wine_type" in ing else {}
        self.suppliers = menu["suppliers"]
        self.items = catalog.items.set_index("item_id")
        self.recipe_gross: dict[str, list[tuple[str, float]]] = {}
        for item_id, g in catalog.recipes.groupby("item_id"):
            self.recipe_gross[item_id] = list(zip(g["ingredient_id"], g["qty_gross"], strict=True))
        self.by_supplier: dict[str, list[str]] = {}
        for i, s in self.ing_supplier.items():
            self.by_supplier.setdefault(s, []).append(i)
        self.overuse = self._hidden_overuse()
        self.open_days = set(cal.loc[cal["open"], "date"])
        self.all_days = list(cal["date"])
        self.last_day = self.all_days[-1]

        self.lots = {i: deque() for i in self.ing_kind}
        self.stock = dict.fromkeys(self.ing_kind, 0.0)
        self.pending: list[dict] = []
        self.open_bottles: dict[str, list[list]] = {i: [] for i in self.ing_kind if self.ing_kind[i] == "wine"}
        self.hist = {i: deque(maxlen=self.inv["baseline_policy"]["history_days"]) for i in self.ing_kind}
        self.order_seq = 0
        # outputs
        self.purchases: list[dict] = []
        self.waste: list[dict] = []
        self.bottles: list[dict] = []
        self.counts: list[dict] = []
        self.usage_theory: list[dict] = []
        self.stockouts: list[dict] = []
        self.physical_loss: list[dict] = []  # ground truth of every physical loss, by cause (validation only)

    # ------------------------------------------------------------------ helpers
    def _hidden_overuse(self) -> dict:
        h = self.inv["hidden_overuse"]
        out = {}
        for i, k in self.ing_kind.items():
            kind = "drink" if k == "wine" else k
            out[i] = 0.0 if k == "wine" else h["by_ingredient"].get(i, h["by_kind"][kind])
        return out

    def avg_use(self, ing: str) -> float:
        h = self.hist[ing]
        return float(np.mean(h)) if h else 0.0

    def _cost(self, ing: str, date: pd.Timestamp) -> float:
        return float(self.costs.at[date, ing])

    def _add_lot(self, ing: str, date: pd.Timestamp, qty: float) -> None:
        self.lots[ing].append([date + pd.Timedelta(days=self.ing_shelf[ing]), qty])
        self.stock[ing] += qty

    def _take(self, ing: str, qty: float) -> float:
        """Remove up to qty from the oldest lots; returns what was actually removed."""
        need = qty
        lots = self.lots[ing]
        while need > 1e-12 and lots:
            if lots[0][1] <= need + 1e-12:
                need -= lots[0][1]
                lots.popleft()
            else:
                lots[0][1] -= need
                need = 0.0
        taken = qty - need
        self.stock[ing] = max(0.0, self.stock[ing] - taken)
        return taken

    def _next_open_on_or_after(self, date: pd.Timestamp) -> pd.Timestamp | None:
        d = date
        for _ in range(10):
            if d in self.open_days:
                return d
            d += pd.Timedelta(days=1)
        return None

    def _arrival(self, order_date: pd.Timestamp, supplier: str) -> pd.Timestamp | None:
        return self._next_open_on_or_after(order_date + pd.Timedelta(days=self.suppliers[supplier]["lead_days"]))

    def _next_order_day(self, after: pd.Timestamp, supplier: str) -> pd.Timestamp:
        days = self.suppliers[supplier]["order_days"]
        d = after + pd.Timedelta(days=1)
        while d.weekday() not in days:
            d += pd.Timedelta(days=1)
        return d

    def _open_days_between(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Open days in [start, end]."""
        if end < start:
            return 0
        return int(sum(1 for d in pd.date_range(start, end) if d in self.open_days))

    # ------------------------------------------------------------------ start
    def seed_initial_stock(self, first_lines: pd.DataFrame, n_days: int, days_of_stock: float = 2.5) -> None:
        """Initial stock and use history, estimated from the first days of demand (only used in the warm-up)."""
        use: dict[str, float] = {}
        for item_id, qty in first_lines.groupby("item_id")["qty"].sum().items():
            for ing, g in self.recipe_gross.get(item_id, []):
                use[ing] = use.get(ing, 0.0) + g * qty / max(n_days, 1)
        start = self.all_days[0]
        for ing in self.ing_kind:
            u = use.get(ing, 0.0)
            self.hist[ing].extend([u] * self.hist[ing].maxlen)
            if u > 0:
                self._add_lot(ing, start, u * days_of_stock)

    # ------------------------------------------------------------------ one day
    def run_day(self, date: pd.Timestamp, lines: pd.DataFrame | None, record: bool = True) -> pd.DataFrame | None:
        """Simulate one calendar day. `lines` are the desired order lines of an open day (sorted by time).

        Returns the lines with a `sold_qty` column (None on closed days)."""
        is_open = date in self.open_days
        # count day: closed Monday, before anything else happens
        if date.weekday() == self.inv["count_weekday"] and record:
            self._count(date)
        # order placement (also on the closed day: the manager orders on Mondays)
        self._place_orders(date)
        if not is_open:
            return None
        self._receive(date)
        self._expire(date, record)
        sold = self._serve(date, lines, record)
        self._operational_waste(date, record)
        self._close_bottles(date, record)
        return sold

    def _place_orders(self, date: pd.Timestamp) -> None:
        for sup, info in self.suppliers.items():
            if date.weekday() not in info["order_days"]:
                continue
            arrival = self._arrival(date, sup)
            if arrival is None:
                continue
            next_arrival = self._arrival(self._next_order_day(date, sup), sup) or (arrival + pd.Timedelta(days=7))
            horizon = max(self._open_days_between(arrival, next_arrival - pd.Timedelta(days=1)), 1)
            until_arrival = self._open_days_between(date, arrival - pd.Timedelta(days=1))
            for ing in self.by_supplier.get(sup, []):
                avg = self.avg_use(ing)
                if avg <= 0:
                    continue
                incoming = sum(p["qty"] for p in self.pending if p["ing"] == ing and p["arrive"] <= arrival)
                projected = self.stock[ing] + incoming - avg * until_arrival
                target = self.policy.target(self, ing, horizon, arrival)
                qty = target - projected
                if qty <= 1e-9:
                    continue
                pack = self.ing_pack[ing]
                qty = math.ceil(qty / pack - 1e-9) * pack
                agreed = self._cost(ing, date)
                invoiced = agreed
                if self.rng.random() < self.settings["costs"]["invoice_error_prob"]:
                    invoiced = agreed * (1 + self.settings["costs"]["invoice_error_pct"])
                received = qty * (0.85 if self.rng.random() < 0.02 else 1.0)
                self.order_seq += 1
                row = {"order_id": f"PO{self.order_seq:06d}", "ordered_on": date, "supplier_id": sup, "ingredient_id": ing,
                       "qty_ordered": qty, "qty_received": round(received, 3), "agreed_price": round(agreed, 4),
                       "invoiced_price": round(invoiced, 4), "received_on": arrival}
                self.pending.append({"arrive": arrival, "ing": ing, "qty": received, "row": row})

    def _receive(self, date: pd.Timestamp) -> None:
        keep = []
        for p in self.pending:
            if p["arrive"] <= date:
                self._add_lot(p["ing"], date, p["qty"])
                self.purchases.append(p["row"])
            else:
                keep.append(p)
        self.pending = keep

    def _expire(self, date: pd.Timestamp, record: bool) -> None:
        wl = self.inv["waste"]["log_prob_expiry"]
        for ing, lots in self.lots.items():
            while lots and lots[0][0] <= date:
                exp_date, qty = lots.popleft()
                self.stock[ing] = max(0.0, self.stock[ing] - qty)
                if qty <= 1e-9:
                    continue
                cost = qty * self._cost(ing, date)
                self.physical_loss.append({"date": date, "ingredient_id": ing, "qty": qty, "cost": cost, "cause": "expiry"})
                if record and round(qty, 3) > 0 and self.rng.random() < wl:
                    self.waste.append({"date": date, "ingredient_id": ing, "qty": round(qty, 3), "reason": "expired", "cost": round(cost, 2)})

    def _glass(self, ing: str, date: pd.Timestamp) -> bool:
        """Pour one glass. Opens a new bottle if needed. Returns False if there is no wine left."""
        bottles = self.open_bottles[ing]
        for b in bottles:
            if b[1] > 0:
                b[1] -= 1
                b[3] += 1
                return True
        if self.stock[ing] < 1.0 - 1e-9:
            return False
        self._take(ing, 1.0)
        cap = max(3, int(round(self.rng.normal(self.inv["pours_actual_mean"], 0.5))))
        bottles.append([date, cap - 1, cap, 1])
        return True

    def _serve(self, date: pd.Timestamp, lines: pd.DataFrame, record: bool) -> pd.DataFrame:
        sold_qty = np.zeros(len(lines), dtype=int)
        day_use: dict[str, float] = {}
        day_missed: dict[str, float] = {}  # recipe use of the units we could not serve: the owner knows he ran out
        items = lines["item_id"].to_numpy()
        qtys = lines["qty"].to_numpy()
        void = lines["is_void"].to_numpy()
        for k in range(len(lines)):
            if void[k]:
                continue
            item_id = items[k]
            is_glass = self.items.at[item_id, "category"] == WINE_GLASS_CAT
            recipe = self.recipe_gross[item_id]
            sold = 0
            for _ in range(int(qtys[k])):
                if is_glass:
                    ing = recipe[0][0]
                    if not self._glass(ing, date):
                        day_missed[ing] = day_missed.get(ing, 0.0) + recipe[0][1]
                        continue
                    day_use[ing] = day_use.get(ing, 0.0) + recipe[0][1]
                else:
                    phys = [(ing, g * (1.0 + self.overuse[ing])) for ing, g in recipe]
                    if any(self.stock[ing] + 1e-9 < q for ing, q in phys):
                        for ing, g in recipe:
                            day_missed[ing] = day_missed.get(ing, 0.0) + g
                        continue
                    for ing, q in phys:
                        self._take(ing, q)
                    for (ing, q), (_, g) in zip(phys, recipe, strict=True):
                        day_use[ing] = day_use.get(ing, 0.0) + g
                        if q > g and record:
                            self.physical_loss.append({"date": date, "ingredient_id": ing, "qty": q - g,
                                                       "cost": (q - g) * self._cost(ing, date), "cause": "overuse"})
                sold += 1
            sold_qty[k] = sold
            if sold < qtys[k] and record:
                self.stockouts.append({"date": date, "line_time": lines["line_time"].iat[k], "item_id": item_id, "lost_qty": int(qtys[k] - sold)})
        for ing in self.ing_kind:
            self.hist[ing].append(day_use.get(ing, 0.0) + day_missed.get(ing, 0.0))
            if record and day_use.get(ing):
                self.usage_theory.append({"date": date, "ingredient_id": ing, "qty": day_use[ing]})
        out = lines.copy()
        out["sold_qty"] = sold_qty
        return out

    def _operational_waste(self, date: pd.Timestamp, record: bool) -> None:
        w = self.inv["waste"]
        n = int(self.rng.poisson(w["operational_events_per_day"]))
        if n == 0:
            return
        weights = np.array([max(self.avg_use(i), 0.0) * self._cost(i, date) if self.ing_kind[i] != "wine" else 0.0 for i in self.ing_kind])
        if weights.sum() <= 0:
            return
        ids = list(self.ing_kind)
        reasons = list(w["reasons"])
        rp = np.array([w["reasons"][r] for r in reasons])
        for _ in range(n):
            ing = ids[int(self.rng.choice(len(ids), p=weights / weights.sum()))]
            qty = max(self.avg_use(ing), 0.0) * self.rng.uniform(0.01, 0.06)
            if self.ing_unit[ing] == "ud":
                qty = max(1.0, round(qty))
            taken = self._take(ing, qty)
            if taken <= 1e-9:
                continue
            cost = taken * self._cost(ing, date)
            self.physical_loss.append({"date": date, "ingredient_id": ing, "qty": taken, "cost": cost, "cause": "operational"})
            if record and round(taken, 3) > 0 and self.rng.random() < w["log_prob_operational"]:
                reason = reasons[int(self.rng.choice(len(reasons), p=rp / rp.sum()))]
                self.waste.append({"date": date, "ingredient_id": ing, "qty": round(taken, 3), "reason": reason, "cost": round(cost, 2)})

    def _close_bottles(self, date: pd.Timestamp, record: bool) -> None:
        for ing, bottles in self.open_bottles.items():
            glass_days = self.inv["glass_days"][self.wine_type[ing]]
            keep = []
            for b in bottles:
                opened, left, cap, poured = b
                age = (date - opened).days
                if left <= 0 or age >= glass_days:
                    discarded = left > 0
                    if record:
                        self.bottles.append({"wine_ingredient_id": ing, "opened_on": opened, "closed_on": date, "glasses_poured": poured,
                                             "glasses_discarded": max(left, 0), "capacity_glasses": cap})
                    if discarded:
                        cost = self._cost(ing, date) * left / cap
                        self.physical_loss.append({"date": date, "ingredient_id": ing, "qty": left / cap, "cost": cost, "cause": "open_bottle"})
                        if record and round(left / cap, 3) > 0 and self.rng.random() < 0.5:
                            self.waste.append({"date": date, "ingredient_id": ing, "qty": round(left / cap, 3), "reason": "open_bottle_expired", "cost": round(cost, 2)})
                else:
                    keep.append(b)
            self.open_bottles[ing] = keep

    def _count(self, date: pd.Timestamp) -> None:
        noise = self.inv["count_noise"]
        for ing in self.ing_kind:
            true = self.stock[ing]
            if self.ing_kind[ing] == "wine" or self.ing_unit[ing] == "ud":
                counted = float(round(true * (1 + self.rng.normal(0, noise / 3))))
            else:
                counted = round(true * (1 + self.rng.normal(0, noise)), 2)
            self.counts.append({"count_date": date, "ingredient_id": ing, "counted_qty": max(counted, 0.0), "_true_qty": true})
