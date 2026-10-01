"""Calendar, weather and the demand model: how many guests come each day, and what they order."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prime_cost.data.catalog import WINE_BOTTLE_CAT, WINE_GLASS_CAT, Catalog

HOURS = list(range(13, 25))  # 24 = the hour after midnight, still part of the same business day


def daypart_of(hour: int, dayparts: dict) -> str:
    for name, hours in dayparts.items():
        if hour in hours:
            return name
    return "evening"


def build_calendar(settings: dict, rng: np.random.Generator) -> pd.DataFrame:
    """One row per calendar day, from `warmup_days` before the period start to its end."""
    p = settings["period"]
    start = pd.Timestamp(p["start"]) - pd.Timedelta(days=p["warmup_days"])
    dates = pd.date_range(start, p["end"], freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["weekday"] = cal["date"].dt.weekday
    cal["month"] = cal["date"].dt.month
    mmdd = cal["date"].dt.strftime("%m-%d")
    cfg_c = settings["calendar"]
    cal["open"] = ~(cal["weekday"].isin(cfg_c["closed_weekdays"]) | mmdd.isin(cfg_c["closed_dates"]))
    cal["holiday"] = mmdd.isin(cfg_c["holidays"])

    w = settings["weather"]
    doy = cal["date"].dt.dayofyear.to_numpy()
    seasonal = w["temp_mean"] + w["temp_amplitude"] * np.cos(2 * np.pi * (doy - w["temp_peak_doy"]) / 365.25)
    noise = np.zeros(len(cal))
    eps = rng.normal(0, w["temp_sigma"], len(cal))
    for i in range(len(cal)):  # AR(1): weather is persistent from one day to the next
        noise[i] = (0.7 * noise[i - 1] if i else 0.0) + eps[i] * np.sqrt(1 - 0.49)
    cal["temp_c"] = np.round(seasonal + noise, 1)
    rain_p = cal["month"].map(w["rain_prob_by_month"]).to_numpy()
    cal["rain"] = rng.random(len(cal)) < rain_p

    # first Wednesday (configurable) of each month: wine tasting night
    t = settings["demand"]["tasting"]
    day_of_month = cal["date"].dt.day
    cal["tasting"] = (cal["weekday"] == t["weekday"]) & (((day_of_month - 1) // 7 + 1) == t["week_of_month"])

    d = settings["demand"]
    months_since = (cal["date"].dt.year - dates[0].year) * 12 + (cal["month"] - dates[0].month)
    base = cal["weekday"].map(d["base_covers"]).astype(float).fillna(0.0)
    expected = (
        base
        * cal["month"].map(d["seasonality"]).to_numpy()
        * (1 + d["growth_per_month"]) ** months_since.to_numpy()
        * np.where(cal["holiday"], cfg_c["holiday_uplift"], 1.0)
        * np.where(cal["tasting"], 1 + t["uplift"], 1.0)
        * np.where(cal["rain"], 1 - d["rain_penalty"], 1.0)
    )
    cal["expected_covers"] = np.where(cal["open"], expected, 0.0)
    sigma = d["noise_sigma"]
    shock = np.exp(rng.normal(0, sigma, len(cal)) - sigma**2 / 2)
    cal["covers"] = np.where(cal["open"], np.rint(cal["expected_covers"] * shock), 0).astype(int)

    te = d["terrace"]
    share = np.clip((cal["temp_c"] - te["temp_min"]) / (te["temp_full"] - te["temp_min"]), 0, 1) * te["max_share"]
    cal["terrace_share"] = np.where(cal["rain"], 0.0, share).round(3)
    return cal


def _hour_weights(settings: dict, weekday: int) -> np.ndarray:
    hw = settings["demand"]["hour_weights"]
    key = "sun" if weekday == 6 else ("fri_sat" if weekday in (4, 5) else "weekday")
    w = np.array([hw[key][h] for h in HOURS], dtype=float)
    return w / w.sum()


def _price_effect(catalog: Catalog, prices_day: pd.Series) -> pd.Series:
    items = catalog.items.set_index("item_id")
    return (prices_day / items["base_price"]) ** items["elasticity"]


def generate_demand(cal: pd.DataFrame, catalog: Catalog, prices: pd.DataFrame, settings: dict, rng: np.random.Generator):
    """Tickets and the lines guests WANT to order (before stock availability is applied)."""
    b = settings["basket"]
    d = settings["demand"]
    dayparts = settings["calendar"]["dayparts"]
    items = catalog.items.set_index("item_id")
    cats = {c: items.index[items["category"] == c].to_numpy() for c in items["category"].unique()}
    food_cat, bev_cat, des_cat, cof_cat = "Tapas", "Bebidas", "Postres", "Cafe"
    sizes = np.array(sorted(d["party_sizes"]))
    size_p = np.array([d["party_sizes"][s] for s in sizes], dtype=float)
    size_p /= size_p.sum()
    hour_to_part = {h: daypart_of(h, dayparts) for h in HOURS}
    part_names = list(dayparts)

    def dp_mult(item_id: str, part: str) -> float:
        dp = items.at[item_id, "daypart"]
        return float(dp[part]) if isinstance(dp, dict) else 1.0

    tickets_parts, lines_parts = [], []
    tid = 0
    for _, day in cal[cal["open"]].iterrows():
        date = day["date"]
        pe = _price_effect(catalog, prices.loc[date])
        # --- tickets of the day: party sizes until the day's covers are reached
        covers = int(day["covers"])
        draw = rng.choice(sizes, size=max(covers, 1), p=size_p)
        cs = np.cumsum(draw)
        n = int(np.searchsorted(cs, covers) + 1)
        party = draw[:n].copy()
        party[-1] -= cs[n - 1] - covers
        party = party[party > 0]
        n_in = len(party)
        hours = rng.choice(HOURS, size=n_in, p=_hour_weights(settings, int(day["weekday"])))
        # delivery orders (one guest each)
        season = d["seasonality"][int(day["month"])]
        n_del = int(rng.poisson(d["delivery"]["orders_per_day"].get(int(day["weekday"]), 0) * season))
        del_hours = rng.choice(HOURS, size=n_del, p=_hour_weights(settings, 1)) if n_del else np.array([], dtype=int)
        cov = np.concatenate([party, np.ones(n_del, dtype=int)])
        hour = np.concatenate([hours, del_hours]).astype(int)
        is_del = np.concatenate([np.zeros(n_in, bool), np.ones(n_del, bool)])
        n_t = len(cov)
        if n_t == 0:
            continue
        minute = rng.integers(0, 60, n_t)
        channel = np.where(is_del, "delivery", "sala")
        u = rng.random(n_t)
        channel = np.where(~is_del & (u < day["terrace_share"]), "terraza", channel)
        channel = np.where(~is_del & (channel == "sala") & (cov <= 3) & (rng.random(n_t) < d["barra_share"]), "barra", channel)
        part = np.array([hour_to_part[h] for h in hour])

        # --- how many of each kind of thing, per ticket
        def cat_factor(cat: str) -> float:
            sub = items.loc[cats[cat]]
            w = sub["pop"] / sub["pop"].sum()
            return float((w * pe.loc[sub.index]).sum())

        f = {c: cat_factor(c) for c in cats}
        cvr = cov.astype(float)
        n_food = np.maximum(1, rng.poisson((b["food_per_cover"] * cvr + b["food_base"]) * f[food_cat]))
        n_food = np.where(is_del, 1 + rng.poisson(settings["demand"]["delivery"]["extra_items_mean"], n_t), n_food)
        n_bev = rng.poisson(b["drinks_per_cover"] * cvr * f[bev_cat] * np.where(is_del, 0.6, 1.0))
        n_gl = np.where(is_del, 0, rng.poisson(b["glasses_per_cover"] * cvr * f[WINE_GLASS_CAT]))
        p_btl = np.where(cov >= 2, b["bottle_prob_pair"], b["bottle_prob_solo"]) * f[WINE_BOTTLE_CAT]
        has_btl = rng.random(n_t) < np.where(is_del, 0.03, p_btl)
        extra = (cov >= 4) & has_btl & (rng.random(n_t) < b["extra_bottle_prob_large"])
        n_btl = has_btl.astype(int) + extra.astype(int)
        des_m = np.array([np.mean([dp_mult(i, pp) for i in cats[des_cat]]) for pp in part_names])
        cof_m = np.array([np.mean([dp_mult(i, pp) for i in cats[cof_cat]]) for pp in part_names])
        pi = {pp: k for k, pp in enumerate(part_names)}
        part_idx = np.array([pi[x] for x in part])
        n_des = rng.poisson(b["dessert_per_cover"] * cvr * des_m[part_idx] * f[des_cat] * np.where(is_del, 1.4, 1.0))
        n_cof = np.where(is_del, 0, rng.poisson(b["coffee_per_cover"] * cvr * cof_m[part_idx] * f[cof_cat]))

        counts = {food_cat: n_food, bev_cat: n_bev, WINE_GLASS_CAT: n_gl, WINE_BOTTLE_CAT: n_btl, des_cat: n_des, cof_cat: n_cof}
        t_idx_all, item_all = [], []
        for cat, cnt in counts.items():
            ids = cats[cat]
            for pp in part_names:
                tsel = np.where(part == pp)[0]
                if len(tsel) == 0:
                    continue
                reps = cnt[tsel]
                total = int(reps.sum())
                if total == 0:
                    continue
                w = np.array([items.at[i, "pop"] * dp_mult(i, pp) * pe[i] for i in ids], dtype=float)
                chosen = rng.choice(ids, size=total, p=w / w.sum())
                t_idx_all.append(np.repeat(tsel, reps))
                item_all.append(chosen)
        if not t_idx_all:
            continue
        t_idx = np.concatenate(t_idx_all)
        item = np.concatenate(item_all)
        line = pd.DataFrame({"t": t_idx, "item_id": item})
        line = line.groupby(["t", "item_id"], as_index=False).size().rename(columns={"size": "qty"})
        offset = rng.integers(0, 45, len(line))
        line["minute_off"] = offset
        line["unit_price"] = prices.loc[date, line["item_id"]].to_numpy()
        line["discount_pct"] = np.where(rng.random(len(line)) < b["discount"]["prob"], b["discount"]["pct"], 0.0)
        line["is_void"] = rng.random(len(line)) < b["void_prob"]
        ticket_ids = np.array([f"T{tid + k + 1:08d}" for k in range(n_t)])
        line["ticket_id"] = ticket_ids[line["t"].to_numpy()]
        opened = date + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
        line["line_time"] = opened[line["t"].to_numpy()] + pd.to_timedelta(line["minute_off"], unit="m")
        line["date"] = date
        lines_parts.append(line.drop(columns=["t", "minute_off"]))
        card = np.where(is_del, True, rng.random(n_t) < settings["payments"]["card_share"])
        tickets_parts.append(pd.DataFrame({
            "ticket_id": ticket_ids, "date": date, "opened_at": opened, "hour": hour, "daypart": part,
            "channel": channel, "covers": cov, "pay_method": np.where(card, "card", "cash"),
        }))
        tid += n_t

    tickets = pd.concat(tickets_parts, ignore_index=True)
    lines = pd.concat(lines_parts, ignore_index=True)
    lines = lines.sort_values(["line_time", "ticket_id"]).reset_index(drop=True)
    return tickets, lines
