"""Master data (ingredients, items, recipes, suppliers) and its prices over time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINE_GLASS_CAT = "Vino por copa"
WINE_BOTTLE_CAT = "Vino botella"
PACK_DEFAULT = {"kg": 0.5, "l": 1.0, "ud": 1.0}


@dataclass
class Catalog:
    suppliers: pd.DataFrame
    ingredients: pd.DataFrame
    items: pd.DataFrame
    recipes: pd.DataFrame  # item_id, ingredient_id, qty_net (per serving), qty_gross (net / (1 - trim))


def build_catalog(menu: dict) -> Catalog:
    suppliers = pd.DataFrame(
        [{"supplier_id": k, "name": v["name"], "lead_days": v["lead_days"], "min_order_eur": v["min_order_eur"]}
         for k, v in menu["suppliers"].items()]
    )
    ing_rows = []
    for k, v in menu["ingredients"].items():
        ing_rows.append({
            "ingredient_id": k, "name": v["name"], "unit": v["unit"], "base_cost": float(v["cost"]),
            "shelf_days": int(v["shelf"]), "trim": float(v["trim"]), "supplier_id": v["supplier"],
            "kind": v["kind"], "pack": float(v.get("pack", PACK_DEFAULT[v["unit"]])),
        })
    item_rows, recipe_rows = [], []
    for k, v in menu["items"].items():
        item_rows.append({
            "item_id": k, "name": v["name"], "category": v["cat"], "base_price": float(v["price"]),
            "pop": float(v["pop"]), "elasticity": float(v["e"]), "wine_id": None,
            "daypart": v.get("daypart"),
        })
        for ing, q in v["recipe"].items():
            recipe_rows.append({"item_id": k, "ingredient_id": ing, "qty_net": float(q)})
    for wid, w in menu["wines"].items():
        bottle = f"bottle_{wid}"
        ing_rows.append({
            "ingredient_id": bottle, "name": f"{w['name']} (botella)", "unit": "ud", "base_cost": float(w["cost"]),
            "shelf_days": 3650, "trim": 0.0, "supplier_id": "S_VINO", "kind": "wine", "pack": 6.0, "wine_type": w["type"],
        })
        item_rows.append({
            "item_id": f"btl_{wid}", "name": f"{w['name']} (botella)", "category": WINE_BOTTLE_CAT,
            "base_price": float(w["bottle_price"]), "pop": float(w["pop"]), "elasticity": -0.4,
            "wine_id": bottle, "daypart": None,
        })
        recipe_rows.append({"item_id": f"btl_{wid}", "ingredient_id": bottle, "qty_net": 1.0})
        if w.get("by_glass"):
            item_rows.append({
                "item_id": f"glass_{wid}", "name": f"{w['name']} (copa)", "category": WINE_GLASS_CAT,
                "base_price": float(w["glass_price"]), "pop": float(w["pop"]), "elasticity": -0.5,
                "wine_id": bottle, "daypart": None,
            })
            recipe_rows.append({"item_id": f"glass_{wid}", "ingredient_id": bottle, "qty_net": 0.2})
    ingredients = pd.DataFrame(ing_rows)
    items = pd.DataFrame(item_rows)
    recipes = pd.DataFrame(recipe_rows).merge(ingredients[["ingredient_id", "trim"]], on="ingredient_id")
    recipes["qty_gross"] = recipes["qty_net"] / (1.0 - recipes["trim"])
    recipes = recipes.drop(columns="trim")
    return Catalog(suppliers=suppliers, ingredients=ingredients, items=items, recipes=recipes)


def price_matrix(catalog: Catalog, settings: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Menu price per day and item, after the price rounds. Index = dates, columns = item_id."""
    items = catalog.items
    factor = pd.DataFrame(1.0, index=dates, columns=items["item_id"])
    for rnd in settings["pricing"]["rounds"]:
        cats = set(rnd["categories"])
        cols = items.loc[items["category"].isin(cats), "item_id"]
        mask = dates >= pd.Timestamp(rnd["date"])
        factor.loc[mask, cols] *= 1.0 + rnd["change"]
    base = items.set_index("item_id")["base_price"]
    prices = factor * base
    is_bottle = (items.set_index("item_id")["category"] == WINE_BOTTLE_CAT)
    out = prices.copy()
    for col in prices.columns:
        out[col] = prices[col].round(0) if is_bottle[col] else (prices[col] * 10).round() / 10
    return out


def cost_matrix(catalog: Catalog, settings: dict, dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Purchase price per day and ingredient: monthly random walk plus named shocks. Index = dates."""
    c = settings["costs"]
    ing = catalog.ingredients
    months = dates.to_period("M")
    uniq = months.unique()
    out = {}
    for _, row in ing.iterrows():
        steps = rng.normal(c["monthly_drift"], c["monthly_sigma"], size=len(uniq))
        steps[0] = 0.0
        walk = pd.Series(np.exp(np.cumsum(steps)), index=uniq)
        daily = walk.reindex(months).to_numpy() * row["base_cost"]
        shock = c["shocks"].get(row["ingredient_id"])
        if shock:
            start, end = pd.Timestamp(shock["start"]), pd.Timestamp(shock["end"])
            ramp = np.clip((dates - start).days / max((end - start).days, 1), 0.0, 1.0)
            mult = 1.0 + shock["change"] * np.asarray(ramp)
            if "hold_until" in shock:
                mult = np.where(dates > pd.Timestamp(shock["hold_until"]), 1.0, mult)
            daily = daily * mult
        out[row["ingredient_id"]] = np.round(daily, 4)
    return pd.DataFrame(out, index=dates)
