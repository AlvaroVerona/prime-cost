"""Daily demand (guests) forecast for the next 7 days, evaluated out of time.

Direct multi-horizon design: every day is a possible forecast origin O. For each target day D (1 to 7 days ahead) the features are
only what was known at O: the calendar and weather of D, the recent level, the average of the same weekday over the last
weeks, and the horizon. Models are compared with two honest baselines on the last weeks of data that were never used to fit
anything (rolling-origin, refit every week).

At test time the weather features are NOT the true weather: noise is added to mimic a real weather forecast.
Run: `python -m prime_cost.forecasting.demand`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from prime_cost.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, load_settings

CAL_FEATURES = ["weekday", "month", "doy_sin", "doy_cos", "holiday", "tasting", "temp_c", "rain", "trend"]
ORIGIN_FEATURES = ["level", "dow_mean4", "dow_mean8", "h"]


def load_open_days(raw_dir=RAW_DIR) -> pd.DataFrame:
    cal = pd.read_parquet(raw_dir / "calendar.parquet")
    cal = cal[cal["open"]].copy().sort_values("date").reset_index(drop=True)
    cal["doy_sin"] = np.sin(2 * np.pi * cal["date"].dt.dayofyear / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * cal["date"].dt.dayofyear / 365.25)
    cal["month"] = cal["date"].dt.month
    cal["trend"] = (cal["date"] - cal["date"].min()).dt.days
    for c in ("holiday", "tasting", "rain"):
        cal[c] = cal[c].astype(int)
    return cal


def build_samples(cal: pd.DataFrame, horizon: int, level_days: int) -> pd.DataFrame:
    """One row per (origin, target day): features known at the origin and the realized covers."""
    cov = cal.set_index("date")["covers"]
    rows = []
    dates = cal["date"].tolist()
    for origin in pd.date_range(dates[0] + pd.Timedelta(days=level_days), dates[-1] - pd.Timedelta(days=1)):
        past = cov[(cov.index < origin) & (cov.index >= origin - pd.Timedelta(days=level_days))]
        if len(past) < 10:
            continue
        level = past.mean()
        targets = cal[(cal["date"] > origin) & (cal["date"] <= origin + pd.Timedelta(days=horizon))]
        for _, t in targets.iterrows():
            same = cov[(cov.index < origin) & (cov.index.weekday == t["date"].weekday())]
            if len(same) < 4:
                continue
            rows.append({"origin": origin, "date": t["date"], "h": (t["date"] - origin).days, "level": level,
                         "dow_mean4": same.tail(4).mean(), "dow_mean8": same.tail(8).mean(), "covers": t["covers"],
                         **{c: t[c] for c in CAL_FEATURES}})
    return pd.DataFrame(rows)


def noisy_weather(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    out["temp_c"] = out["temp_c"] + rng.normal(0, cfg["temp_sigma"], len(out))
    flip = rng.random(len(out)) < cfg["rain_flip_prob"]
    out["rain"] = np.where(flip, 1 - out["rain"], out["rain"])
    return out


def _fit(name: str, train: pd.DataFrame, gbm: dict):
    """Models predict covers / recent level (a multiplier), so growth and season level are handled by the level."""
    y = train["covers"] / train["level"]
    if name == "ridge":
        X = _ridge_design(train)
        return Ridge(alpha=3.0).fit(X, y)
    X = train[CAL_FEATURES + ORIGIN_FEATURES]
    return HistGradientBoostingRegressor(max_iter=gbm["max_iter"], learning_rate=gbm["learning_rate"], max_depth=gbm["max_depth"],
                                         min_samples_leaf=gbm["min_samples_leaf"], random_state=0).fit(X, y)


def _ridge_design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df["weekday"].astype(int), prefix="wd", dtype=float)
    for k in range(7):
        if f"wd_{k}" not in X:
            X[f"wd_{k}"] = 0.0
    X = X[[f"wd_{k}" for k in range(7)]]
    for c in ["holiday", "tasting", "rain", "doy_sin", "doy_cos"]:
        X[c] = df[c].to_numpy()
    X["temp"] = df["temp_c"].to_numpy() / 10.0
    X["dow_ratio"] = (df["dow_mean4"] / df["level"]).to_numpy()
    X["rain_x_temp"] = (df["rain"] * df["temp_c"]).to_numpy() / 10.0
    return X


def _predict(name: str, model, df: pd.DataFrame) -> np.ndarray:
    if name == "ridge":
        return model.predict(_ridge_design(df)) * df["level"].to_numpy()
    return model.predict(df[CAL_FEATURES + ORIGIN_FEATURES]) * df["level"].to_numpy()


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    return {"wape": round(float(np.abs(err).sum() / actual.sum()), 4), "mae": round(float(np.abs(err).mean()), 2),
            "bias_pct": round(float(err.sum() / actual.sum()), 4), "n": len(actual)}


def evaluate(samples: pd.DataFrame, holdout_start: pd.Timestamp, cfg: dict, seed: int = 0):
    """Rolling-origin evaluation on the holdout: refit each week on everything known before that week."""
    rng = np.random.default_rng(seed)
    test = samples[samples["date"] >= holdout_start].copy()
    test = noisy_weather(test, cfg["weather_forecast_error"], rng)
    test["pred_baseline_dow4"] = test["dow_mean4"]
    # baseline 2: recent level times the historical weekday profile (profile fitted on training data only)
    out_parts = []
    weeks = sorted(test["origin"].dt.to_period("W-SUN").unique())
    for wk in weeks:
        te = test[test["origin"].dt.to_period("W-SUN") == wk].copy()
        cut = te["origin"].min()
        train = samples[samples["date"] < cut]  # target realized before the week starts, so its origin is earlier too
        if len(train) < cfg["min_train_days"]:
            continue
        prof = (train["covers"] / train["level"]).groupby(train["weekday"]).mean()
        te["pred_level_profile"] = te["level"] * te["weekday"].map(prof)
        for name in ("ridge", "gbm"):
            te[f"pred_{name}"] = _predict(name, _fit(name, train, cfg["gbm"]), te)
        out_parts.append(te)
    ev = pd.concat(out_parts, ignore_index=True)
    return ev


def run(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR, settings: dict | None = None, processed_dir=PROCESSED_DIR) -> dict:
    s = settings or load_settings()
    cfg = s["analysis"]["forecast"]
    cal = load_open_days(raw_dir)
    samples = build_samples(cal, cfg["horizon_days"], cfg["level_days"])
    holdout_start = pd.Timestamp(s["period"]["end"]) - pd.Timedelta(days=s["holdout_days"] - 1)
    ev = evaluate(samples, holdout_start, cfg)
    preds = ["pred_baseline_dow4", "pred_level_profile", "pred_ridge", "pred_gbm"]
    res = {p.replace("pred_", ""): metrics(ev["covers"].to_numpy(), ev[p].to_numpy()) for p in preds}
    by_h = {p.replace("pred_", ""): {int(h): metrics(g["covers"].to_numpy(), g[p].to_numpy())["wape"] for h, g in ev.groupby("h")} for p in preds}
    best = min(res, key=lambda k: res[k]["wape"])
    # prediction interval for the chosen model: calibrate the multiplicative error on the FIRST half of the evaluation period,
    # then measure the coverage on the SECOND half (never on the data the interval was fitted on)
    mid = ev["date"].min() + (ev["date"].max() - ev["date"].min()) / 2
    calib, check = ev[ev["date"] <= mid], ev[ev["date"] > mid]
    q = (1 - cfg["interval"]) / 2
    lo_q, hi_q = np.quantile((calib["covers"] / calib[f"pred_{best}"]).to_numpy(), [q, 1 - q])
    ev["forecast"] = ev[f"pred_{best}"]
    ev["lower"] = ev["forecast"] * lo_q
    ev["upper"] = ev["forecast"] * hi_q
    check = ev.loc[check.index]
    coverage = float(((check["covers"] >= check["lower"]) & (check["covers"] <= check["upper"])).mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=["origin"]).assign(origin=ev["origin"]).to_csv(output_dir / "forecast_eval.csv", index=False)
    summary = {"models": res, "wape_by_horizon": by_h, "best_model": best, "interval": cfg["interval"], "interval_coverage_out_of_sample": round(coverage, 3),
               "holdout_days": int(ev["date"].nunique()), "improvement_vs_baseline_pct": round(1 - res[best]["wape"] / res["baseline_dow4"]["wape"], 4)}
    (output_dir / "forecast_metrics.json").write_text(json.dumps(summary, indent=2))
    # daily forecasts (all origins, all models) for the downstream optimizers
    processed_dir.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(processed_dir / "forecast_holdout.parquet", index=False)
    print({k: v["wape"] for k, v in res.items()}, "best:", best, f"| interval coverage {coverage:.0%}")
    return {"eval": ev, "summary": summary, "samples": samples}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
