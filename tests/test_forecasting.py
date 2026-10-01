import numpy as np
import pandas as pd
import pytest

from prime_cost.forecasting import demand


@pytest.fixture(scope="module")
def cal(pipeline):
    return demand.load_open_days(pipeline.raw)


def test_origin_features_use_only_the_past(cal):
    s = demand.build_samples(cal, horizon=7, level_days=28)
    row = s.iloc[len(s) // 2]
    cov = cal.set_index("date")["covers"]
    same = cov[(cov.index < row["origin"]) & (cov.index.weekday == row["date"].weekday())]
    assert row["dow_mean4"] == pytest.approx(same.tail(4).mean())
    assert row["date"] > row["origin"]
    assert (s["date"] > s["origin"]).all() and s["h"].between(1, 7).all()


def test_metrics():
    m = demand.metrics(np.array([100.0, 100.0]), np.array([110.0, 90.0]))
    assert m["wape"] == pytest.approx(0.10) and m["bias_pct"] == pytest.approx(0.0)


def test_noisy_weather_changes_only_weather_columns(cal):
    df = demand.build_samples(cal, 7, 28).head(200)
    noisy = demand.noisy_weather(df, {"temp_sigma": 2.0, "rain_flip_prob": 0.3}, np.random.default_rng(0))
    assert (noisy["covers"] == df["covers"]).all()
    assert not (noisy["temp_c"] == df["temp_c"]).all()
    assert set(noisy["rain"].unique()) <= {0, 1}


def test_rolling_origin_never_trains_on_the_test_week(pipeline):
    s = pipeline.settings
    cfg = {**s["analysis"]["forecast"], "min_train_days": 60}
    samples = demand.build_samples(demand.load_open_days(pipeline.raw), cfg["horizon_days"], cfg["level_days"])
    hold = pd.Timestamp(s["period"]["end"]) - pd.Timedelta(days=s["holdout_days"] - 1)
    ev = demand.evaluate(samples, hold, cfg)
    assert len(ev) > 0 and {"pred_gbm", "pred_ridge", "pred_baseline_dow4", "pred_level_profile"} <= set(ev.columns)
    assert (ev["date"] >= hold).all()
    assert np.isfinite(ev[["pred_gbm", "pred_ridge"]].to_numpy()).all()
