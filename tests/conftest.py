"""
Shared fixtures backed by a synthetic 3-route x 400-day dataset. Tests never
read anything from data/, so the suite runs anywhere the package installs.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.build_features import FeatureBuilder
from src.features.registry import TABULAR_FEATURES

TRAIN_END = "2023-11-01"

_ROUTES = ["AAA-BBB", "CCC-DDD", "EEE-FFF"]
_ROUTE_DELAY_OFFSET = {"AAA-BBB": 5.0, "CCC-DDD": 18.0, "EEE-FFF": 35.0}
_ROUTE_FLIGHT_MEAN = {"AAA-BBB": 40.0, "CCC-DDD": 90.0, "EEE-FFF": 150.0}

_SEVERITY_LEVELS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_SEVERITY_WEIGHTS = np.array([0.45, 0.20, 0.15, 0.10, 0.06, 0.04])


def _build_synthetic():
    """
    Builds the shared synthetic base with every input column FeatureBuilder
    consumes. The target mixes a per-route level, a weekly sinusoid, and a
    weather-severity term so simple models have real signal to find.
    Post-train temperatures are shifted +15 and visibility floors -6000, and
    both windows get ~5% NaNs, so an imputation value computed on the full
    data provably differs from the train-window median. Every route spans
    every date, a route with fewer than two train rows would export a None
    std in the feature state.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    day_of_week = dates.dayofweek.to_numpy()
    n = len(dates)

    frames = []
    for route in _ROUTES:
        apt1_severity = rng.choice(_SEVERITY_LEVELS, size=n, p=_SEVERITY_WEIGHTS)
        apt2_severity = rng.choice(_SEVERITY_LEVELS, size=n, p=_SEVERITY_WEIGHTS)
        severity_max = np.maximum(apt1_severity, apt2_severity)

        delay = (
            _ROUTE_DELAY_OFFSET[route]
            + 6.0 * np.sin(2 * np.pi * day_of_week / 7)
            + 2.5 * severity_max
            + rng.normal(0.0, 3.0, size=n)
        )
        flights = (
            _ROUTE_FLIGHT_MEAN[route]
            + 8.0 * (day_of_week < 5)
            + rng.normal(0.0, 4.0, size=n)
        )

        apt1_precip = rng.gamma(1.0, 2.0, size=n)
        apt2_precip = rng.gamma(1.0, 2.0, size=n)
        apt1_snow = rng.gamma(0.3, 1.0, size=n)
        apt2_snow = rng.gamma(0.3, 1.0, size=n)
        apt1_wind = rng.gamma(4.0, 4.0, size=n)
        apt2_wind = rng.gamma(4.0, 4.0, size=n)
        max_wind = np.maximum(apt1_wind, apt2_wind)
        total_precip = apt1_precip + apt2_precip
        hourly_severity = rng.choice(_SEVERITY_LEVELS, size=n, p=_SEVERITY_WEIGHTS)

        apt1_vis_min = rng.uniform(2000.0, 24000.0, size=n)
        apt2_vis_min = rng.uniform(2000.0, 24000.0, size=n)
        apt1_gust = apt1_wind * rng.uniform(1.1, 1.8, size=n)
        apt2_gust = apt2_wind * rng.uniform(1.1, 1.8, size=n)
        apt1_fzra = rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 3.0], size=n)
        apt2_fzra = rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 3.0], size=n)

        frames.append(pd.DataFrame({
            "date": dates,
            "route": route,
            "avg_arr_delay": delay,
            "flight_count": flights,
            "apt1_severity": apt1_severity,
            "apt2_severity": apt2_severity,
            "weather_severity_max": severity_max,
            "weather_severity_combined": apt1_severity + apt2_severity,
            "has_adverse_weather": (severity_max >= 3).astype(float),
            "apt1_temp_avg": rng.normal(55.0, 12.0, size=n),
            "apt2_temp_avg": rng.normal(60.0, 12.0, size=n),
            "apt1_precip_total": apt1_precip,
            "apt2_precip_total": apt2_precip,
            "apt1_snowfall": apt1_snow,
            "apt2_snowfall": apt2_snow,
            "apt1_wind_speed_max": apt1_wind,
            "apt2_wind_speed_max": apt2_wind,
            "total_precip": total_precip,
            "total_snowfall": apt1_snow + apt2_snow,
            "max_wind": max_wind,
            "peak_wind_operating": max_wind * rng.uniform(0.7, 1.0, size=n),
            "precip_operating": total_precip * rng.uniform(0.4, 1.0, size=n),
            "max_hourly_severity": np.maximum(severity_max, hourly_severity),
            "storm_hours": rng.choice([0.0, 0.0, 0.0, 1.0, 2.0, 4.0], size=n),
            "morning_severity": rng.choice([0.0, 0.0, 1.0, 2.0, 3.0], size=n),
            "evening_severity": rng.choice([0.0, 0.0, 1.0, 2.0, 3.0], size=n),
            "apt1_visibility_min": apt1_vis_min,
            "apt2_visibility_min": apt2_vis_min,
            "apt1_visibility_p10": apt1_vis_min * rng.uniform(1.0, 1.4, size=n),
            "apt2_visibility_p10": apt2_vis_min * rng.uniform(1.0, 1.4, size=n),
            "apt1_cape_max": rng.gamma(0.8, 400.0, size=n),
            "apt2_cape_max": rng.gamma(0.8, 400.0, size=n),
            "apt1_gust_max": apt1_gust,
            "apt2_gust_max": apt2_gust,
            "apt1_gust_spread": apt1_gust - apt1_wind,
            "apt2_gust_spread": apt2_gust - apt2_wind,
            "apt1_low_cloud_hours": rng.choice(np.arange(0.0, 13.0), size=n),
            "apt2_low_cloud_hours": rng.choice(np.arange(0.0, 13.0), size=n),
            "apt1_freezing_rain_hours": apt1_fzra,
            "apt2_freezing_rain_hours": apt2_fzra,
            "has_freezing_rain": ((apt1_fzra > 0) | (apt2_fzra > 0)).astype(float),
        }))

    df = pd.concat(frames, ignore_index=True)

    post_train = df["date"] >= pd.Timestamp(TRAIN_END)
    df.loc[post_train, ["apt1_temp_avg", "apt2_temp_avg"]] += 15.0

    # a post-train visibility regime shift makes a full-data median leak
    # detectable, same trick as the temperature shift above
    vis_cols = [
        "apt1_visibility_min", "apt2_visibility_min",
        "apt1_visibility_p10", "apt2_visibility_p10",
    ]
    df.loc[post_train, vis_cols] = (df.loc[post_train, vis_cols] - 6000.0).clip(lower=200.0)

    for col in ["apt1_temp_avg", "apt2_temp_avg"] + vis_cols:
        for mask in (~post_train, post_train):
            idx = df.index[mask].to_numpy()
            drop = rng.choice(idx, size=int(len(idx) * 0.05), replace=False)
            df.loc[drop, col] = np.nan

    severity_cols = [
        "apt1_severity", "apt2_severity", "weather_severity_max",
        "weather_severity_combined", "has_adverse_weather",
    ]
    for col in severity_cols:
        drop = rng.choice(df.index.to_numpy(), size=6, replace=False)
        df.loc[drop, col] = np.nan

    for col in ["total_precip", "max_wind", "storm_hours", "morning_severity",
                "apt1_cape_max", "apt2_gust_max", "apt1_freezing_rain_hours"]:
        drop = rng.choice(df.index.to_numpy(), size=8, replace=False)
        df.loc[drop, col] = np.nan

    return df


_SYNTHETIC_BASE = _build_synthetic()


@pytest.fixture
def synthetic_df():
    """Fresh copy per test, several tests mutate rows in place."""
    return _SYNTHETIC_BASE.copy()


@pytest.fixture
def train_end():
    """Train cutoff used across the suite, 304 train days and 96 test days."""
    return TRAIN_END


@pytest.fixture
def built(synthetic_df, train_end):
    """Features built with the standard train cutoff, returns (features, builder)."""
    builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    features = builder.build()
    return features, builder


@pytest.fixture(scope="session")
def tiny_lgbm():
    """
    20-tree LightGBM fit on the train window, returns (model, features).
    Tests share the returned frame, mutations would leak between them.
    """
    features = FeatureBuilder(_SYNTHETIC_BASE.copy(), train_end_date=TRAIN_END).build()
    train = features[features["date"] < pd.Timestamp(TRAIN_END)]
    model = lgb.LGBMRegressor(
        n_estimators=20,
        num_leaves=15,
        learning_rate=0.2,
        min_child_samples=10,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train[TABULAR_FEATURES], train["avg_arr_delay"])
    return model, features
