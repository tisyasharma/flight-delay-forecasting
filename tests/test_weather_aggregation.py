"""
Tests for the aviation hourly-to-daily aggregation: operating-hour gating,
threshold semantics, and the visibility NaN contract (missing must never
read as zero visibility).
"""

import numpy as np
import pandas as pd

from src.weather.common import (
    FREEZING_RAIN_CODES,
    LOW_CLOUD_THRESHOLD,
    aggregate_aviation_hourly_to_daily,
)


def make_hourly_day(airport="AAA", date="2023-01-15", **overrides):
    """Builds one airport-day of 24 hourly rows with benign defaults."""
    df = pd.DataFrame({
        "datetime": pd.date_range(date, periods=24, freq="h"),
        "airport": airport,
        "visibility": 20000.0,
        "cape": 0.0,
        "lifted_index": 5.0,
        "cloud_cover_low": 10.0,
        "weather_code": 0,
        "wind_gusts": 20.0,
        "wind_speed": 10.0,
        "freezing_level": 3000.0,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_visibility_gated_to_operating_hours():
    df = make_hourly_day()
    # dense fog at 3am only; operating hours (6-23) stay clear
    df.loc[df["datetime"].dt.hour == 3, "visibility"] = 100.0

    agg = aggregate_aviation_hourly_to_daily(df)

    assert agg.loc[0, "visibility_min"] == 20000.0


def test_visibility_p10_matches_operating_quantile():
    df = make_hourly_day()
    operating_values = np.linspace(1000, 18000, 18)
    df.loc[df["datetime"].dt.hour >= 6, "visibility"] = operating_values

    agg = aggregate_aviation_hourly_to_daily(df)

    expected = pd.Series(operating_values).quantile(0.1)
    assert np.isclose(agg.loc[0, "visibility_p10"], expected)
    assert np.isclose(agg.loc[0, "visibility_min"], 1000.0)


def test_freezing_rain_hours_count_all_24_hours():
    df = make_hourly_day()
    hours = df["datetime"].dt.hour
    # one occurrence outside operating hours must still count
    df.loc[hours == 3, "weather_code"] = 66
    df.loc[hours == 12, "weather_code"] = 67
    df.loc[hours == 13, "weather_code"] = 56
    df.loc[hours == 14, "weather_code"] = 61  # plain rain, not freezing

    agg = aggregate_aviation_hourly_to_daily(df)

    assert agg.loc[0, "freezing_rain_hours"] == 3
    assert 61 not in FREEZING_RAIN_CODES


def test_low_cloud_hours_threshold_is_inclusive_and_operating_gated():
    df = make_hourly_day()
    hours = df["datetime"].dt.hour
    df.loc[hours == 2, "cloud_cover_low"] = 100.0  # outside operating hours
    df.loc[hours == 10, "cloud_cover_low"] = float(LOW_CLOUD_THRESHOLD)
    df.loc[hours == 11, "cloud_cover_low"] = LOW_CLOUD_THRESHOLD - 0.5

    agg = aggregate_aviation_hourly_to_daily(df)

    assert agg.loc[0, "low_cloud_hours"] == 1


def test_gust_spread_uses_same_operating_window():
    df = make_hourly_day()
    hours = df["datetime"].dt.hour
    df.loc[hours == 10, "wind_gusts"] = 45.0
    df.loc[hours == 15, "wind_speed"] = 18.0
    # a bigger gust at 1am must not leak into the operating-hour max
    df.loc[hours == 1, "wind_gusts"] = 90.0

    agg = aggregate_aviation_hourly_to_daily(df)

    assert agg.loc[0, "gust_max"] == 45.0
    assert agg.loc[0, "gust_spread"] == 27.0


def test_missing_visibility_stays_nan_while_counts_fill_zero():
    df = make_hourly_day(visibility=np.nan, cape=np.nan)

    agg = aggregate_aviation_hourly_to_daily(df)

    assert pd.isna(agg.loc[0, "visibility_min"])
    assert pd.isna(agg.loc[0, "visibility_p10"])
    assert agg.loc[0, "cape_max"] == 0.0


def test_all_none_object_columns_do_not_raise():
    # uncovered archive periods return every value as None, which pandas
    # builds as object columns; aggregation must degrade to NaN, not raise
    df = make_hourly_day()
    for col in ["visibility", "cape", "weather_code", "lifted_index"]:
        df[col] = [None] * len(df)

    agg = aggregate_aviation_hourly_to_daily(df)

    assert pd.isna(agg.loc[0, "visibility_min"])
    assert agg.loc[0, "cape_max"] == 0.0
    assert agg.loc[0, "freezing_rain_hours"] == 0


def test_grouping_by_airport_and_date():
    frames = [
        make_hourly_day(airport="AAA", date="2023-01-15"),
        make_hourly_day(airport="AAA", date="2023-01-16"),
        make_hourly_day(airport="BBB", date="2023-01-15"),
    ]
    df = pd.concat(frames, ignore_index=True)

    agg = aggregate_aviation_hourly_to_daily(df)

    assert len(agg) == 3
    assert set(zip(agg["airport"], agg["date"].dt.strftime("%Y-%m-%d"))) == {
        ("AAA", "2023-01-15"),
        ("AAA", "2023-01-16"),
        ("BBB", "2023-01-15"),
    }
