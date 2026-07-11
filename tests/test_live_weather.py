"""
Tests for the live weather assembly's coverage contract: every airport-day
in the requested span must be present or the run aborts loudly, and the
live-API request must carry enough forecast days that an airport sitting
behind UTC (Hawaii at the morning cron) still covers the horizon's last day.
Fetches are stubbed; the payload-to-frames path runs for real.
"""

import pandas as pd
import pytest

from src.pipelines import live_weather


def make_payload(start, n_days):
    """A live-API payload covering n_days from start, benign weather."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    hours = pd.date_range(start, periods=n_days * 24, freq="h")
    return {
        "daily": {
            "time": [str(d.date()) for d in dates],
            "temperature_2m_max": [10.0] * n_days,
            "temperature_2m_min": [2.0] * n_days,
            "temperature_2m_mean": [6.0] * n_days,
            "precipitation_sum": [0.0] * n_days,
            "rain_sum": [0.0] * n_days,
            "snowfall_sum": [0.0] * n_days,
            "wind_speed_10m_max": [15.0] * n_days,
            "wind_gusts_10m_max": [25.0] * n_days,
            "weather_code": [1] * n_days,
        },
        "hourly": {
            "time": [h.isoformat() for h in hours],
            "temperature_2m": [6.0] * len(hours),
            "precipitation": [0.0] * len(hours),
            "snowfall": [0.0] * len(hours),
            "wind_speed_10m": [10.0] * len(hours),
            "wind_gusts_10m": [20.0] * len(hours),
            "weather_code": [1] * len(hours),
            "visibility": [20000.0] * len(hours),
            "cape": [0.0] * len(hours),
            "lifted_index": [5.0] * len(hours),
            "cloud_cover_low": [10.0] * len(hours),
            "freezing_level_height": [3000.0] * len(hours),
        },
    }


AIRPORTS = {"TST": {"lat": 0.0, "lon": 0.0, "tz": "UTC"}}


def today_utc():
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def test_build_weather_span_covers_future_span(monkeypatch):
    today = today_utc()
    start, end = today, today + pd.Timedelta(days=2)
    seen = {}

    def fake_fetch(airport_code, lat, lon, timezone, past_days=10, forecast_days=8):
        seen["forecast_days"] = forecast_days
        return make_payload(today, forecast_days)

    monkeypatch.setattr(live_weather, "fetch_live_forecast", fake_fetch)
    weather, aviation = live_weather.build_weather_span(
        start, end, airports=AIRPORTS, sleep_s=0)

    assert len(weather) == 3 and len(aviation) == 3
    # one day of slack past the horizon: forecast_days counts from the
    # airport's local today, which can trail the UTC date
    assert seen["forecast_days"] == (end - today).days + 2


def test_build_weather_span_aborts_on_uncovered_day(monkeypatch):
    today = today_utc()
    start, end = today, today + pd.Timedelta(days=4)

    def short_fetch(airport_code, lat, lon, timezone, past_days=10, forecast_days=8):
        return make_payload(today, (end - today).days - 1)

    monkeypatch.setattr(live_weather, "fetch_live_forecast", short_fetch)
    with pytest.raises(RuntimeError, match="uncovered"):
        live_weather.build_weather_span(start, end, airports=AIRPORTS, sleep_s=0)
