"""
Weather assembly for live forecasting: covers every day from the last BTS
actual through the forecast horizon by stitching three sources in order of
preference: the era5 archive and the gfs_seamless historical-forecast
archive for settled days (each lags realtime by a few days), and the live
forecast API for the recent seam and the future. The live API is pinned to
models=gfs_seamless and serves the training variables under identical names,
so every frame this module produces matches the training weather schema and
flows through the same aggregation code the training fetchers use.
"""

import time

import pandas as pd
import requests

from src.training.fetch_weather_data import (
    AIRPORT_COORDS,
    fetch_hourly_weather_for_airport,
    fetch_weather_for_airport,
    merge_hourly_into_daily,
)
from src.weather.common import (
    aggregate_aviation_hourly_to_daily,
    aggregate_hourly_to_daily,
    process_weather_data,
)
from src.weather.gfs_history import HOURLY_VARS, fetch_hourly_chunk

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "gfs_seamless"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "weather_code",
]
BASIC_HOURLY_VARS = [
    "temperature_2m", "precipitation", "snowfall",
    "wind_speed_10m", "wind_gusts_10m", "weather_code",
]

MAX_RETRIES = 5
BASE_DELAY = 2


def fetch_live_forecast(airport_code, lat, lon, timezone, past_days=10, forecast_days=8):
    """
    One live-API call per airport carrying the daily block, the basic hourly
    block, and the aviation hourly block together. Returns the raw payload.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "hourly": BASIC_HOURLY_VARS + HOURLY_VARS,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone,
        "models": FORECAST_MODEL,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(FORECAST_URL, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait = BASE_DELAY * (2 ** attempt)
            print(f"live fetch {airport_code} failed ({e}), retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"live forecast fetch failed for {airport_code}")


def live_payload_to_frames(payload, airport_code):
    """
    Splits a live-API payload into the three training-schema frames: processed
    daily weather, hourly-derived daily aggregates, and aviation aggregates.
    """
    daily = payload.get("daily", {})
    daily_df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "airport": airport_code,
        "temp_max": daily.get("temperature_2m_max"),
        "temp_min": daily.get("temperature_2m_min"),
        "temp_avg": daily.get("temperature_2m_mean"),
        "precip_total": daily.get("precipitation_sum"),
        "rain": daily.get("rain_sum"),
        "snowfall": daily.get("snowfall_sum"),
        "wind_speed_max": daily.get("wind_speed_10m_max"),
        "wind_gusts_max": daily.get("wind_gusts_10m_max"),
        "weather_code": daily.get("weather_code"),
    })
    daily_df = process_weather_data(daily_df)

    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    basic_hourly = pd.DataFrame({
        "datetime": times,
        "airport": airport_code,
        "temp": hourly.get("temperature_2m"),
        "precip": hourly.get("precipitation"),
        "snowfall": hourly.get("snowfall"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "wind_gusts": hourly.get("wind_gusts_10m"),
        "weather_code": hourly.get("weather_code"),
    })
    hourly_agg = aggregate_hourly_to_daily(basic_hourly)

    aviation_hourly = pd.DataFrame({
        "datetime": times,
        "airport": airport_code,
        "visibility": hourly.get("visibility"),
        "cape": hourly.get("cape"),
        "lifted_index": hourly.get("lifted_index"),
        "cloud_cover_low": hourly.get("cloud_cover_low"),
        "weather_code": hourly.get("weather_code"),
        "wind_gusts": hourly.get("wind_gusts_10m"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "freezing_level": hourly.get("freezing_level_height"),
    })
    aviation_df = aggregate_aviation_hourly_to_daily(aviation_hourly)

    daily_df = merge_hourly_into_daily(daily_df, hourly_agg)
    return daily_df, aviation_df


def fetch_archive_frames(airport_code, coords, start, end):
    """
    Settled-history frames from the archive sources for one airport: era5
    daily+hourly merged, and gfs_seamless aviation aggregates. Days past the
    archives' recency lag come back NaN and are dropped here, the live API
    covers them.
    """
    tz = coords.get("tz", "America/New_York")
    daily = process_weather_data(fetch_weather_for_airport(
        airport_code, coords["lat"], coords["lon"], start, end, timezone=tz))
    hourly = fetch_hourly_weather_for_airport(
        airport_code, coords["lat"], coords["lon"], start, end, timezone=tz)
    daily = merge_hourly_into_daily(daily, aggregate_hourly_to_daily(hourly))
    daily = daily.dropna(subset=["temp_max"])

    aviation = fetch_hourly_chunk(
        airport_code, coords["lat"], coords["lon"], start, end, tz)
    aviation = aggregate_aviation_hourly_to_daily(aviation)
    aviation = aviation.dropna(subset=["gust_max"])
    return daily, aviation


def build_weather_span(start, end, airports=None, sleep_s=1.5):
    """
    Weather and aviation frames covering [start, end] for every airport:
    archive rows preferred where settled, live-API rows fill the seam and the
    future. Raises if any airport-day in the span ends up uncovered.
    """
    airports = airports or AIRPORT_COORDS
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    # forecast_days counts from each airport's LOCAL today, and at the 09:30 UTC
    # cron, Hawaii is still on the previous local date, so a horizon computed
    # from the UTC date needs a day of slack or the last day comes back
    # missing there and the coverage check aborts every scheduled run
    forecast_days = max(1, min(16, int((end_ts - today).days) + 2))

    weather_frames, aviation_frames = [], []
    for airport, coords in airports.items():
        arc_daily, arc_aviation = (pd.DataFrame(), pd.DataFrame())
        if start_ts < today:
            arc_daily, arc_aviation = fetch_archive_frames(
                airport, coords, str(start_ts.date()), str(min(end_ts, today).date()))

        payload = fetch_live_forecast(
            airport, coords["lat"], coords["lon"],
            coords.get("tz", "America/New_York"), forecast_days=forecast_days)
        live_daily, live_aviation = live_payload_to_frames(payload, airport)

        daily = pd.concat([arc_daily, live_daily], ignore_index=True)
        daily = daily.drop_duplicates(subset=["airport", "date"], keep="first")
        aviation = pd.concat([arc_aviation, live_aviation], ignore_index=True)
        aviation["date"] = pd.to_datetime(aviation["date"])
        aviation = aviation.drop_duplicates(subset=["airport", "date"], keep="first")

        daily = daily[(daily["date"] >= start_ts) & (daily["date"] <= end_ts)]
        aviation = aviation[(aviation["date"] >= start_ts) & (aviation["date"] <= end_ts)]

        # the abort-on-uncovered policy applies to both frames: a missing
        # aviation day would otherwise sail through the left join as NaN and
        # be filled benign, which is a silent lie rather than a loud failure
        expected = (end_ts - start_ts).days + 1
        for label, frame in (("weather", daily), ("aviation", aviation)):
            if len(frame) < expected:
                have = set(frame["date"])
                missing = [d for d in pd.date_range(start_ts, end_ts) if d not in have]
                raise RuntimeError(f"{airport}: {label} uncovered for {missing[:4]}")

        weather_frames.append(daily)
        aviation_frames.append(aviation)
        time.sleep(sleep_s)

    return (pd.concat(weather_frames, ignore_index=True),
            pd.concat(aviation_frames, ignore_index=True))
