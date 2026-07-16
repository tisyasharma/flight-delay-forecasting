"""
Fetches archived GFS forecast weather from Open-Meteo's historical-forecast API
for the aviation variables the ERA5 archive does not carry (visibility, CAPE,
lifted index, low cloud cover, gusts, freezing level), aggregates them to daily
operating-hour features per airport, and writes data/weather/aviation_daily.csv.

The GFS archive is used instead of the reanalysis archive because ERA5 returns
all-null for visibility, CAPE, lifted index, and freezing level height, and its
weather codes never encode freezing rain.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.training.fetch_weather_data import (
    AIRPORT_COORDS,
    DATE_END,
    extension_start,
)
from src.weather.common import aggregate_aviation_hourly_to_daily, drop_unsettled_tail

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEATHER_DIR = PROJECT_ROOT / "data" / "weather"
OUTPUT_PATH = WEATHER_DIR / "aviation_daily.csv"

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# pinned to the model family the live forecast API will use, so training and
# serving see the same weather distribution
FORECAST_MODEL = "gfs_seamless"

HOURLY_VARS = [
    "visibility",
    "cape",
    "lifted_index",
    "cloud_cover_low",
    "weather_code",
    "wind_gusts_10m",
    "wind_speed_10m",
    "freezing_level_height",
]

MAX_RETRIES = 5
BASE_DELAY = 2


def fetch_hourly_chunk(airport_code, lat, lon, start_date, end_date, timezone):
    """Fetches one airport-year chunk of hourly GFS history, with backoff."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": timezone,
        "models": FORECAST_MODEL,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            return pd.DataFrame({
                "datetime": pd.to_datetime(hourly.get("time", [])),
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

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = BASE_DELAY * (2 ** attempt)
                print(f"rate limited, waiting {wait_time}s...", end=" ")
                time.sleep(wait_time)
            else:
                raise e
        except requests.exceptions.RequestException as e:
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"request failed ({e}), retrying in {wait_time}s...", end=" ")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to fetch GFS history for {airport_code} after {MAX_RETRIES} retries"
    )


def year_chunks(start_date, end_date):
    """Splits a date range into per-year (start, end) string pairs."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    chunks = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((str(chunk_start.date()), str(chunk_end.date())))
    return chunks


def fetch_airport_history(airport_code, coords, start_date, end_date):
    """Fetches all year chunks for one airport and aggregates them to daily."""
    frames = []
    for chunk_start, chunk_end in year_chunks(start_date, end_date):
        frames.append(fetch_hourly_chunk(
            airport_code, coords["lat"], coords["lon"],
            chunk_start, chunk_end,
            timezone=coords.get("tz", "America/New_York"),
        ))
        time.sleep(2)

    hourly = pd.concat(frames, ignore_index=True)
    hourly = drop_unsettled_tail(hourly, ["weather_code", "visibility", "wind_speed"])
    return aggregate_aviation_hourly_to_daily(hourly)


def load_existing():
    """Loads the existing aviation CSV if present, for resuming partial fetches."""
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def fetch_all_aviation_weather(resume=True):
    """
    Fetches or extends GFS aviation history for all airports, checkpointing
    after each. Airports already current through DATE_END are skipped,
    airports with older rows are extended from the day after their last date.
    """
    existing_df = load_existing() if resume else None
    all_data = [existing_df] if existing_df is not None else []

    plans = {a: extension_start(existing_df, a) for a in AIRPORT_COORDS}
    current = [a for a, s in plans.items() if s is None]
    print(f"Aviation: {len(plans) - len(current)} airports to fetch or extend, "
          f"{len(current)} already current through {DATE_END}")

    failed = []
    for airport, coords in AIRPORT_COORDS.items():
        start = plans[airport]
        if start is None:
            continue
        print(f"{airport} ({coords['name']}) from {start}...", end=" ")

        try:
            daily = fetch_airport_history(airport, coords, start, DATE_END)
            all_data.append(daily)
            print(f"{len(daily)} days")

            # save after each airport in case something crashes
            temp_combined = pd.concat(all_data, ignore_index=True)
            temp_combined = temp_combined.sort_values(["airport", "date"]).reset_index(drop=True)
            temp_combined.to_csv(OUTPUT_PATH, index=False)

        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(airport)
            continue

        time.sleep(3)

    if not all_data:
        raise RuntimeError("No aviation weather data fetched")
    if failed:
        raise RuntimeError(f"aviation weather fetch failed for {', '.join(failed)}, "
                           "rerun to extend the remaining airports")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["airport", "date"])
    combined = combined.sort_values(["airport", "date"]).reset_index(drop=True)

    return combined


def main():
    """Fetches all airports and reports coverage per column."""
    df = fetch_all_aviation_weather(resume=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{len(df):,} daily records, {df['airport'].nunique()} airports")
    print(f"{df['date'].min()} to {df['date'].max()}")

    missing = set(AIRPORT_COORDS.keys()) - set(df["airport"].unique())
    if missing:
        print(f"Missing airports: {', '.join(sorted(missing))}")

    value_cols = [c for c in df.columns if c not in ("airport", "date")]
    print("\nNon-null rates:")
    for col in value_cols:
        print(f"  {col}: {df[col].notna().mean():.1%}")


if __name__ == "__main__":
    main()
