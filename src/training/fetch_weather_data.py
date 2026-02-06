import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

AIRPORT_COORDS = {
    "ANC": {"lat": 61.1744, "lon": -149.9964, "name": "Anchorage", "tz": "America/Anchorage"},
    "ATL": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta", "tz": "America/New_York"},
    "BOS": {"lat": 42.3656, "lon": -71.0096, "name": "Boston", "tz": "America/New_York"},
    "DCA": {"lat": 38.8512, "lon": -77.0402, "name": "Washington DC", "tz": "America/New_York"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "name": "Denver", "tz": "America/Denver"},
    "DFW": {"lat": 32.8998, "lon": -97.0403, "name": "Dallas-Fort Worth", "tz": "America/Chicago"},
    "EWR": {"lat": 40.6895, "lon": -74.1745, "name": "Newark", "tz": "America/New_York"},
    "FLL": {"lat": 26.0742, "lon": -80.1506, "name": "Fort Lauderdale", "tz": "America/New_York"},
    "HNL": {"lat": 21.3187, "lon": -157.9225, "name": "Honolulu", "tz": "Pacific/Honolulu"},
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York JFK", "tz": "America/New_York"},
    "KOA": {"lat": 19.7388, "lon": -156.0456, "name": "Kona", "tz": "Pacific/Honolulu"},
    "LAS": {"lat": 36.0840, "lon": -115.1537, "name": "Las Vegas", "tz": "America/Los_Angeles"},
    "LAX": {"lat": 33.9416, "lon": -118.4085, "name": "Los Angeles", "tz": "America/Los_Angeles"},
    "LGA": {"lat": 40.7769, "lon": -73.8740, "name": "New York LaGuardia", "tz": "America/New_York"},
    "LIH": {"lat": 21.9760, "lon": -159.3390, "name": "Lihue", "tz": "Pacific/Honolulu"},
    "MCO": {"lat": 28.4312, "lon": -81.3081, "name": "Orlando", "tz": "America/New_York"},
    "MIA": {"lat": 25.7959, "lon": -80.2870, "name": "Miami", "tz": "America/New_York"},
    "OGG": {"lat": 20.8986, "lon": -156.4305, "name": "Maui", "tz": "Pacific/Honolulu"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare", "tz": "America/Chicago"},
    "PHX": {"lat": 33.4373, "lon": -112.0078, "name": "Phoenix", "tz": "America/Phoenix"},
    "SEA": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle", "tz": "America/Los_Angeles"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "name": "San Francisco", "tz": "America/Los_Angeles"},
    "SLC": {"lat": 40.7899, "lon": -111.9791, "name": "Salt Lake City", "tz": "America/Denver"},
}

DATE_START = "2019-01-01"
DATE_END = "2025-06-30"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

MAX_RETRIES = 5
BASE_DELAY = 2


def fetch_weather_for_airport(airport_code, lat, lon, start_date, end_date, timezone="America/New_York"):
    """Fetches daily weather from Open-Meteo for one airport, retries on rate limit."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code"
        ],
        "timezone": timezone
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            df = pd.DataFrame({
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
                "weather_code": daily.get("weather_code")
            })
            return df

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = BASE_DELAY * (2 ** attempt)
                print(f"rate limited, waiting {wait_time}s...", end=" ")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError(f"Failed to fetch weather for {airport_code} after {MAX_RETRIES} retries")


def weather_code_to_condition(code):
    """Maps WMO weather code to a readable condition string."""
    if code is None or pd.isna(code):
        return "clear"

    code = int(code)

    if code == 0:
        return "clear"
    elif code in [1, 2, 3]:
        return "cloudy"
    elif code in [45, 48]:
        return "fog"
    elif code in [51, 53]:
        return "drizzle"
    elif code in [55]:
        return "dense_drizzle"
    elif code in [56, 57]:
        return "freezing_drizzle"
    elif code in [61, 63]:
        return "rain"
    elif code in [65]:
        return "heavy_rain"
    elif code in [66, 67]:
        return "freezing_rain"
    elif code in [80]:
        return "rain_showers"
    elif code in [81, 82]:
        return "heavy_showers"
    elif code in [71, 73, 75, 77]:
        return "snow"
    elif code in [85, 86]:
        return "snow_showers"
    elif code in [95, 96, 99]:
        return "thunderstorm"
    else:
        return "other"


def condition_to_severity(condition):
    """Converts condition string to a 0-5 severity score."""
    severity_map = {
        "clear": 0,
        "cloudy": 1,
        "fog": 2,
        "drizzle": 2,
        "other": 2,
        "dense_drizzle": 3,
        "rain": 3,
        "rain_showers": 3,
        "heavy_rain": 4,
        "heavy_showers": 4,
        "freezing_drizzle": 4,
        "freezing_rain": 4,
        "snow": 4,
        "snow_showers": 4,
        "thunderstorm": 5
    }
    return severity_map.get(condition, 2)


def process_weather_data(df):
    """Adds condition labels, severity, and binary weather flags."""
    df["condition"] = df["weather_code"].apply(weather_code_to_condition)
    df["severity"] = df["condition"].apply(condition_to_severity)

    df["precip_total"] = df["precip_total"].fillna(0)
    df["rain"] = df["rain"].fillna(0)
    df["snowfall"] = df["snowfall"].fillna(0)

    df["has_precipitation"] = (df["precip_total"] > 0.1).astype(int)
    df["has_snow"] = (df["snowfall"] > 0).astype(int)
    df["is_adverse"] = (df["severity"] >= 3).astype(int)

    df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df


def fetch_hourly_weather_for_airport(airport_code, lat, lon, start_date, end_date, timezone="America/New_York"):
    """Fetches hourly weather from Open-Meteo for one airport, retries on rate limit."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "snowfall",
            "wind_speed_10m",
            "wind_gusts_10m",
            "weather_code"
        ],
        "timezone": timezone
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly.get("time", [])),
                "airport": airport_code,
                "temp": hourly.get("temperature_2m"),
                "precip": hourly.get("precipitation"),
                "snowfall": hourly.get("snowfall"),
                "wind_speed": hourly.get("wind_speed_10m"),
                "wind_gusts": hourly.get("wind_gusts_10m"),
                "weather_code": hourly.get("weather_code")
            })
            return df

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = BASE_DELAY * (2 ** attempt)
                print(f"rate limited, waiting {wait_time}s...", end=" ")
                time.sleep(wait_time)
            else:
                raise e

    raise RuntimeError(f"Failed to fetch hourly weather for {airport_code} after {MAX_RETRIES} retries")


def aggregate_hourly_to_daily(hourly_df):
    """Computes operating-hour-aware daily aggregates from hourly weather data."""
    hourly_df["date"] = hourly_df["datetime"].dt.date
    hourly_df["hour"] = hourly_df["datetime"].dt.hour

    hourly_df["hourly_condition"] = hourly_df["weather_code"].apply(weather_code_to_condition)
    hourly_df["hourly_severity"] = hourly_df["hourly_condition"].apply(condition_to_severity)

    operating_mask = (hourly_df["hour"] >= 6) & (hourly_df["hour"] <= 23)
    morning_mask = (hourly_df["hour"] >= 5) & (hourly_df["hour"] <= 10)
    evening_mask = (hourly_df["hour"] >= 16) & (hourly_df["hour"] <= 21)

    groups = hourly_df.groupby(["airport", "date"])
    operating_groups = hourly_df[operating_mask].groupby(["airport", "date"])
    morning_groups = hourly_df[morning_mask].groupby(["airport", "date"])
    evening_groups = hourly_df[evening_mask].groupby(["airport", "date"])

    daily_agg = pd.DataFrame()
    daily_agg["peak_wind_operating"] = operating_groups["wind_speed"].max()
    daily_agg["precip_operating"] = operating_groups["precip"].sum()
    daily_agg["max_hourly_severity"] = groups["hourly_severity"].max()
    daily_agg["storm_hours"] = groups["hourly_severity"].apply(lambda x: (x >= 4).sum())
    daily_agg["morning_severity"] = morning_groups["hourly_severity"].max()
    daily_agg["evening_severity"] = evening_groups["hourly_severity"].max()

    daily_agg = daily_agg.reset_index()
    daily_agg["date"] = pd.to_datetime(daily_agg["date"])

    for col in ["peak_wind_operating", "precip_operating", "max_hourly_severity",
                "storm_hours", "morning_severity", "evening_severity"]:
        daily_agg[col] = daily_agg[col].fillna(0)

    return daily_agg


def load_existing_weather():
    """Loads existing weather CSV if it exists, for resuming partial fetches."""
    output_path = WEATHER_DIR / "weather_daily.csv"
    if output_path.exists():
        df = pd.read_csv(output_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def fetch_all_weather(resume=True):
    """Fetches weather for all airports, saves after each one in case it crashes."""
    all_data = []
    existing_airports = set()

    if resume:
        existing_df = load_existing_weather()
        if existing_df is not None:
            existing_airports = set(existing_df["airport"].unique())
            all_data.append(existing_df)
            print(f"Found existing data for {len(existing_airports)} airports")

    airports_to_fetch = {k: v for k, v in AIRPORT_COORDS.items() if k not in existing_airports}
    print(f"{len(airports_to_fetch)} airports to fetch, {len(existing_airports)} already done")

    for airport, coords in airports_to_fetch.items():
        print(f"{airport} ({coords['name']})...", end=" ")

        try:
            df = fetch_weather_for_airport(
                airport, coords["lat"], coords["lon"],
                DATE_START, DATE_END,
                timezone=coords.get("tz", "America/New_York")
            )
            df = process_weather_data(df)
            all_data.append(df)
            print(f"{len(df)} days")

            # save after each airport in case something crashes
            temp_combined = pd.concat(all_data, ignore_index=True)
            temp_combined = temp_combined.sort_values(["airport", "date"]).reset_index(drop=True)
            temp_combined.to_csv(WEATHER_DIR / "weather_daily.csv", index=False)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        time.sleep(3)

    if not all_data:
        raise RuntimeError("No weather data fetched")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["airport", "date"])
    combined = combined.sort_values(["airport", "date"]).reset_index(drop=True)

    return combined


def load_existing_hourly():
    """Loads existing hourly-derived daily CSV if it exists."""
    output_path = WEATHER_DIR / "weather_hourly_agg.csv"
    if output_path.exists():
        df = pd.read_csv(output_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def fetch_all_hourly_weather(resume=True):
    """Fetches hourly weather for all airports and aggregates to daily features."""
    all_data = []
    existing_airports = set()

    if resume:
        existing_df = load_existing_hourly()
        if existing_df is not None:
            existing_airports = set(existing_df["airport"].unique())
            all_data.append(existing_df)
            print(f"Found existing hourly data for {len(existing_airports)} airports")

    airports_to_fetch = {k: v for k, v in AIRPORT_COORDS.items() if k not in existing_airports}
    print(f"Hourly: {len(airports_to_fetch)} airports to fetch, {len(existing_airports)} already done")

    for airport, coords in airports_to_fetch.items():
        print(f"  {airport} ({coords['name']}) hourly...", end=" ")

        try:
            hourly_df = fetch_hourly_weather_for_airport(
                airport, coords["lat"], coords["lon"],
                DATE_START, DATE_END,
                timezone=coords.get("tz", "America/New_York")
            )
            daily_agg = aggregate_hourly_to_daily(hourly_df)
            all_data.append(daily_agg)
            print(f"{len(daily_agg)} days")

            temp_combined = pd.concat(all_data, ignore_index=True)
            temp_combined = temp_combined.sort_values(["airport", "date"]).reset_index(drop=True)
            temp_combined.to_csv(WEATHER_DIR / "weather_hourly_agg.csv", index=False)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        time.sleep(5)

    if not all_data:
        print("No hourly weather data fetched")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["airport", "date"])
    combined = combined.sort_values(["airport", "date"]).reset_index(drop=True)

    return combined


def merge_hourly_into_daily(daily_df, hourly_agg_df):
    """Merges hourly-derived daily aggregates into the main daily weather CSV."""
    hourly_cols = [
        "peak_wind_operating", "precip_operating", "max_hourly_severity",
        "storm_hours", "morning_severity", "evening_severity"
    ]

    # drop these columns from daily if they already exist (from a previous merge)
    for col in hourly_cols:
        if col in daily_df.columns:
            daily_df = daily_df.drop(columns=[col])

    merged = daily_df.merge(
        hourly_agg_df[["airport", "date"] + hourly_cols],
        on=["airport", "date"],
        how="left"
    )

    for col in hourly_cols:
        merged[col] = merged[col].fillna(0)

    return merged


def main():
    """Fetches all airport weather (daily + hourly) and saves to CSV."""
    weather_df = fetch_all_weather(resume=True)

    output_path = WEATHER_DIR / "weather_daily.csv"
    weather_df.to_csv(output_path, index=False)

    print(f"\n{len(weather_df):,} daily records, {weather_df['airport'].nunique()} airports")
    print(f"{weather_df['date'].min()} to {weather_df['date'].max()}")

    missing = set(AIRPORT_COORDS.keys()) - set(weather_df["airport"].unique())
    if missing:
        print(f"Missing daily: {', '.join(sorted(missing))}")

    # fetch and merge hourly-derived features
    print("\nFetching hourly weather...")
    hourly_agg = fetch_all_hourly_weather(resume=True)

    if hourly_agg is not None:
        weather_df = merge_hourly_into_daily(weather_df, hourly_agg)
        weather_df.to_csv(output_path, index=False)
        print(f"Merged hourly features into {output_path}")
        print(f"Final columns: {list(weather_df.columns)}")


if __name__ == "__main__":
    main()
