"""
Fetches historical weather data from Open-Meteo API for all airports.
Date range: 2019-01-01 to 2025-06-30
"""

import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WEATHER_DIR = DATA_DIR / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

# coordinates for the 15 airports in our dataset
AIRPORT_COORDS = {
    "ATL": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta"},
    "BOS": {"lat": 42.3656, "lon": -71.0096, "name": "Boston"},
    "DCA": {"lat": 38.8512, "lon": -77.0402, "name": "Washington DC"},
    "DEN": {"lat": 39.8561, "lon": -104.6737, "name": "Denver"},
    "FLL": {"lat": 26.0742, "lon": -80.1506, "name": "Fort Lauderdale"},
    "HNL": {"lat": 21.3187, "lon": -157.9225, "name": "Honolulu"},
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York JFK"},
    "LAS": {"lat": 36.0840, "lon": -115.1537, "name": "Las Vegas"},
    "LAX": {"lat": 33.9416, "lon": -118.4085, "name": "Los Angeles"},
    "LGA": {"lat": 40.7769, "lon": -73.8740, "name": "New York LaGuardia"},
    "MCO": {"lat": 28.4312, "lon": -81.3081, "name": "Orlando"},
    "OGG": {"lat": 20.8986, "lon": -156.4305, "name": "Maui"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare"},
    "PHX": {"lat": 33.4373, "lon": -112.0078, "name": "Phoenix"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "name": "San Francisco"}
}

DATE_START = "2019-01-01"
DATE_END = "2025-06-30"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# retry settings for rate limiting
MAX_RETRIES = 5
BASE_DELAY = 2


def fetch_weather_for_airport(airport_code, lat, lon, start_date, end_date):
    """
    Pulls daily weather for one airport from Open-Meteo.
    Retries with exponential backoff if we get rate-limited.
    """
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
        "timezone": "America/New_York"
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

    raise Exception(f"Failed after {MAX_RETRIES} retries")


def weather_code_to_condition(code):
    """
    Maps WMO weather codes to simple condition categories.
    Reference: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
    """
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
    """
    Turns weather conditions into a 0-5 severity scale based on flight impact.

    0 = clear, no issues
    1 = cloudy, minimal impact
    2 = fog or light drizzle, some visibility issues
    3 = rain, moderate delays
    4 = heavy rain, freezing precip, snow, significant delays
    5 = thunderstorm, ground stops likely

    Freezing rain/drizzle is rated 4 because ice on aircraft is dangerous
    and often grounds flights completely.
    """
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
    """Adds derived features like condition labels and severity scores."""
    df["condition"] = df["weather_code"].apply(weather_code_to_condition)
    df["severity"] = df["condition"].apply(condition_to_severity)

    # fill nulls with 0 for precipitation fields
    df["precip_total"] = df["precip_total"].fillna(0)
    df["rain"] = df["rain"].fillna(0)
    df["snowfall"] = df["snowfall"].fillna(0)

    # binary flags for quick filtering
    df["has_precipitation"] = (df["precip_total"] > 0.1).astype(int)
    df["has_snow"] = (df["snowfall"] > 0).astype(int)
    df["is_adverse"] = (df["severity"] >= 3).astype(int)

    df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df


def load_existing_weather():
    """Loads previously fetched weather data if it exists (for resuming)."""
    output_path = WEATHER_DIR / "weather_daily.csv"
    if output_path.exists():
        df = pd.read_csv(output_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def fetch_all_weather(resume=True):
    """
    Fetches weather for all airports.
    Saves progress after each airport so we can resume if something fails.
    """
    all_data = []
    existing_airports = set()

    if resume:
        existing_df = load_existing_weather()
        if existing_df is not None:
            existing_airports = set(existing_df["airport"].unique())
            all_data.append(existing_df)
            print(f"Resuming: found existing data for {len(existing_airports)} airports")

    airports_to_fetch = {k: v for k, v in AIRPORT_COORDS.items() if k not in existing_airports}

    print(f"Fetching weather data from {DATE_START} to {DATE_END}")
    print(f"Airports to fetch: {len(airports_to_fetch)} (skipping {len(existing_airports)} already fetched)")

    for airport, coords in airports_to_fetch.items():
        print(f"Fetching {airport} ({coords['name']})...", end=" ")

        try:
            df = fetch_weather_for_airport(
                airport,
                coords["lat"],
                coords["lon"],
                DATE_START,
                DATE_END
            )
            df = process_weather_data(df)
            all_data.append(df)
            print(f"{len(df)} days")

            # save after each airport so we can resume if it crashes
            temp_combined = pd.concat(all_data, ignore_index=True)
            temp_combined = temp_combined.sort_values(["airport", "date"]).reset_index(drop=True)
            temp_combined.to_csv(WEATHER_DIR / "weather_daily.csv", index=False)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        time.sleep(3)

    if not all_data:
        raise RuntimeError("No weather data fetched successfully")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["airport", "date"])
    combined = combined.sort_values(["airport", "date"]).reset_index(drop=True)

    return combined


def main():
    print("Fetching historical weather data...")

    weather_df = fetch_all_weather(resume=True)

    output_path = WEATHER_DIR / "weather_daily.csv"
    weather_df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Records: {len(weather_df):,}")
    print(f"Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
    print(f"Airports: {weather_df['airport'].nunique()}")

    missing = set(AIRPORT_COORDS.keys()) - set(weather_df["airport"].unique())
    if missing:
        print(f"\nMissing airports: {', '.join(sorted(missing))}")
        print("Run the script again to retry fetching missing airports.")

    print("\nSeverity Distribution:")
    severity_counts = weather_df.groupby("severity").size()
    for level, count in severity_counts.items():
        pct = count / len(weather_df) * 100
        print(f"  Level {level}: {count:,} ({pct:.1f}%)")

    print("\nCondition Distribution:")
    condition_counts = weather_df.groupby("condition").size().sort_values(ascending=False)
    for condition, count in condition_counts.items():
        pct = count / len(weather_df) * 100
        print(f"  {condition}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
