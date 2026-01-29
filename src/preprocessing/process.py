"""
Processes raw BTS flight data into daily route-level demand.
Picks top 20 routes, aggregates daily, and merges weather data.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
WEATHER_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "weather"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "FL_DATE",
    "REPORTING_AIRLINE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED",
    "DISTANCE"
]


def load_raw_data():
    """Load all raw CSV files into a single DataFrame."""
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    print(f"Loading {len(csv_files)} CSV files...")

    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading"):
        df = pd.read_csv(
            csv_file,
            usecols=lambda c: c in REQUIRED_COLS,
            low_memory=False
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} total flight records")

    return df


def clean_data(df):
    """Clean and preprocess the raw data."""
    print("Cleaning data...")

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    # some BTS files use OP_UNIQUE_CARRIER instead of REPORTING_AIRLINE
    if "REPORTING_AIRLINE" not in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "REPORTING_AIRLINE"})

    df["ORIGIN"] = df["ORIGIN"].astype(str).str.strip().str.upper()
    df["DEST"] = df["DEST"].astype(str).str.strip().str.upper()
    df["REPORTING_AIRLINE"] = df["REPORTING_AIRLINE"].astype(str).str.strip().str.upper()

    for col in ["DEP_DELAY", "ARR_DELAY", "DISTANCE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CANCELLED"] = df["CANCELLED"].fillna(0).astype(int)
    df = df.dropna(subset=["FL_DATE", "ORIGIN", "DEST"])

    # directional routes since LAX->JFK has different patterns than JFK->LAX
    df["route"] = df["ORIGIN"] + "-" + df["DEST"]

    print(f"Cleaned data: {len(df):,} records")
    return df


def identify_top_routes(df, n=20):
    """Identify the top N busiest routes by total flight count."""
    print(f"Identifying top {n} routes...")

    route_counts = df.groupby("route").size().reset_index(name="flight_count")
    route_counts = route_counts.sort_values("flight_count", ascending=False)

    top_routes = route_counts.head(n)["route"].tolist()

    print("Top routes by flight count:")
    for i, route in enumerate(top_routes, 1):
        count = route_counts[route_counts["route"] == route]["flight_count"].values[0]
        print(f"  {i}. {route}: {count:,} flights")

    return top_routes


def aggregate_daily(df, top_routes):
    """Aggregate data to daily route-level metrics."""
    print("Aggregating to daily route metrics...")

    df_filtered = df[df["route"].isin(top_routes)].copy()

    daily = df_filtered.groupby(["FL_DATE", "route"]).agg(
        flight_count=("ORIGIN", "size"),
        cancelled_count=("CANCELLED", "sum"),
        avg_dep_delay=("DEP_DELAY", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
        total_distance=("DISTANCE", "sum"),
        carriers=("REPORTING_AIRLINE", lambda x: ",".join(sorted(x.unique())))
    ).reset_index()

    daily["cancel_rate"] = daily["cancelled_count"] / daily["flight_count"]
    daily = daily.rename(columns={"FL_DATE": "date"})
    daily = daily.sort_values(["route", "date"]).reset_index(drop=True)

    print(f"Daily aggregated data: {len(daily):,} records")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    return daily


def add_route_metadata(daily):
    """Add route-level metadata."""
    print("Adding route metadata...")

    route_stats = daily.groupby("route").agg(
        avg_daily_flights=("flight_count", "mean"),
        total_flights=("flight_count", "sum"),
        avg_distance=("total_distance", lambda x: (x / daily.loc[x.index, "flight_count"]).mean())
    ).reset_index()

    daily = daily.merge(route_stats, on="route", how="left")
    return daily


def fill_missing_dates(daily):
    """Fill in missing dates with zero flights."""
    print("Filling missing dates...")

    routes = daily["route"].unique()
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    full_index = pd.MultiIndex.from_product(
        [date_range, routes],
        names=["date", "route"]
    )
    full_df = pd.DataFrame(index=full_index).reset_index()

    daily = full_df.merge(daily, on=["date", "route"], how="left")

    daily["flight_count"] = daily["flight_count"].fillna(0).astype(int)
    daily["cancelled_count"] = daily["cancelled_count"].fillna(0).astype(int)

    for col in ["avg_dep_delay", "avg_arr_delay", "cancel_rate"]:
        daily[col] = daily[col].fillna(0)

    route_meta = daily.groupby("route")[["avg_daily_flights", "total_flights", "avg_distance"]].first()
    for col in route_meta.columns:
        daily[col] = daily["route"].map(route_meta[col])

    print(f"Final dataset: {len(daily):,} records")
    return daily


def load_weather_data():
    """Load weather data."""
    weather_path = WEATHER_DATA_DIR / "weather_daily.csv"
    print(f"Loading weather data from {weather_path}...")

    weather = pd.read_csv(weather_path)
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def merge_weather_data(daily, weather):
    """Merge weather for both airports (apt1=origin, apt2=destination)."""
    print("Merging weather data...")

    # origin and destination from directional route
    daily["airport1"] = daily["route"].apply(lambda r: r.split("-")[0])
    daily["airport2"] = daily["route"].apply(lambda r: r.split("-")[1])

    weather_cols = [
        "temp_avg", "temp_max", "temp_min", "temp_range",
        "precip_total", "snowfall", "wind_speed_max", "wind_gusts_max",
        "severity", "has_precipitation", "has_snow", "is_adverse"
    ]

    weather_subset = weather[["date", "airport"] + weather_cols].copy()

    weather1 = weather_subset.rename(
        columns={col: f"apt1_{col}" for col in weather_cols}
    )
    weather1 = weather1.rename(columns={"airport": "airport1"})

    weather2 = weather_subset.rename(
        columns={col: f"apt2_{col}" for col in weather_cols}
    )
    weather2 = weather2.rename(columns={"airport": "airport2"})

    daily = daily.merge(weather1, on=["date", "airport1"], how="left")
    daily = daily.merge(weather2, on=["date", "airport2"], how="left")

    # combined weather features - worst case matters most for delays
    daily["weather_severity_max"] = daily[["apt1_severity", "apt2_severity"]].max(axis=1)
    daily["weather_severity_combined"] = daily[["apt1_severity", "apt2_severity"]].mean(axis=1)
    daily["has_adverse_weather"] = (
        (daily["apt1_is_adverse"] == 1) | (daily["apt2_is_adverse"] == 1)
    ).astype(int)

    daily["total_precip"] = daily["apt1_precip_total"].fillna(0) + daily["apt2_precip_total"].fillna(0)
    daily["total_snowfall"] = daily["apt1_snowfall"].fillna(0) + daily["apt2_snowfall"].fillna(0)
    daily["max_wind"] = daily[["apt1_wind_speed_max", "apt2_wind_speed_max"]].max(axis=1)

    daily = daily.drop(columns=["airport1", "airport2"])

    weather_match_rate = daily["apt1_severity"].notna().mean() * 100
    print(f"  Weather match rate: {weather_match_rate:.1f}%")

    return daily


def save_processed_data(daily):
    """Save processed data to CSV."""
    output_path = PROCESSED_DATA_DIR / "daily_route_demand.csv"
    daily.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def process_all():
    """Main processing pipeline."""
    print("Starting BTS On-Time Performance data processing...")

    df = load_raw_data()
    df = clean_data(df)
    top_routes = identify_top_routes(df, n=20)
    daily = aggregate_daily(df, top_routes)
    daily = add_route_metadata(daily)
    daily = fill_missing_dates(daily)

    weather = load_weather_data()
    daily = merge_weather_data(daily, weather)

    save_processed_data(daily)

    print("\nProcessing complete!")
    print(f"Routes: {daily['route'].nunique()}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"Total records: {len(daily):,}")

    return daily


if __name__ == "__main__":
    process_all()
