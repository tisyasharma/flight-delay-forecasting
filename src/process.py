import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
WEATHER_DATA_DIR = Path(__file__).parent.parent / "data" / "weather"
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
    """Reads all BTS CSV files and concatenates them into one dataframe."""
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading"):
        df = pd.read_csv(
            csv_file,
            usecols=lambda c: c in REQUIRED_COLS,
            low_memory=False
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"{len(df):,} flight records total")

    return df


def clean_data(df):
    """Cleans up column types, normalizes carrier names, creates route column."""
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # BTS files aren't consistent about which carrier column they use
    if "REPORTING_AIRLINE" not in df.columns and "OP_UNIQUE_CARRIER" in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "REPORTING_AIRLINE"})
    elif "REPORTING_AIRLINE" in df.columns and "OP_UNIQUE_CARRIER" in df.columns:
        df["REPORTING_AIRLINE"] = df["REPORTING_AIRLINE"].fillna(df["OP_UNIQUE_CARRIER"])

    df["ORIGIN"] = df["ORIGIN"].astype(str).str.strip().str.upper()
    df["DEST"] = df["DEST"].astype(str).str.strip().str.upper()
    df["REPORTING_AIRLINE"] = df["REPORTING_AIRLINE"].astype(str).str.strip().str.upper()

    for col in ["DEP_DELAY", "ARR_DELAY", "DISTANCE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CANCELLED"] = df["CANCELLED"].fillna(0).astype(int)
    df = df.dropna(subset=["FL_DATE", "ORIGIN", "DEST"])

    # directional routes, LAX->JFK has different patterns than JFK->LAX
    df["route"] = df["ORIGIN"] + "-" + df["DEST"]

    return df


def load_frozen_routes():
    """
    The route set pinned by the training feature state, if one exists. The
    modeled set is frozen: route_encoded and every route-level statistic are
    keyed to it, so a rank shift in new months must never change the set.
    """
    state_path = PROCESSED_DATA_DIR / "feature_state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return sorted(json.load(f)["route_codes"])


def identify_top_routes(df, n=20):
    """Finds the n busiest routes by total flight count."""
    route_counts = df.groupby("route").size().reset_index(name="flight_count")
    route_counts = route_counts.sort_values("flight_count", ascending=False)

    top_df = route_counts.head(n)
    top_routes = top_df["route"].tolist()

    print(f"Top {n} routes:")
    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"  {i}. {row['route']}: {row['flight_count']:,}")

    return top_routes


def aggregate_inbound_by_destination(df, airports):
    """
    Aggregates all-carrier inbound arrivals per destination airport-day,
    computed before the route filter so it covers every inbound flight, not
    just the modeled routes. Backtest evidence only; the model feature uses
    the modeled-routes variant built inside FeatureBuilder.
    """
    inbound = df[df["DEST"].isin(airports)]
    hub = inbound.groupby(["FL_DATE", "DEST"]).agg(
        inbound_flights=("ARR_DELAY", "count"),
        inbound_avg_arr_delay=("ARR_DELAY", "mean"),
    ).reset_index()

    hub = hub.rename(columns={"FL_DATE": "date", "DEST": "airport"})
    hub = hub.sort_values(["airport", "date"]).reset_index(drop=True)
    return hub


def aggregate_daily(df, top_routes):
    """Groups flights into daily route-level averages."""
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

    print(f"Aggregated: {len(daily):,} route-day records")

    return daily


def add_route_metadata(daily):
    """Adds per-route aggregate stats like avg daily flights and distance."""
    route_stats = daily.groupby("route").agg(
        avg_daily_flights=("flight_count", "mean"),
        total_flights=("flight_count", "sum"),
        avg_distance=("total_distance", lambda x: (x / daily.loc[x.index, "flight_count"]).mean())
    ).reset_index()

    daily = daily.merge(route_stats, on="route", how="left")
    return daily


def fill_missing_dates(daily):
    """Fills calendar gaps so every route has a row for every date."""
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

    for col in ["avg_dep_delay", "avg_arr_delay"]:
        daily[col] = daily.groupby("route")[col].ffill(limit=7)
        daily[col] = daily[col].fillna(0)
    daily["cancel_rate"] = daily["cancel_rate"].fillna(0)

    route_meta = daily.groupby("route")[["avg_daily_flights", "total_flights", "avg_distance"]].first()
    for col in route_meta.columns:
        daily[col] = daily["route"].map(route_meta[col])

    print(f"After fill: {len(daily):,} records")
    return daily


def load_weather_data():
    """Loads the processed weather CSV."""
    weather_path = WEATHER_DATA_DIR / "weather_daily.csv"
    weather = pd.read_csv(weather_path)
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def merge_weather_data(daily, weather):
    """Joins weather data for both origin and destination airports onto daily."""
    # split route into origin/dest
    daily["airport1"] = daily["route"].apply(lambda r: r.split("-")[0])
    daily["airport2"] = daily["route"].apply(lambda r: r.split("-")[1])

    weather_cols = [
        "temp_avg", "temp_max", "temp_min", "temp_range",
        "precip_total", "snowfall", "wind_speed_max", "wind_gusts_max",
        "severity", "has_precipitation", "has_snow", "is_adverse",
        "peak_wind_operating", "precip_operating", "max_hourly_severity",
        "storm_hours", "morning_severity", "evening_severity"
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

    # combined features, worst-case matters most for delays
    daily["weather_severity_max"] = daily[["apt1_severity", "apt2_severity"]].max(axis=1)
    daily["weather_severity_combined"] = daily[["apt1_severity", "apt2_severity"]].mean(axis=1)
    daily["has_adverse_weather"] = (
        (daily["apt1_is_adverse"] == 1) | (daily["apt2_is_adverse"] == 1)
    ).astype(int)

    daily["total_precip"] = daily["apt1_precip_total"].fillna(0) + daily["apt2_precip_total"].fillna(0)
    daily["total_snowfall"] = daily["apt1_snowfall"].fillna(0) + daily["apt2_snowfall"].fillna(0)
    daily["max_wind"] = daily[["apt1_wind_speed_max", "apt2_wind_speed_max"]].max(axis=1)

    # hourly-derived combined features (worst-case across both airports)
    hourly_max_cols = [
        ("peak_wind_operating", "peak_wind_operating"),
        ("max_hourly_severity", "max_hourly_severity"),
        ("morning_severity", "morning_severity"),
        ("evening_severity", "evening_severity"),
    ]
    for col_suffix, out_name in hourly_max_cols:
        apt1_col = f"apt1_{col_suffix}"
        apt2_col = f"apt2_{col_suffix}"
        if apt1_col in daily.columns and apt2_col in daily.columns:
            daily[out_name] = daily[[apt1_col, apt2_col]].max(axis=1)

    hourly_sum_cols = [("precip_operating", "precip_operating"), ("storm_hours", "storm_hours")]
    for col_suffix, out_name in hourly_sum_cols:
        apt1_col = f"apt1_{col_suffix}"
        apt2_col = f"apt2_{col_suffix}"
        if apt1_col in daily.columns and apt2_col in daily.columns:
            daily[out_name] = daily[apt1_col].fillna(0) + daily[apt2_col].fillna(0)

    daily = daily.drop(columns=["airport1", "airport2"])

    match_rate = daily["apt1_severity"].notna().mean() * 100
    print(f"Weather match rate: {match_rate:.1f}%")

    return daily


def load_aviation_data():
    """Loads the GFS-derived aviation weather CSV."""
    aviation_path = WEATHER_DATA_DIR / "aviation_daily.csv"
    aviation = pd.read_csv(aviation_path)
    aviation["date"] = pd.to_datetime(aviation["date"])
    return aviation


def merge_aviation_data(daily, aviation):
    """Joins aviation weather for both origin and destination airports onto daily."""
    daily["airport1"] = daily["route"].apply(lambda r: r.split("-")[0])
    daily["airport2"] = daily["route"].apply(lambda r: r.split("-")[1])

    aviation_cols = [
        "visibility_min", "visibility_p10", "cape_max", "gust_max",
        "gust_spread", "low_cloud_hours", "freezing_rain_hours",
    ]

    aviation_subset = aviation[["date", "airport"] + aviation_cols].copy()

    aviation1 = aviation_subset.rename(
        columns={col: f"apt1_{col}" for col in aviation_cols}
    )
    aviation1 = aviation1.rename(columns={"airport": "airport1"})

    aviation2 = aviation_subset.rename(
        columns={col: f"apt2_{col}" for col in aviation_cols}
    )
    aviation2 = aviation2.rename(columns={"airport": "airport2"})

    daily = daily.merge(aviation1, on=["date", "airport1"], how="left")
    daily = daily.merge(aviation2, on=["date", "airport2"], how="left")

    daily["has_freezing_rain"] = (
        (daily["apt1_freezing_rain_hours"] > 0) | (daily["apt2_freezing_rain_hours"] > 0)
    ).astype(int)

    daily = daily.drop(columns=["airport1", "airport2"])

    match_rate = daily["apt1_visibility_min"].notna().mean() * 100
    print(f"Aviation weather match rate: {match_rate:.1f}%")

    return daily


def save_processed_data(daily):
    """Writes the processed daily dataframe to CSV."""
    output_path = PROCESSED_DATA_DIR / "daily_route_demand.csv"
    daily.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def process_all():
    """Runs the full pipeline from raw BTS data to processed daily CSV."""
    df = load_raw_data()
    df = clean_data(df)
    top_routes = identify_top_routes(df, n=50)

    frozen = load_frozen_routes()
    if frozen is not None:
        drift = sorted(set(frozen) ^ set(top_routes))
        if drift:
            print(f"Route ranking drifted on {len(drift)} routes vs the frozen set "
                  f"({', '.join(drift[:6])}{'...' if len(drift) > 6 else ''}); "
                  "keeping the frozen set")
        top_routes = frozen

    route_airports = sorted({a for route in top_routes for a in route.split("-")})
    hub = aggregate_inbound_by_destination(df, route_airports)
    hub_path = PROCESSED_DATA_DIR / "hub_inbound_daily.csv"
    hub.to_csv(hub_path, index=False)
    print(f"Hub inbound aggregates: {len(hub):,} airport-days")

    daily = aggregate_daily(df, top_routes)
    daily = add_route_metadata(daily)
    daily = fill_missing_dates(daily)

    weather = load_weather_data()
    daily = merge_weather_data(daily, weather)

    aviation = load_aviation_data()
    daily = merge_aviation_data(daily, aviation)

    save_processed_data(daily)

    print(f"\nDone: {daily['route'].nunique()} routes, {len(daily):,} records")
    print(f"{daily['date'].min().date()} to {daily['date'].max().date()}")

    return daily


if __name__ == "__main__":
    process_all()
