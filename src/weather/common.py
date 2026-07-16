"""
Pure weather transformations shared by the training fetchers and any live
pipeline: WMO code mapping, severity scoring, and hourly-to-daily aggregation.
No I/O happens here.
"""

import pandas as pd

# WMO codes for freezing drizzle and freezing rain
FREEZING_RAIN_CODES = {56, 57, 66, 67}

# percent low-cloud cover treated as a broken-or-worse deck (BKN is 5/8 oktas,
# about 62 percent), used as the ceiling proxy since no gridded ceiling exists
LOW_CLOUD_THRESHOLD = 60


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


def drop_unsettled_tail(df, probe_cols):
    """
    Trims trailing days the upstream archive has not settled yet. Open-Meteo
    returns the full requested range and pads unpublished days with nulls,
    and the downstream fill rules would turn those into fabricated clear
    days, so rows after the last date with any real value are dropped and
    the next run's extension refetches them. Interior gaps are left alone,
    they belong to the fill rules.
    """
    present = [c for c in probe_cols if c in df.columns]
    if df.empty or not present:
        return df
    date_col = "datetime" if "datetime" in df.columns else "date"
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    settled = df[present].notna().any(axis=1)
    if not settled.any():
        return df.iloc[0:0]
    return df[dates <= dates[settled].max()]


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


def aggregate_aviation_hourly_to_daily(hourly_df):
    """
    Computes daily aviation-weather aggregates from hourly GFS data: visibility
    floor and 10th percentile, convective potential, gusts and gust spread,
    low-cloud hours (ceiling proxy), and freezing-rain hours. Visibility stays
    NaN when unobserved because 0 would mean fog, the opposite of missing.
    Zero-filling is left to the feature builder's imputation.
    """
    hourly_df["date"] = hourly_df["datetime"].dt.date
    hourly_df["hour"] = hourly_df["datetime"].dt.hour

    # periods without archive coverage (Alaska and Hawaii before 2022) arrive
    # as all-None object columns, coerce so aggregation yields NaN, not errors
    value_cols = ["visibility", "cape", "lifted_index", "cloud_cover_low",
                  "weather_code", "wind_gusts", "wind_speed", "freezing_level"]
    for col in value_cols:
        hourly_df[col] = pd.to_numeric(hourly_df[col], errors="coerce")

    operating = hourly_df[(hourly_df["hour"] >= 6) & (hourly_df["hour"] <= 23)]

    groups = hourly_df.groupby(["airport", "date"])
    operating_groups = operating.groupby(["airport", "date"])

    daily_agg = pd.DataFrame()
    daily_agg["visibility_min"] = operating_groups["visibility"].min()
    daily_agg["visibility_p10"] = operating_groups["visibility"].quantile(0.1)
    daily_agg["cape_max"] = operating_groups["cape"].max()
    daily_agg["gust_max"] = operating_groups["wind_gusts"].max()
    daily_agg["gust_spread"] = daily_agg["gust_max"] - operating_groups["wind_speed"].max()
    daily_agg["low_cloud_hours"] = operating_groups["cloud_cover_low"].apply(
        lambda x: (x >= LOW_CLOUD_THRESHOLD).sum()
    )
    # freezing rain outside operating hours still ices surfaces for the
    # morning bank, so this one counts all 24 hours like storm_hours does
    daily_agg["freezing_rain_hours"] = groups["weather_code"].apply(
        lambda x: x.isin(FREEZING_RAIN_CODES).sum()
    )
    daily_agg["lifted_index_min"] = operating_groups["lifted_index"].min()
    daily_agg["freezing_level_min"] = groups["freezing_level"].min()

    daily_agg = daily_agg.reset_index()
    daily_agg["date"] = pd.to_datetime(daily_agg["date"])

    for col in ["cape_max", "gust_max", "gust_spread", "low_cloud_hours",
                "freezing_rain_hours"]:
        daily_agg[col] = daily_agg[col].fillna(0)

    return daily_agg
