"""
Pure weather transformations shared by the training fetchers and any live
pipeline: WMO code mapping, severity scoring, and hourly-to-daily aggregation.
No I/O happens here.
"""

import pandas as pd


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
