"""
Single source of truth for model feature names.

Every model-facing feature list is derived from FEATURE_GROUPS, so adding or
renaming a feature happens in exactly one place. src/config.py re-exports the
derived lists for backward compatibility.
"""

FEATURE_GROUPS = {
    "calendar": [
        "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
        "is_weekend", "is_month_start", "is_month_end",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "is_federal_holiday", "days_to_holiday", "days_from_holiday",
        "is_holiday_week", "is_school_break",
    ],
    "covid": [
        "is_covid_period", "is_covid_recovery", "is_post_covid",
    ],
    "route_stats": [
        "route_encoded", "route_mean_demand", "route_std_demand",
        "route_median_demand", "route_delay_mean", "route_delay_std",
    ],
    # explicit lag columns since tree models can't see sequence position
    "delay_lags": [
        "lag_1_arr_delay", "lag_7_arr_delay", "lag_14_arr_delay", "lag_28_arr_delay",
        "rolling_mean_7_arr_delay", "rolling_mean_14_arr_delay",
        "rolling_std_7_arr_delay", "rolling_std_14_arr_delay",
        "ewm_7_arr_delay",
    ],
    "weather_daily": [
        "apt1_severity", "apt2_severity",
        "weather_severity_max", "weather_severity_combined",
        "has_adverse_weather", "severe_weather_level", "both_clear", "weather_diff",
        "apt1_temp_avg", "apt2_temp_avg",
        "apt1_precip_total", "apt2_precip_total",
        "apt1_snowfall", "apt2_snowfall",
        "apt1_wind_speed_max", "apt2_wind_speed_max",
        "total_precip", "total_snowfall", "max_wind",
    ],
    "weather_lags": [
        "weather_severity_lag_1", "weather_severity_lag_3", "weather_severity_lag_7",
    ],
    "weather_hourly": [
        "peak_wind_operating", "precip_operating", "max_hourly_severity",
        "storm_hours", "morning_severity", "evening_severity",
    ],
    # GFS-derived aviation weather. lifted index is fetched but excluded here
    # because its archived distribution is not stationary across years
    "aviation_weather": [
        "apt1_visibility_min", "apt2_visibility_min",
        "apt1_visibility_p10", "apt2_visibility_p10",
        "apt1_cape_max", "apt2_cape_max",
        "apt1_gust_max", "apt2_gust_max",
        "apt1_gust_spread", "apt2_gust_spread",
        "apt1_low_cloud_hours", "apt2_low_cloud_hours",
        "apt1_freezing_rain_hours", "apt2_freezing_rain_hours",
        "has_freezing_rain",
    ],
    # network state from the modeled routes only, so serving can roll it
    # forward recursively alongside the route's own delay lags
    "hub_state": [
        "hub_inbound_lag_1", "hub_inbound_roll_7",
    ],
}


def _pick(group, names):
    """
    Returns names in the given order after asserting each one belongs to the
    group, so a rename in FEATURE_GROUPS can't silently orphan a subset list.
    """
    missing = [n for n in names if n not in FEATURE_GROUPS[group]]
    if missing:
        raise ValueError(f"not in feature group '{group}': {missing}")
    return list(names)


TABULAR_FEATURES = (
    FEATURE_GROUPS["calendar"]
    + FEATURE_GROUPS["covid"]
    + FEATURE_GROUPS["route_stats"]
    + FEATURE_GROUPS["delay_lags"]
    + FEATURE_GROUPS["weather_daily"]
    + FEATURE_GROUPS["weather_lags"]
    + FEATURE_GROUPS["weather_hourly"]
    + FEATURE_GROUPS["aviation_weather"]
    + FEATURE_GROUPS["hub_state"]
)

WEATHER_FEATURES = (
    FEATURE_GROUPS["weather_daily"]
    + FEATURE_GROUPS["weather_lags"]
    + FEATURE_GROUPS["weather_hourly"]
    + FEATURE_GROUPS["aviation_weather"]
)

# target history -- sequence models see [t-28..t-1] and predict t
SEQUENCE_MODEL_FEATURES = (
    ["avg_arr_delay"]
    + _pick("calendar", [
        "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_federal_holiday", "is_holiday_week",
        "is_school_break",
    ])
    + FEATURE_GROUPS["covid"]
    + _pick("weather_daily", [
        "weather_severity_max",
        "apt1_severity", "apt2_severity",
        "has_adverse_weather",
        "apt1_temp_avg", "apt2_temp_avg",
        "total_precip", "total_snowfall", "max_wind",
        "severe_weather_level",
    ])
    + _pick("weather_hourly", [
        "peak_wind_operating", "storm_hours",
        "morning_severity", "evening_severity",
    ])
)
