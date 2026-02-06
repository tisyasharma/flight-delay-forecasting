import torch

TRAIN_END = "2024-01-01"
VAL_END = "2024-07-01"
TEST_START = "2024-07-01"
TEST_END = "2025-06-30"
DATA_START = "2019-01-01"

COVID_START = "2020-03-01"
COVID_PEAK_END = "2021-06-01"
COVID_RECOVERY_END = "2022-06-01"

SEQUENCE_LENGTH = 28

WALK_FORWARD_FOLDS = [
    {
        "train_end": "2022-07-01",
        "val_end": "2023-01-01",
        "test_start": "2023-01-01",
        "test_end": "2023-06-30",
    },
    {
        "train_end": "2023-01-01",
        "val_end": "2023-07-01",
        "test_start": "2023-07-01",
        "test_end": "2023-12-31",
    },
    {
        "train_end": "2023-07-01",
        "val_end": "2024-01-01",
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
    },
    {
        "train_end": "2024-01-01",
        "val_end": "2024-07-01",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
]


def get_device():
    """Picks the best available torch device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

TABULAR_FEATURES = [
    # calendar
    'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    'is_federal_holiday', 'days_to_holiday', 'days_from_holiday',
    'is_holiday_week', 'is_school_break',

    # covid
    'is_covid_period', 'is_covid_recovery', 'is_post_covid',

    # route stats
    'route_encoded', 'route_mean_demand', 'route_std_demand',
    'route_median_demand', 'route_delay_mean', 'route_delay_std',

    # explicit lag columns since XGBoost can't see sequence position
    'lag_1_arr_delay', 'lag_7_arr_delay', 'lag_14_arr_delay', 'lag_28_arr_delay',
    'rolling_mean_7_arr_delay', 'rolling_mean_14_arr_delay',
    'rolling_std_7_arr_delay', 'rolling_std_14_arr_delay',
    'ewm_7_arr_delay',

    # weather
    'apt1_severity', 'apt2_severity',
    'weather_severity_max', 'weather_severity_combined',
    'has_adverse_weather', 'severe_weather_level', 'both_clear', 'weather_diff',
    'apt1_temp_avg', 'apt2_temp_avg',
    'apt1_precip_total', 'apt2_precip_total',
    'apt1_snowfall', 'apt2_snowfall',
    'apt1_wind_speed_max', 'apt2_wind_speed_max',
    'total_precip', 'total_snowfall', 'max_wind',
    'weather_severity_lag_1', 'weather_severity_lag_3', 'weather_severity_lag_7',

    # hourly-derived features
    'peak_wind_operating', 'precip_operating', 'max_hourly_severity',
    'storm_hours', 'morning_severity', 'evening_severity',
]

SEQUENCE_MODEL_FEATURES = [
    # target history -- model sees [t-28..t-1] and predicts t
    'avg_arr_delay',

    # calendar
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_federal_holiday', 'is_holiday_week',
    'is_school_break',
    'is_covid_period', 'is_covid_recovery', 'is_post_covid',

    # same-day weather (ERA5 reanalysis, observed conditions)
    'weather_severity_max',
    'apt1_severity', 'apt2_severity',
    'has_adverse_weather',
    'apt1_temp_avg', 'apt2_temp_avg',
    'total_precip', 'total_snowfall', 'max_wind',
    'severe_weather_level',

    # hourly-derived features
    'peak_wind_operating', 'storm_hours',
    'morning_severity', 'evening_severity',
]

# weather features for ablation study
WEATHER_FEATURES = [
    'apt1_severity', 'apt2_severity',
    'weather_severity_max', 'weather_severity_combined',
    'has_adverse_weather', 'severe_weather_level', 'both_clear', 'weather_diff',
    'apt1_temp_avg', 'apt2_temp_avg',
    'apt1_precip_total', 'apt2_precip_total',
    'apt1_snowfall', 'apt2_snowfall',
    'apt1_wind_speed_max', 'apt2_wind_speed_max',
    'total_precip', 'total_snowfall', 'max_wind',
    'weather_severity_lag_1', 'weather_severity_lag_3', 'weather_severity_lag_7',

    # hourly-derived features
    'peak_wind_operating', 'precip_operating', 'max_hourly_severity',
    'storm_hours', 'morning_severity', 'evening_severity',
]
