"""
Shared configuration: chronological split dates and feature lists per model type.
XGBoost gets 57 features (needs explicit lags), LSTM/TCN get 22 (see target in sequence).
"""

TRAIN_END = "2024-01-01"
VAL_END = "2024-07-01"
TEST_START = "2024-07-01"

XGBOOST_FEATURES = [
    # Calendar (17)
    'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    'is_federal_holiday', 'days_to_holiday', 'days_from_holiday',
    'is_holiday_week', 'is_school_break',

    # COVID period (3)
    'is_covid_period', 'is_covid_recovery', 'is_post_covid',

    # Route stats (6)
    'route_encoded', 'route_mean_demand', 'route_std_demand',
    'route_median_demand', 'route_delay_mean', 'route_delay_std',

    # XGBoost can't see sequences, so we have explicit lag columns
    'lag_1_arr_delay', 'lag_7_arr_delay', 'lag_14_arr_delay', 'lag_28_arr_delay',
    'rolling_mean_7_arr_delay', 'rolling_mean_14_arr_delay',
    'rolling_std_7_arr_delay', 'rolling_std_14_arr_delay',
    'ewm_7_arr_delay',

    # Weather (22)
    'apt1_severity', 'apt2_severity',
    'weather_severity_max', 'weather_severity_combined',
    'has_adverse_weather', 'severe_weather_level', 'both_clear', 'weather_diff',
    'apt1_temp_avg', 'apt2_temp_avg',
    'apt1_precip_total', 'apt2_precip_total',
    'apt1_snowfall', 'apt2_snowfall',
    'apt1_wind_speed_max', 'apt2_wind_speed_max',
    'total_precip', 'total_snowfall', 'max_wind',
    'weather_severity_lag_1', 'weather_severity_lag_3', 'weather_severity_lag_7'
]

SEQUENCE_MODEL_FEATURES = [
    # target history in the sequence, model sees [t-28 to t-1] and predicts t
    'avg_arr_delay',

    # calendar context
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_federal_holiday', 'is_holiday_week',
    'is_school_break',
    'is_covid_period', 'is_covid_recovery', 'is_post_covid',

    # same-day weather (makes this a nowcast rather than pure forecast)
    'weather_severity_max',
    'apt1_severity', 'apt2_severity',
    'has_adverse_weather',
    'apt1_temp_avg', 'apt2_temp_avg',
    'total_precip', 'total_snowfall', 'max_wind',
    'severe_weather_level',
]

# all weather features, used for ablation study
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
    'weather_rolling_3d',
]
