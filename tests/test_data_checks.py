"""
The rebuild gates must pass on well-formed data and name the failure when a
contract breaks.
"""

import numpy as np
import pandas as pd

from src.data_checks import (
    check_calendar_shape,
    check_feature_completeness,
    check_holiday_flags,
    check_raw_completeness,
    check_route_set,
    check_wind_seam,
)


def _calendar_frame(routes=("A-B", "C-D"), start="2024-01-01", days=10):
    dates = pd.date_range(start, periods=days, freq="D")
    frames = [pd.DataFrame({"date": dates, "route": route}) for route in routes]
    df = pd.concat(frames, ignore_index=True)
    df["avg_arr_delay"] = 1.0
    return df


def test_route_set_matches_state():
    df = _calendar_frame()
    state = {"route_codes": {"A-B": 0, "C-D": 1}}
    assert check_route_set(df, state) == []

    drifted = {"route_codes": {"A-B": 0, "X-Y": 1}}
    failures = check_route_set(df, drifted)
    assert len(failures) == 1 and "X-Y" in failures[0] and "C-D" in failures[0]


def test_calendar_shape_complete_and_broken():
    df = _calendar_frame()
    assert check_calendar_shape(df, data_start="2024-01-01") == []

    failures = check_calendar_shape(df.iloc[:-1], data_start="2024-01-01")
    assert len(failures) == 1 and "rows" in failures[0]


def test_feature_completeness_flags_nulls():
    df = _calendar_frame()
    df["feat"] = 1.0
    assert check_feature_completeness(df, ["feat"]) == []

    df.loc[0, "feat"] = np.nan
    failures = check_feature_completeness(df, ["feat"])
    assert len(failures) == 1 and "feat" in failures[0]


def test_raw_completeness_threshold():
    df = pd.DataFrame({"avg_arr_delay": [1.0] * 90 + [np.nan] * 10})
    assert check_raw_completeness(df, columns=["avg_arr_delay"], threshold=0.9) == []
    assert len(check_raw_completeness(df, columns=["avg_arr_delay"], threshold=0.95)) == 1


def test_holiday_flags_present_and_zeroed():
    """The zeroed-flag regression (pandas isin behavior change) must be caught."""
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "is_federal_holiday": 0})
    df.loc[df["date"].dt.dayofyear.isin(range(1, 340, 30)), "is_federal_holiday"] = 1
    assert check_holiday_flags(df) == []

    df["is_federal_holiday"] = 0
    failures = check_holiday_flags(df)
    assert len(failures) == 1 and "federal holidays" in failures[0]


def _wind_frame(shift=0.0):
    rng = np.random.default_rng(11)
    days = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    dates = days.repeat(10)
    seasonal = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    values = seasonal + rng.normal(0, 2, len(dates))
    values = values + np.where(dates >= pd.Timestamp("2024-10-01"), shift, 0.0)
    return pd.DataFrame({"date": dates, "max_wind": values})


def test_wind_seam_quiet_and_breached():
    assert check_wind_seam(_wind_frame()) == []

    failures = check_wind_seam(_wind_frame(shift=-8.0))
    assert len(failures) == 1 and "PSI breach" in failures[0]
