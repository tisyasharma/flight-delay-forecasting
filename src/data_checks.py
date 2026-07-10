"""
Data-quality gates for the feature rebuild, encoding the checks that caught
the Dec 2024 wind seam. Runs after every rebuild (make features) and is the
template for the pre-retrain gate in the live pipeline. Nonzero exit on any
breach so automation stops instead of training on contaminated data.
"""

import json
from pathlib import Path

import pandas as pd

from src.config import DATA_START, TABULAR_FEATURES
from src.monitoring.psi import month_matched_psi

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
RAW_PATH = PROCESSED_DATA_DIR / "daily_route_demand.csv"
STATE_PATH = PROCESSED_DATA_DIR / "feature_state.json"

TARGET = "avg_arr_delay"
RAW_KEY_COLUMNS = ["avg_arr_delay", "max_wind", "weather_severity_max"]
RAW_NON_NULL_THRESHOLD = 0.98
SEAM_PSI_THRESHOLD = 0.25
SEAM_WINDOW_MONTHS = 6


def check_route_set(df, state):
    """The feature table must cover exactly the routes the state was built on."""
    expected = set(state["route_codes"])
    actual = set(df["route"].unique())
    failures = []
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        failures.append(f"route set drift, missing={missing} extra={extra}")
    return failures


def check_calendar_shape(df, data_start=DATA_START):
    """One row per route per day from data_start, the fill contract the models trained on."""
    failures = []
    min_date = df["date"].min()
    if min_date != pd.Timestamp(data_start):
        failures.append(f"date range starts {min_date.date()}, expected {data_start}")
    n_days = (df["date"].max() - pd.Timestamp(data_start)).days + 1
    expected_rows = df["route"].nunique() * n_days
    if len(df) != expected_rows:
        failures.append(f"{len(df):,} rows, expected {expected_rows:,} (routes x days)")
    return failures


def check_feature_completeness(df, feature_cols, target=TARGET):
    """Model features and the target must be fully populated after the fill pass."""
    cols = [c for c in feature_cols if c in df.columns] + [target]
    null_rates = df[cols].isna().mean()
    bad = null_rates[null_rates > 0]
    if len(bad):
        return [f"nulls in filled columns: {dict(bad.round(4))}"]
    return []


def check_raw_completeness(raw_df, columns=RAW_KEY_COLUMNS, threshold=RAW_NON_NULL_THRESHOLD):
    """Key raw columns must stay near-complete before any fill logic runs."""
    cols = [c for c in columns if c in raw_df.columns]
    rates = raw_df[cols].notna().mean()
    bad = rates[rates < threshold]
    if len(bad):
        return [f"raw completeness below {threshold}: {dict(bad.round(4))}"]
    return []


def check_holiday_flags(df, min_per_year=9):
    """
    Every complete year must carry federal-holiday flags. A pandas 3.x isin
    behavior change once zeroed the column silently, this catches that class of
    regression on every rebuild. A year counts as complete when its data reaches
    December, so a partial trailing year is not falsely flagged.
    """
    flagged = df.loc[df["is_federal_holiday"] == 1, "date"]
    per_year = flagged.dt.year.value_counts()

    complete_years = []
    for year, group in df.groupby(df["date"].dt.year):
        if group["date"].max().month == 12:
            complete_years.append(year)

    missing = [y for y in complete_years if per_year.get(y, 0) < min_per_year]
    if missing:
        return [f"complete years with fewer than {min_per_year} flagged federal holidays: {missing}"]
    return []


def check_wind_seam(df, months=SEAM_WINDOW_MONTHS, threshold=SEAM_PSI_THRESHOLD):
    """
    Month-matched PSI on max_wind over the trailing window. This is the exact
    signal that scored 0.78 on the seam-contaminated data and 0.011 after the
    era5 pin and refetch.
    """
    window_end = df["date"].max() + pd.Timedelta(days=1)
    window_start = window_end - pd.DateOffset(months=months)
    result = month_matched_psi(df["date"], df["max_wind"], window_start, window_end)
    failures = []
    if result["max"] > threshold:
        failures.append(
            f"max_wind month-matched PSI breach: max {result['max']:.3f} "
            f"(threshold {threshold}), months {result['months']}"
        )
    return failures


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    raw_df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    with open(STATE_PATH) as f:
        state = json.load(f)

    failures = (
        check_route_set(df, state)
        + check_calendar_shape(df)
        + check_feature_completeness(df, TABULAR_FEATURES)
        + check_raw_completeness(raw_df)
        + check_holiday_flags(df)
        + check_wind_seam(df)
    )

    for message in failures:
        print(f"FAIL: {message}")
    if failures:
        return 1

    print(
        f"data checks passed: {len(df):,} rows, {df['route'].nunique()} routes, "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
