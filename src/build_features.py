import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.config import COVID_START, COVID_PEAK_END, COVID_RECOVERY_END

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

HOLIDAY_YEARS_START = 2019


class FeatureBuilder:
    """
    train_end_date controls leakage prevention: route stats and imputation
    only use data before this date.

    Passing state (a dict from a previous export_state call) makes the builder
    reuse train-time route encodings, route stats, and fill values instead of
    recomputing them from df, so features for new dates match what the models
    were trained on without needing the original training data.
    """

    def __init__(self, df, train_end_date=None, state=None):
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values(["route", "date"]).reset_index(drop=True)
        self.train_end_date = pd.Timestamp(train_end_date) if train_end_date else None

        self.state = state
        if self.state is not None:
            unknown = set(self.df["route"].unique()) - set(self.state["route_codes"])
            if unknown:
                raise ValueError(f"routes missing from feature state: {sorted(unknown)}")

        # extend the calendar one year past the data so days_to_holiday never
        # silently degrades to the 30-day cap for dates near the end of range
        max_year = int(self.df["date"].dt.year.max())
        self.us_holidays = holidays.US(years=range(HOLIDAY_YEARS_START, max_year + 2))

        self._route_codes = None
        self._route_stats = None
        self._temp_fill_values = {}
        self._aviation_fill_values = {}
        self._lag_fill_medians = {}

    def add_temporal_features(self):
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["day_of_month"] = self.df["date"].dt.day
        self.df["month"] = self.df["date"].dt.month
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["year"] = self.df["date"].dt.year
        self.df["week_of_year"] = self.df["date"].dt.isocalendar().week.astype(int)

        self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        self.df["is_month_start"] = (self.df["day_of_month"] <= 3).astype(int)
        self.df["is_month_end"] = (self.df["day_of_month"] >= 28).astype(int)

        # cyclical so day 0 and day 6 are neighbors
        self.df["day_of_week_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_of_week_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["month_sin"] = np.sin(2 * np.pi * (self.df["month"] - 1) / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * (self.df["month"] - 1) / 12)

        return self

    def add_holiday_features(self):
        self.df["is_federal_holiday"] = self.df["date"].isin(self.us_holidays).astype(int)

        holiday_dates = pd.DatetimeIndex(sorted(self.us_holidays.keys()))
        unique_dates = self.df["date"].unique()

        days_to = {}
        days_from = {}
        for date in unique_dates:
            future = holiday_dates[holiday_dates > date]
            days_to[date] = min(int((future[0] - date).days), 30) if len(future) > 0 else 30
            past = holiday_dates[holiday_dates < date]
            days_from[date] = min(int((date - past[-1]).days), 30) if len(past) > 0 else 30

        self.df["days_to_holiday"] = self.df["date"].map(days_to)
        self.df["days_from_holiday"] = self.df["date"].map(days_from)

        self.df["is_holiday_week"] = (
            (self.df["days_to_holiday"] <= 3) | (self.df["days_from_holiday"] <= 3)
        ).astype(int)

        def is_school_break(date):
            # rough approximation of US school breaks
            month, day = date.month, date.day
            # winter break
            if month == 12 and day >= 20:
                return 1
            if month == 1 and day <= 5:
                return 1
            # spring break (staggered across colleges, K-12, and workforce travel)
            if month == 3 and 8 <= day <= 24:
                return 1
            # summer
            if month == 6 or month == 7 or month == 8:
                return 1
            # thanksgiving
            if month == 11 and day >= 20 and day <= 30:
                return 1
            return 0

        self.df["is_school_break"] = self.df["date"].apply(is_school_break)

        return self

    def add_covid_features(self):
        covid_start = pd.Timestamp(COVID_START)
        covid_peak_end = pd.Timestamp(COVID_PEAK_END)
        recovery_end = pd.Timestamp(COVID_RECOVERY_END)

        self.df["is_covid_period"] = (
            (self.df["date"] >= covid_start) & (self.df["date"] < covid_peak_end)
        ).astype(int)

        self.df["is_covid_recovery"] = (
            (self.df["date"] >= covid_peak_end) & (self.df["date"] < recovery_end)
        ).astype(int)

        self.df["is_post_covid"] = (self.df["date"] >= recovery_end).astype(int)

        return self

    def add_lag_features(self):
        lag_periods = [1, 7, 14, 28]
        target_cols = ["avg_arr_delay"]

        for target in target_cols:
            suffix = f"_{target.replace('avg_', '')}"

            for lag in lag_periods:
                self.df[f"lag_{lag}{suffix}"] = self.df.groupby("route")[target].shift(lag)

            # shift(1) before rolling so we don't include today's value
            for window in [7, 14]:
                self.df[f"rolling_mean_{window}{suffix}"] = (
                    self.df.groupby("route")[target]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )
                self.df[f"rolling_std_{window}{suffix}"] = (
                    self.df.groupby("route")[target]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
                )

            self.df[f"ewm_7{suffix}"] = (
                self.df.groupby("route")[target]
                .transform(lambda x: x.shift(1).ewm(span=7, min_periods=1).mean())
            )

        return self

    def add_route_features(self):
        """Route-level stats computed on training data only to prevent leakage."""
        if self.state is not None:
            self._route_codes = dict(self.state["route_codes"])
            self._route_stats = self.state["route_stats"]

            self.df["route_encoded"] = self.df["route"].map(self._route_codes).astype(int)

            stats = pd.DataFrame.from_dict(self._route_stats, orient="index")
            stats.index.name = "route"
            self.df = self.df.merge(stats.reset_index(), on="route", how="left")

            return self

        self.df["route_encoded"] = self.df["route"].astype("category").cat.reorder_categories(
            sorted(self.df["route"].unique())
        ).cat.codes

        if self.train_end_date:
            train_mask = self.df["date"] < self.train_end_date
            stats_df = self.df[train_mask]
        else:
            stats_df = self.df

        route_stats = stats_df.groupby("route")["flight_count"].agg(["mean", "std", "median"])
        route_stats.columns = ["route_mean_demand", "route_std_demand", "route_median_demand"]

        self.df = self.df.merge(route_stats, on="route", how="left")

        delay_stats = stats_df.groupby("route")["avg_arr_delay"].agg(["mean", "std"])
        delay_stats.columns = ["route_delay_mean", "route_delay_std"]
        delay_stats["route_delay_std"] = delay_stats["route_delay_std"].fillna(0)

        self.df = self.df.merge(delay_stats, on="route", how="left")

        self._route_codes = {
            route: code for code, route in enumerate(sorted(self.df["route"].unique()))
        }
        combined = route_stats.join(delay_stats)
        self._route_stats = {
            route: {col: (float(val) if pd.notna(val) else None) for col, val in row.items()}
            for route, row in combined.iterrows()
        }

        return self

    def add_weather_features(self):
        if self.state is None:
            if self.train_end_date:
                train_mask = self.df["date"] < self.train_end_date
            else:
                train_mask = pd.Series(True, index=self.df.index)

        # missing severity = clear weather
        severity_cols = [
            "apt1_severity", "apt2_severity", "weather_severity_max",
            "weather_severity_combined", "has_adverse_weather"
        ]
        for col in severity_cols:
            self.df[col] = self.df[col].fillna(0)

        # temps: fill with training median
        for col in ["apt1_temp_avg", "apt2_temp_avg"]:
            if self.state is not None:
                fill_value = self.state["temp_fill_values"][col]
            else:
                train_median = self.df.loc[train_mask, col].median()
                fill_value = train_median if pd.notna(train_median) else 50.0
            self._temp_fill_values[col] = float(fill_value)
            self.df[col] = self.df[col].fillna(fill_value)

        # precip and wind default to 0
        precip_wind_cols = [
            "apt1_precip_total", "apt2_precip_total",
            "apt1_snowfall", "apt2_snowfall",
            "apt1_wind_speed_max", "apt2_wind_speed_max",
            "total_precip", "total_snowfall", "max_wind",
            "peak_wind_operating", "precip_operating",
            "max_hourly_severity", "storm_hours",
            "morning_severity", "evening_severity",
        ]
        precip_wind_cols = [c for c in precip_wind_cols if c in self.df.columns]
        for col in precip_wind_cols:
            self.df[col] = self.df[col].fillna(0)

        # 0 = ok, 1 = bad, 2 = really bad
        self.df["severe_weather_level"] = (
            (self.df["weather_severity_max"] >= 4).astype(int) +
            (self.df["weather_severity_max"] >= 5).astype(int)
        )

        self.df["both_clear"] = (
            (self.df["apt1_severity"] == 0) & (self.df["apt2_severity"] == 0)
        ).astype(int)

        self.df["weather_diff"] = abs(
            self.df["apt1_severity"] - self.df["apt2_severity"]
        )

        for lag in [1, 3, 7]:
            self.df[f"weather_severity_lag_{lag}"] = (
                self.df.groupby("route")["weather_severity_max"].shift(lag)
            )

        lag_cols = [c for c in self.df.columns if "weather_severity_lag" in c]
        for col in lag_cols:
            self.df[col] = self.df[col].fillna(0)

        return self

    def add_hub_features(self):
        """
        Cross-route network state: the average arrival delay over all modeled
        routes into each destination airport on prior days, joined back to
        route-days by destination. Computed from the modeled routes only, so a
        live pipeline can rebuild it from rolled-forward predictions without
        any additional data source.
        """
        hub_daily = (
            self.df.assign(dest=self.df["route"].str.split("-").str[1])
            .groupby(["dest", "date"])["avg_arr_delay"]
            .mean()
            .rename("hub_state")
            .reset_index()
            .sort_values(["dest", "date"])
        )

        grouped = hub_daily.groupby("dest")["hub_state"]
        hub_daily["hub_inbound_lag_1"] = grouped.shift(1)
        hub_daily["hub_inbound_roll_7"] = grouped.transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )

        self.df["_dest"] = self.df["route"].str.split("-").str[1]
        self.df = self.df.merge(
            hub_daily[["dest", "date", "hub_inbound_lag_1", "hub_inbound_roll_7"]],
            left_on=["_dest", "date"],
            right_on=["dest", "date"],
            how="left",
        ).drop(columns=["_dest", "dest"])

        return self

    def add_aviation_features(self):
        """
        Fills the GFS aviation columns. Visibility gets the train-window median
        because zero would mean fog, the opposite of missing; convective, gust,
        cloud, and freezing-rain columns default to a benign zero.
        """
        if self.state is None:
            if self.train_end_date:
                train_mask = self.df["date"] < self.train_end_date
            else:
                train_mask = pd.Series(True, index=self.df.index)

        visibility_cols = [
            "apt1_visibility_min", "apt2_visibility_min",
            "apt1_visibility_p10", "apt2_visibility_p10",
        ]
        for col in visibility_cols:
            if self.state is not None:
                fill_value = self.state["aviation_fill_values"][col]
            else:
                train_median = self.df.loc[train_mask, col].median()
                fill_value = train_median if pd.notna(train_median) else 20000.0
            self._aviation_fill_values[col] = float(fill_value)
            self.df[col] = self.df[col].fillna(fill_value)

        zero_cols = [
            "apt1_cape_max", "apt2_cape_max",
            "apt1_gust_max", "apt2_gust_max",
            "apt1_gust_spread", "apt2_gust_spread",
            "apt1_low_cloud_hours", "apt2_low_cloud_hours",
            "apt1_freezing_rain_hours", "apt2_freezing_rain_hours",
            "has_freezing_rain",
        ]
        zero_cols = [c for c in zero_cols if c in self.df.columns]
        for col in zero_cols:
            self.df[col] = self.df[col].fillna(0)

        return self

    def fill_missing_values(self):
        lag_cols = [c for c in self.df.columns if c.startswith(("lag_", "rolling_", "ewm_", "hub_"))]

        if self.state is None:
            if self.train_end_date:
                train_mask = self.df["date"] < self.train_end_date
                train_df = self.df[train_mask]
            else:
                train_df = self.df

        for col in lag_cols:
            if self.state is not None:
                train_medians = pd.Series(self.state["lag_fill_medians"][col], dtype=float)
            else:
                train_medians = train_df.groupby("route")[col].median()
            self._lag_fill_medians[col] = {
                route: (float(val) if pd.notna(val) else None)
                for route, val in train_medians.items()
            }
            fill_values = self.df["route"].map(train_medians).fillna(0)
            self.df[col] = self.df[col].fillna(fill_values)

        return self

    def export_state(self):
        """
        Serializes everything the train window gated during build (route
        encodings, route stats, imputation values) as a JSON-safe dict.
        Requires build() to have run first.
        """
        if self._route_stats is None:
            raise RuntimeError("export_state() requires build() to have been called")

        if self.train_end_date is not None:
            train_end = str(self.train_end_date.date())
        elif self.state is not None:
            train_end = self.state.get("train_end_date")
        else:
            train_end = None

        return {
            "train_end_date": train_end,
            "route_codes": self._route_codes,
            "route_stats": self._route_stats,
            "temp_fill_values": self._temp_fill_values,
            "aviation_fill_values": self._aviation_fill_values,
            "lag_fill_medians": self._lag_fill_medians,
        }

    def build(self):
        self.add_temporal_features()
        self.add_holiday_features()
        self.add_covid_features()
        self.add_lag_features()
        self.add_hub_features()
        self.add_route_features()
        self.add_weather_features()
        self.add_aviation_features()
        self.fill_missing_values()

        return self.df


def build_features(input_path=None, output_path=None, train_end_date="2024-01-01",
                   state_path=None):
    if input_path is None:
        input_path = PROCESSED_DATA_DIR / "daily_route_demand.csv"

    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)

    builder = FeatureBuilder(df, train_end_date=train_end_date)
    df_features = builder.build()

    print(f"{len(df_features.columns)} columns total")

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "features.csv"

    df_features.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # state is only written when a path is given, so experimental builds
    # cannot silently overwrite the canonical train-time state
    if state_path is not None:
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(builder.export_state(), f, indent=2)
        print(f"Saved feature state to {state_path}")

    return df_features


if __name__ == "__main__":
    build_features(
        train_end_date="2024-01-01",
        state_path=PROCESSED_DATA_DIR / "feature_state.json",
    )
