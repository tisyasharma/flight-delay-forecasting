import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore", category=FutureWarning)


class NaiveBaseline:
    """Predicts yesterday's delay value."""

    def __init__(self, target_col="avg_arr_delay"):
        self.name = "Naive"
        self.target_col = target_col

    def fit(self, train_df):
        """Stores per-route medians as fallback for missing lag values."""
        self.route_median_ = train_df.groupby("route")[self.target_col].median()
        self.global_median_ = train_df[self.target_col].median()
        return self

    def predict(self, df):
        """Returns yesterday's delay, falls back to route median if missing."""
        predictions = df.groupby("route")[self.target_col].shift(1)
        fill_values = df["route"].map(self.route_median_)
        predictions = predictions.fillna(fill_values)
        return predictions.fillna(self.global_median_)


class SeasonalNaiveBaseline:
    """Predicts the value from the same weekday last week."""

    def __init__(self, seasonality=7, target_col="avg_arr_delay"):
        self.name = f"SeasonalNaive_{seasonality}"
        self.seasonality = seasonality
        self.target_col = target_col

    def fit(self, train_df):
        """Stores per-route medians as fallback."""
        self.route_median_ = train_df.groupby("route")[self.target_col].median()
        self.global_median_ = train_df[self.target_col].median()
        return self

    def predict(self, df):
        """Returns the delay from the same weekday last week."""
        predictions = df.groupby("route")[self.target_col].shift(self.seasonality)
        fill_values = df["route"].map(self.route_median_)
        predictions = predictions.fillna(fill_values)
        return predictions.fillna(self.global_median_)


class MovingAverageBaseline:
    """Rolling window average over the past N days."""

    def __init__(self, window=7, target_col="avg_arr_delay"):
        self.name = f"MovingAverage_{window}"
        self.window = window
        self.target_col = target_col

    def fit(self, train_df):
        """Stores per-route medians as fallback."""
        self.route_median_ = train_df.groupby("route")[self.target_col].median()
        self.global_median_ = train_df[self.target_col].median()
        return self

    def predict(self, df):
        """Rolling mean of the past N days, shifted by 1 to avoid leaking today."""
        predictions = (
            df.groupby("route")[self.target_col]
            .transform(lambda x: x.shift(1).rolling(self.window, min_periods=1).mean())
        )
        fill_values = df["route"].map(self.route_median_)
        predictions = predictions.fillna(fill_values)
        return predictions.fillna(self.global_median_)


class ProphetModel:
    """Prophet time series model, trained per route."""

    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 daily_seasonality=False, add_holidays=True, target_col="avg_arr_delay"):
        self.name = "Prophet"
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.add_holidays = add_holidays
        self.target_col = target_col
        self.models = {}

    def fit(self, train_df):
        """Trains one Prophet model per route."""
        routes = train_df["route"].unique()

        for route in routes:
            route_data = train_df[train_df["route"] == route][["date", self.target_col]].copy()
            route_data.columns = ["ds", "y"]
            route_data["ds"] = pd.to_datetime(route_data["ds"])

            model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
            )

            if self.add_holidays:
                model.add_country_holidays(country_name="US")

            model.fit(route_data)
            self.models[route] = model

    def predict(self, df):
        """Generates predictions for each route using its fitted model."""
        df = df.copy()
        df = df.sort_values(["route", "date"]).reset_index(drop=True)

        predictions = pd.Series(index=df.index, dtype=float)

        for route in df["route"].unique():
            route_mask = df["route"] == route
            route_data = df.loc[route_mask, ["date"]].copy()
            route_data.columns = ["ds"]
            route_data["ds"] = pd.to_datetime(route_data["ds"])

            if route in self.models:
                forecast = self.models[route].predict(route_data)
                predictions.loc[route_mask] = forecast["yhat"].values
            else:
                predictions.loc[route_mask] = 0.0

        return predictions
