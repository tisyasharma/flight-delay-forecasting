import warnings

import pandas as pd

from src.models.simple_baselines import (
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalNaiveBaseline,
)

warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = [
    "NaiveBaseline",
    "SeasonalNaiveBaseline",
    "MovingAverageBaseline",
    "ProphetModel",
]


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
        # deferred so importing this module doesn't require prophet installed
        from prophet import Prophet

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
