"""
Dependency-light baselines over route/date dataframes, importable without
torch.
"""


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
