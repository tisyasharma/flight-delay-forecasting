"""
Parity between RecursiveForecaster's feature recomputation and FeatureBuilder
is the single guard that makes the forward forecasts trustworthy: if the
recomputed lag/rolling/ewm/hub columns drift from what the models saw in
training, every rolled prediction is silently wrong.

One synthetic route is renamed so two routes share a destination airport,
otherwise every hub mean would be a single route's own series and the pooling
path would go untested.
"""

import numpy as np
import pandas as pd
import pytest

from src.build_features import FeatureBuilder
from src.forecasting.recursive import TARGET_DEPENDENT, RecursiveForecaster

N_ROLL = 40


class _StubModel:
    """Returns a constant, prediction values are irrelevant to feature parity."""

    def __init__(self, value=0.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _OracleModel:
    """
    Feeds the true actuals back as predictions, one day per call in step
    order. With a perfect model the recursion's mixed series equals the
    actual series, so recomputed features must match the built table at
    every depth, not just the first step.
    """

    def __init__(self, day_arrays):
        self._days = iter(day_arrays)

    def predict(self, X):
        return next(self._days)


@pytest.fixture
def shared_dest(synthetic_df, train_end):
    """Features and builder over routes where BBB has two inbound routes."""
    df = synthetic_df.copy()
    df.loc[df["route"] == "CCC-DDD", "route"] = "CCC-BBB"
    builder = FeatureBuilder(df, train_end_date=train_end)
    features = builder.build()
    return features, builder.export_state()


def _actual_grid(features, dates):
    """Route-sorted target arrays for the given dates, one array per day."""
    rows = features[features["date"].isin(dates)].sort_values(["date", "route"])
    return [g["avg_arr_delay"].values for _, g in rows.groupby("date", sort=True)]


def test_first_step_features_match_built_table(shared_dest):
    features, state = shared_dest
    last_actual = features["date"].max() - pd.Timedelta(days=1)

    models = {alpha: _StubModel() for alpha in (0.1, 0.5, 0.9)}
    forecaster = RecursiveForecaster(models, state)
    out = forecaster.forecast(features, last_actual, n_days=1, return_features=True)

    first_day = features[features["date"] == last_actual + pd.Timedelta(days=1)]
    first_day = first_day.sort_values("route")

    for col in TARGET_DEPENDENT:
        assert np.allclose(out[col].values, first_day[col].values), col


def test_oracle_recursion_matches_built_table_at_depth(shared_dest):
    features, state = shared_dest
    horizon = pd.date_range(
        features["date"].max() - pd.Timedelta(days=N_ROLL - 1),
        features["date"].max(),
        freq="D",
    )
    last_actual = horizon[0] - pd.Timedelta(days=1)
    actuals_by_day = _actual_grid(features, horizon)

    models = {alpha: _OracleModel(actuals_by_day) for alpha in (0.1, 0.5, 0.9)}
    forecaster = RecursiveForecaster(models, state)
    out = forecaster.forecast(features, last_actual, n_days=N_ROLL, return_features=True)

    expected = features[features["date"].isin(horizon)].sort_values(["date", "route"])
    assert len(out) == len(expected)
    for col in TARGET_DEPENDENT:
        assert np.allclose(out[col].values, expected[col].values), col
    # the oracle's forecasts are the actuals themselves
    assert np.allclose(out["q50"].values, expected["avg_arr_delay"].values)


def test_conformal_offsets_widen_by_depth(shared_dest):
    features, state = shared_dest
    last_actual = features["date"].max() - pd.Timedelta(days=3)

    models = {0.1: _StubModel(-5.0), 0.5: _StubModel(0.0), 0.9: _StubModel(5.0)}
    forecaster = RecursiveForecaster(models, state)
    offsets = {1: 1.0, 2: 2.5}
    out = forecaster.forecast(features, last_actual, n_days=3, conformal_offset=offsets)

    by_k = out.groupby("k").first()
    assert np.isclose(by_k.loc[1, "lo_cal"], -5.0 - 1.0)
    assert np.isclose(by_k.loc[2, "hi_cal"], 5.0 + 2.5)
    # depths beyond the deepest fitted offset reuse the deepest value
    assert np.isclose(by_k.loc[3, "hi_cal"], 5.0 + 2.5)
    # quantile crossing is repaired by sorting
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()


def test_forecast_validates_inputs(shared_dest):
    features, state = shared_dest
    models = {alpha: _StubModel() for alpha in (0.1, 0.5, 0.9)}

    with pytest.raises(ValueError, match="0.5 quantile"):
        RecursiveForecaster({0.1: _StubModel()}, state)
    with pytest.raises(ValueError, match="window"):
        RecursiveForecaster(models, state, window=30)

    forecaster = RecursiveForecaster(models, state)
    missing_route = features[features["route"] != "AAA-BBB"]
    with pytest.raises(ValueError, match="routes"):
        forecaster.forecast(missing_route, features["date"].max(), n_days=1)

    holed = features.copy()
    last_actual = holed["date"].max() - pd.Timedelta(days=1)
    holed.loc[
        (holed["route"] == "AAA-BBB") & (holed["date"] == last_actual), "avg_arr_delay"
    ] = np.nan
    with pytest.raises(ValueError, match="NaN target"):
        forecaster.forecast(holed, last_actual, n_days=1)
