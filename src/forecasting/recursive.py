"""
Recursive forward forecasting: rolls the target beyond the last known actual
by feeding each day's median prediction back in as the pseudo-actual for the
next day's features.

The backtest table cannot forecast forward because its delay-lag columns are
read straight from actuals. Of the model features, exactly eleven depend on
the target series: the nine delay lag/rolling/ewm columns plus the two hub
inbound-delay columns (the hub state is an aggregate of the same target over
the modeled routes into each destination). Everything else is calendar,
route statistics, or weather, all available across the horizon. This module
recomputes those eleven from the mixed actual+predicted series, one day at a
time across all routes at once, and leaves every other column untouched.

Feature recomputation must match FeatureBuilder exactly, the parity tests in
tests/test_recursive.py pin that equivalence, first-step against the built
table and full-depth via an oracle model. The ewm has infinite memory, so a
finite trailing window truncates it, at the default 90 days the relative
error is (0.75)**89 ~ 1e-11, far inside np.allclose tolerances.
"""

import numpy as np
import pandas as pd

from src.features.registry import FEATURE_GROUPS, TABULAR_FEATURES

TARGET_DEPENDENT = FEATURE_GROUPS["delay_lags"] + FEATURE_GROUPS["hub_state"]

EWM_ALPHA = 2.0 / (7 + 1)  # span=7, matching FeatureBuilder's ewm_7

MIN_WINDOW = 90


class RecursiveForecaster:
    """
    Rolls quantile forecasts forward day by day from the last known actual.

    models: mapping of quantile alpha to a fitted predictor with .predict(X),
    e.g. {0.1: m10, 0.5: m50, 0.9: m90}. The median feeds the recursion, the
    outer quantiles are scored on the same feature rows each step.

    feature_state: the serialized train-time state (feature_state.json). Only
    its route set is used, as an assertion that the frame and the models talk
    about the same routes, statistics are never recomputed here.
    """

    def __init__(self, models, feature_state, features=None, window=MIN_WINDOW):
        if 0.5 not in models:
            raise ValueError("models must include the 0.5 quantile, it feeds the recursion")
        if window < MIN_WINDOW:
            raise ValueError(
                f"window {window} < {MIN_WINDOW}: lag_28 plus shifted rolling_std_14 "
                "need 29 days and the ewm truncation bound is documented at 90"
            )
        self.models = dict(sorted(models.items()))
        self.alphas = list(self.models)
        self.routes = sorted(feature_state["route_codes"])
        self.features = list(features) if features is not None else list(TABULAR_FEATURES)
        self.window = window

    def forecast(self, frame, last_actual_date, n_days, conformal_offset=0.0,
                 return_features=False):
        """
        frame: feature-table rows (route, date, avg_arr_delay plus every model
        feature column) covering a complete route x date grid from at least
        `window` days before last_actual_date through last_actual_date+n_days.
        Target values after last_actual_date are ignored. The target-dependent
        columns of horizon rows are overwritten with recomputed values, all
        other columns are used as-is.

        conformal_offset: scalar, or a dict mapping recursion depth k to an
        offset, depths beyond the deepest fitted key reuse the deepest value.

        Returns one row per route per horizon day: raw sorted quantiles, the
        conformally widened interval, and the recursion depth k.
        """
        last_actual = pd.Timestamp(last_actual_date)
        grid, target, dates = self._build_grid(frame, last_actual, n_days)
        d0 = dates.get_loc(last_actual) + 1

        dests = pd.Index([r.split("-")[1] for r in self.routes])
        dest_groups = {d: np.flatnonzero(dests == d) for d in dests.unique()}
        hub = np.full((len(dests.unique()), len(dates)), np.nan)
        dest_index = pd.Index(sorted(dest_groups))
        for dest, idx in dest_groups.items():
            hub[dest_index.get_loc(dest)] = target[idx].mean(axis=0)
        route_dest_pos = np.array([dest_index.get_loc(d) for d in dests])

        results = []
        for k in range(1, n_days + 1):
            d = d0 + k - 1
            step = self._step_features(target, hub, route_dest_pos, d)

            rows = grid.loc[grid["date"] == dates[d]].copy()
            for col, values in step.items():
                rows[col] = values

            X = rows[self.features].values
            raw = np.column_stack([m.predict(X) for m in self.models.values()])
            raw.sort(axis=1)

            offset = self._offset_at(conformal_offset, k)
            out = pd.DataFrame({
                "route": rows["route"].values,
                "date": dates[d],
                "k": k,
            })
            for j, alpha in enumerate(self.alphas):
                out[f"q{int(round(alpha * 100))}"] = raw[:, j]
            out["lo_cal"] = raw[:, 0] - offset
            out["hi_cal"] = raw[:, -1] + offset
            if return_features:
                for col in TARGET_DEPENDENT:
                    out[col] = step[col]
            results.append(out)

            mid = raw[:, self.alphas.index(0.5)]
            target[:, d] = mid
            for dest, idx in dest_groups.items():
                hub[dest_index.get_loc(dest), d] = target[idx, d].mean()

        return pd.concat(results, ignore_index=True)

    def _build_grid(self, frame, last_actual, n_days):
        """
        Validates the frame and pivots the target into a route x date matrix
        with NaN beyond the last actual.
        """
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        start = last_actual - pd.Timedelta(days=self.window - 1)
        end = last_actual + pd.Timedelta(days=n_days)
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]

        routes = sorted(frame["route"].unique())
        if routes != self.routes:
            raise ValueError(
                "frame routes do not match the feature state: "
                f"missing {sorted(set(self.routes) - set(routes))}, "
                f"unknown {sorted(set(routes) - set(self.routes))}"
            )

        dates = pd.date_range(start, end, freq="D")
        pivot = frame.pivot(index="route", columns="date", values="avg_arr_delay")
        pivot = pivot.reindex(index=self.routes)
        if list(pivot.columns) != list(dates):
            missing = dates.difference(pivot.columns)
            raise ValueError(f"frame is missing dates: {list(missing[:5])}...")

        target = pivot.values.astype(float).copy()
        history = target[:, : dates.get_loc(last_actual) + 1]
        if np.isnan(history).any():
            raise ValueError("NaN target inside the trailing actual window")
        target[:, dates.get_loc(last_actual) + 1:] = np.nan

        frame = frame.sort_values(["date", "route"]).reset_index(drop=True)
        return frame, target, dates

    def _step_features(self, target, hub, route_dest_pos, d):
        """
        The eleven target-dependent features for day index d, each an array
        over routes in sorted-route order. Mirrors FeatureBuilder: lags are
        plain shifts, rolling stats exclude the current day via shift(1), the
        ewm is adjust-weighted (pandas default), and the hub columns lag and
        roll the destination-mean series the same way.
        """
        out = {}
        for lag in (1, 7, 14, 28):
            out[f"lag_{lag}_arr_delay"] = target[:, d - lag]
        for w in (7, 14):
            block = target[:, d - w:d]
            out[f"rolling_mean_{w}_arr_delay"] = block.mean(axis=1)
            out[f"rolling_std_{w}_arr_delay"] = block.std(axis=1, ddof=1)

        history = target[:, :d]
        weights = (1.0 - EWM_ALPHA) ** np.arange(history.shape[1] - 1, -1, -1)
        out["ewm_7_arr_delay"] = history @ weights / weights.sum()

        hub_hist = hub[route_dest_pos]
        out["hub_inbound_lag_1"] = hub_hist[:, d - 1]
        out["hub_inbound_roll_7"] = hub_hist[:, d - 7:d].mean(axis=1)
        return out

    @staticmethod
    def _offset_at(conformal_offset, k):
        """Resolves the widening offset for depth k."""
        if np.isscalar(conformal_offset):
            return float(conformal_offset)
        deepest = max(conformal_offset)
        return float(conformal_offset[min(k, deepest)])
