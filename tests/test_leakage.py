"""
Leakage guards for FeatureBuilder: anything that happens after train_end_date
must not change route stats, imputation values, or already-built feature rows.
"""

import numpy as np
import pandas as pd

from src.build_features import FeatureBuilder

SPIKE_DATE = "2023-12-01"

ROUTE_STAT_COLS = [
    "route_mean_demand", "route_std_demand", "route_median_demand",
    "route_delay_mean", "route_delay_std",
]


def _spike_post_train(df, weather=False):
    """Corrupts rows on/after SPIKE_DATE hard enough that any leak is visible."""
    out = df.copy()
    post = out["date"] >= pd.Timestamp(SPIKE_DATE)
    out.loc[post, "avg_arr_delay"] += 500.0
    out.loc[post, "flight_count"] += 400.0
    if weather:
        out.loc[post, ["apt1_temp_avg", "apt2_temp_avg"]] += 100.0
        vis_cols = [
            "apt1_visibility_min", "apt2_visibility_min",
            "apt1_visibility_p10", "apt2_visibility_p10",
        ]
        out.loc[post, vis_cols] += 50000.0
    return out


def test_route_stats_unchanged_by_post_train_spike(synthetic_df, train_end):
    """Route stats must ignore post-train corruption and equal a strict train-window recompute."""
    clean_builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    clean = clean_builder.build()
    spiked_builder = FeatureBuilder(_spike_post_train(synthetic_df), train_end_date=train_end)
    spiked = spiked_builder.build()

    for col in ROUTE_STAT_COLS:
        pd.testing.assert_series_equal(clean[col], spiked[col])

    clean_stats = clean_builder.export_state()["route_stats"]
    spiked_stats = spiked_builder.export_state()["route_stats"]
    assert clean_stats == spiked_stats

    # the clean-vs-spiked comparison alone is blind to a leak confined to the
    # unspiked buffer between train_end and SPIKE_DATE, so also pin the stats
    # to values recomputed independently from the strict train window
    train = synthetic_df[synthetic_df["date"] < pd.Timestamp(train_end)]
    demand = train.groupby("route")["flight_count"].agg(["mean", "std", "median"])
    delay = train.groupby("route")["avg_arr_delay"].agg(["mean", "std"])
    for route, stats in clean_stats.items():
        assert stats["route_mean_demand"] == float(demand.loc[route, "mean"])
        assert stats["route_std_demand"] == float(demand.loc[route, "std"])
        assert stats["route_median_demand"] == float(demand.loc[route, "median"])
        assert stats["route_delay_mean"] == float(delay.loc[route, "mean"])
        assert stats["route_delay_std"] == float(delay.loc[route, "std"])


def test_fill_values_unchanged_by_post_train_spike(synthetic_df, train_end):
    """Imputation values serialized in the feature state must ignore post-train data."""
    clean_builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    clean_builder.build()
    spiked_builder = FeatureBuilder(
        _spike_post_train(synthetic_df, weather=True), train_end_date=train_end
    )
    spiked_builder.build()

    clean_state = clean_builder.export_state()
    spiked_state = spiked_builder.export_state()
    assert clean_state["temp_fill_values"] == spiked_state["temp_fill_values"]
    assert clean_state["aviation_fill_values"] == spiked_state["aviation_fill_values"]
    assert clean_state["lag_fill_medians"] == spiked_state["lag_fill_medians"]


def test_rows_before_spike_identical(synthetic_df, train_end):
    """Feature rows dated before the corruption must come out identical."""
    clean = FeatureBuilder(synthetic_df, train_end_date=train_end).build()
    spiked = FeatureBuilder(
        _spike_post_train(synthetic_df, weather=True), train_end_date=train_end
    ).build()

    cutoff = pd.Timestamp(SPIKE_DATE)
    pd.testing.assert_frame_equal(
        clean[clean["date"] < cutoff].reset_index(drop=True),
        spiked[spiked["date"] < cutoff].reset_index(drop=True),
    )


def test_temp_fill_uses_train_median_only(synthetic_df, train_end, built):
    """Temperature NaNs must be filled with the train-window median, not the full-data one."""
    _, builder = built
    state = builder.export_state()

    train_mask = synthetic_df["date"] < pd.Timestamp(train_end)
    for col in ["apt1_temp_avg", "apt2_temp_avg"]:
        train_median = synthetic_df.loc[train_mask, col].median()
        full_median = synthetic_df[col].median()
        assert state["temp_fill_values"][col] == float(train_median)
        # the fixture shifts post-train temps, so a full-data leak would differ
        assert train_median != full_median


def test_hub_state_aggregates_only_past_days_across_routes():
    """Hub features must average sibling routes' delays from strictly prior days."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    rng = np.random.default_rng(7)
    frames = []
    # two routes share destination HHH, one is unrelated
    for route, offset in [("AAA-HHH", 5.0), ("BBB-HHH", 20.0), ("CCC-DDD", 40.0)]:
        frames.append(pd.DataFrame({
            "date": dates,
            "route": route,
            "avg_arr_delay": offset + rng.normal(0.0, 2.0, size=len(dates)),
        }))
    df = pd.concat(frames, ignore_index=True)

    builder = FeatureBuilder(df)
    builder.add_hub_features()
    out = builder.df

    hub_actual = (
        df.assign(dest=df["route"].str.split("-").str[1])
        .groupby(["dest", "date"])["avg_arr_delay"]
        .mean()
        .unstack(level=0)
    )

    row = out[(out["route"] == "AAA-HHH") & (out["date"] == dates[10])].iloc[0]
    expected_lag1 = hub_actual.loc[dates[9], "HHH"]
    expected_roll7 = hub_actual.loc[dates[3]:dates[9], "HHH"].mean()
    assert np.isclose(row["hub_inbound_lag_1"], expected_lag1)
    assert np.isclose(row["hub_inbound_roll_7"], expected_roll7)

    # the unrelated route must see only its own destination's history
    other = out[(out["route"] == "CCC-DDD") & (out["date"] == dates[10])].iloc[0]
    assert np.isclose(other["hub_inbound_lag_1"], hub_actual.loc[dates[9], "DDD"])

    # first day has no past, heads are left for the median fill
    first = out[(out["route"] == "AAA-HHH") & (out["date"] == dates[0])].iloc[0]
    assert pd.isna(first["hub_inbound_lag_1"])


def test_hub_state_filled_from_train_medians(built):
    """Hub head-NaNs must take the per-route median-fill path like other lags."""
    features, builder = built
    state = builder.export_state()

    for col in ["hub_inbound_lag_1", "hub_inbound_roll_7"]:
        assert features[col].notna().all()
        assert set(state["lag_fill_medians"][col]) == set(features["route"].unique())


def test_visibility_fill_uses_train_median_only(synthetic_df, train_end, built):
    """Visibility NaNs must be filled with the train-window median, not the full-data one."""
    _, builder = built
    state = builder.export_state()

    train_mask = synthetic_df["date"] < pd.Timestamp(train_end)
    for col in [
        "apt1_visibility_min", "apt2_visibility_min",
        "apt1_visibility_p10", "apt2_visibility_p10",
    ]:
        train_median = synthetic_df.loc[train_mask, col].median()
        full_median = synthetic_df[col].median()
        assert state["aviation_fill_values"][col] == float(train_median)
        # the fixture drops post-train visibility, so a full-data leak would differ
        assert train_median != full_median


def test_lag_no_cross_route_bleed(synthetic_df, train_end, built):
    """Every lag column must be a within-route shift, never crossing route boundaries."""
    builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    builder.add_lag_features()
    df = builder.df

    for _, group in df.groupby("route"):
        expected = group["avg_arr_delay"].shift(1)
        pd.testing.assert_series_equal(group["lag_1_arr_delay"], expected, check_names=False)
        # a bleed would pull the previous route's last row into this head
        assert np.isnan(group["lag_1_arr_delay"].iloc[0])
        assert group["lag_28_arr_delay"].iloc[:28].isna().all()
        assert group["lag_28_arr_delay"].iloc[28] == group["avg_arr_delay"].iloc[0]

    # weather lags come from the same grouped-shift pattern in
    # add_weather_features, recompute them per route from the built output
    features, _ = built
    for _, group in features.groupby("route"):
        for lag in [1, 3, 7]:
            expected = group["weather_severity_max"].shift(lag).fillna(0)
            pd.testing.assert_series_equal(
                group[f"weather_severity_lag_{lag}"], expected, check_names=False
            )


def test_rolling_shift1_semantics(synthetic_df, train_end):
    """Rolling and ewm features must aggregate strictly past values, shifted before rolling."""
    builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    builder.add_lag_features()
    sub = builder.df[builder.df["route"] == "AAA-BBB"].reset_index(drop=True)
    values = sub["avg_arr_delay"].to_numpy()

    for t in [20, 50, 200]:
        assert np.isclose(sub.loc[t, "rolling_mean_7_arr_delay"], values[t - 7:t].mean())
        assert np.isclose(sub.loc[t, "rolling_std_14_arr_delay"], pd.Series(values[t - 14:t]).std())

    expected_ewm = sub["avg_arr_delay"].shift(1).ewm(span=7, min_periods=1).mean()
    pd.testing.assert_series_equal(sub["ewm_7_arr_delay"], expected_ewm, check_names=False)

    # today's own value must never enter today's rolling window
    target_date = pd.Timestamp("2023-06-15")
    bumped = synthetic_df.copy()
    row = (bumped["route"] == "AAA-BBB") & (bumped["date"] == target_date)
    bumped.loc[row, "avg_arr_delay"] = 1e6

    bumped_builder = FeatureBuilder(bumped, train_end_date=train_end)
    bumped_builder.add_lag_features()
    bumped_sub = bumped_builder.df[bumped_builder.df["route"] == "AAA-BBB"]
    bumped_value = bumped_sub.loc[
        bumped_sub["date"] == target_date, "rolling_mean_7_arr_delay"
    ].iloc[0]
    clean_value = sub.loc[sub["date"] == target_date, "rolling_mean_7_arr_delay"].iloc[0]
    assert bumped_value == clean_value


def test_lag_fill_medians_from_train_window_only(synthetic_df, train_end, built):
    """Lag-column NaNs must be filled with per-route medians computed on the train window."""
    features, builder = built
    state = builder.export_state()

    lag_builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    lag_builder.add_lag_features()
    train_rows = lag_builder.df[lag_builder.df["date"] < pd.Timestamp(train_end)]
    expected = train_rows.groupby("route")["lag_1_arr_delay"].median()
    for route, value in expected.items():
        assert state["lag_fill_medians"]["lag_1_arr_delay"][route] == float(value)

    # the endswith("_std") branch in fill_missing_values never matches the
    # rolling_std_*_arr_delay names, so std columns take the median-fill path
    # like every other lag column -- pinned here because serving relies on
    # these medians being present in the exported state
    std_fills = state["lag_fill_medians"]["rolling_std_7_arr_delay"]
    for route, group in features.groupby("route"):
        assert group["rolling_std_7_arr_delay"].iloc[0] == std_fills[route]

    lag_cols = [c for c in features.columns if c.startswith(("lag_", "rolling_", "ewm_"))]
    assert features[lag_cols].notna().all().all()
