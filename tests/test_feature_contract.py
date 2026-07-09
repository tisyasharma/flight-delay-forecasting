"""
Contract between the feature registry, FeatureBuilder output, and the
serialized train-time state that serving rebuilds features from.
"""

import json

import pandas as pd
import pytest

from src.build_features import FeatureBuilder
from src.features.registry import (
    SEQUENCE_MODEL_FEATURES,
    TABULAR_FEATURES,
    WEATHER_FEATURES,
)

ROUTE_STATE_COLS = [
    "route_encoded", "route_mean_demand", "route_std_demand",
    "route_median_demand", "route_delay_mean", "route_delay_std",
    "apt1_temp_avg", "apt2_temp_avg",
    "apt1_visibility_min", "apt2_visibility_min",
    "apt1_visibility_p10", "apt2_visibility_p10",
]


def test_registry_lists_are_subset_of_built_columns(built):
    """Every registered feature name must exist in FeatureBuilder output."""
    features, _ = built
    columns = set(features.columns)
    assert set(TABULAR_FEATURES) <= columns
    assert set(WEATHER_FEATURES) <= columns
    assert set(SEQUENCE_MODEL_FEATURES) <= columns


def test_feature_list_counts_and_uniqueness():
    """Pins the 80-feature tabular contract and rejects duplicate names."""
    assert len(TABULAR_FEATURES) == 80
    for feature_list in (TABULAR_FEATURES, WEATHER_FEATURES, SEQUENCE_MODEL_FEATURES):
        assert len(feature_list) == len(set(feature_list))


def test_state_round_trip_rebuild_matches(synthetic_df, built):
    """Rebuilding from exported state must reproduce the training features exactly."""
    features, builder = built
    # the dumps/loads round trip proves the state is JSON-serializable
    state = json.loads(json.dumps(builder.export_state()))
    rebuilt = FeatureBuilder(synthetic_df, state=state).build()

    # the fit path encodes routes as int8 category codes while the state path
    # maps to int64, values match, dtypes intentionally do not
    pd.testing.assert_frame_equal(
        features[TABULAR_FEATURES], rebuilt[TABULAR_FEATURES], check_dtype=False
    )


def test_partial_history_rebuild_matches_on_overlap(synthetic_df, built):
    """A rebuild from a truncated history must agree with the full build where they overlap."""
    features, builder = built
    state = builder.export_state()

    cutoff = synthetic_df["date"].max() - pd.Timedelta(days=59)
    partial_raw = synthetic_df[synthetic_df["date"] >= cutoff]
    partial = FeatureBuilder(partial_raw, state=state).build()

    # lag columns are excluded on purpose, their heads differ by design when
    # the history is truncated
    full_overlap = features[features["date"] >= cutoff]
    pd.testing.assert_frame_equal(
        full_overlap.set_index(["route", "date"])[ROUTE_STATE_COLS],
        partial.set_index(["route", "date"])[ROUTE_STATE_COLS],
        check_dtype=False,
    )


def test_unknown_route_with_state_raises(synthetic_df, built):
    """Serving must refuse routes the models never saw."""
    _, builder = built
    state = builder.export_state()

    altered = synthetic_df.copy()
    altered.loc[altered.index[:5], "route"] = "ZZZ-YYY"
    with pytest.raises(ValueError, match="ZZZ-YYY"):
        FeatureBuilder(altered, state=state)


def test_export_state_before_build_raises(synthetic_df, train_end):
    """State export is only meaningful after a build has computed it."""
    builder = FeatureBuilder(synthetic_df, train_end_date=train_end)
    with pytest.raises(RuntimeError, match="build"):
        builder.export_state()
