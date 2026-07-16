"""
Pins the numeric contracts of src.evaluation.metrics: known values, the mape
epsilon filter, NaN masking, and the empty-input conventions callers rely on.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_delay_metrics,
    calculate_metrics,
    calculate_metrics_by_segment,
    calculate_quantile_metrics,
    interval_coverage,
    interval_width,
    mae,
    mape,
    pinball_loss,
    r_squared,
    rmse,
    sort_quantile_predictions,
    weighted_interval_score,
)


def test_classification_metrics_perfect_and_baserate():
    """A perfect score gives PR-AUC 1.0 and reports the base rate as a percent."""
    delay = np.array([5.0, 20.0, 8.0, 30.0, 2.0])  # 2 of 5 exceed 15 min
    score = np.array([0.1, 0.9, 0.2, 0.95, 0.05])
    m = calculate_classification_metrics(delay, score, threshold=15)
    assert m["pr_auc"] == 1.0
    assert m["base_rate"] == 40.0
    assert 0.0 <= m["brier"] <= 1.0


def test_classification_metrics_single_class_returns_none():
    """A fold where nothing crosses the threshold cannot be scored."""
    delay = np.array([1.0, 2.0, 3.0])
    m = calculate_classification_metrics(delay, np.array([0.2, 0.3, 0.1]), threshold=15)
    assert m == {"pr_auc": None, "roc_auc": None, "base_rate": None, "brier": None}


def test_rmse_and_mae_known_values():
    """Hand-computed values for the two core error metrics."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 6.0])
    assert np.isclose(mae(y_true, y_pred), 2.0)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(14.0 / 3.0))


def test_mape_nan_when_majority_near_zero():
    """MAPE degrades to NaN when most actuals sit inside the epsilon band."""
    # 3 of 4 actuals fall inside the |y| <= 1 epsilon band
    y_true = np.array([0.5, 0.5, 0.5, 20.0])
    y_pred = np.array([1.0, 1.0, 1.0, 22.0])
    assert np.isnan(mape(y_true, y_pred))


def test_mape_finite_at_exact_half_boundary():
    """Exactly half valid is NOT filtered, the cutoff is strictly less than half."""
    y_true = np.array([0.5, 0.5, 10.0, 20.0])
    y_pred = np.array([1.0, 1.0, 11.0, 22.0])
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_mape_ignores_near_zero_actuals():
    """The wild prediction sits on a filtered entry and must not matter."""
    y_true = np.array([10.0, 0.5, 20.0])
    y_pred = np.array([11.0, 100.0, 18.0])
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_r_squared_zero_variance_returns_zero():
    """Zero-variance actuals return exactly 0.0 instead of dividing by zero."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert r_squared(y_true, y_pred) == 0.0


def test_calculate_metrics_masks_nan_pairwise():
    """A NaN on either side drops the pair, n_samples reports what survived."""
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.nan, 5.0])
    result = calculate_metrics(y_true, y_pred)
    assert result["n_samples"] == 2
    assert np.isclose(result["mae"], 0.5)
    assert np.isclose(result["rmse"], np.sqrt(0.5))


def test_calculate_metrics_empty_after_mask_warns():
    """Documented no-guard contract: empty input warns and yields NaN metrics."""
    y_true = np.array([np.nan, np.nan, np.nan])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.warns(RuntimeWarning):
        result = calculate_metrics(y_true, y_pred)
    assert np.isnan(result["rmse"])
    assert result["n_samples"] == 0


def test_calculate_delay_metrics_all_nan_returns_none_dict():
    """All-NaN input returns the documented all-None dict rather than raising."""
    y_true = np.full(4, np.nan)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    expected = {
        "mae": None, "within_15": None, "rmse": None, "mape": None,
        "median_ae": None, "threshold_acc": None, "r2": None,
    }
    assert calculate_delay_metrics(y_true, y_pred) == expected


def test_calculate_delay_metrics_known_values():
    """Hand-checked hit rate, threshold accuracy, and median error, with mape mapped to None."""
    y_true = np.array([0.0, 0.5, 0.75, 60.0])
    y_pred = np.array([10.0, 0.5, 16.75, 50.0])
    result = calculate_delay_metrics(y_true, y_pred)

    # absolute errors are exactly [10, 0, 16, 10]
    assert result["within_15"] == 75.0
    assert result["median_ae"] == 10.0
    assert result["threshold_acc"] == 75.0
    assert np.isclose(result["mae"], 9.0)
    # three of four actuals are near zero, so NaN mape maps to None
    assert result["mape"] is None


def test_calculate_metrics_by_segment_matches_manual_slices():
    """Segment grouping must slice exactly like manual boolean masks."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y_pred = np.array([12.0, 18.0, 33.0, 44.0, 45.0, 66.0])
    segments = np.array(["east", "east", "east", "west", "west", "west"])

    results = calculate_metrics_by_segment(y_true, y_pred, segments)

    assert set(results) == {"east", "west"}
    assert results["east"] == calculate_metrics(y_true[:3], y_pred[:3])
    assert results["west"] == calculate_metrics(y_true[3:], y_pred[3:])


def test_pinball_loss_asymmetry_known_values():
    """Hand-checked pinball values on both sides of the actual."""
    y_true = np.array([10.0])
    assert np.isclose(pinball_loss(y_true, np.array([8.0]), 0.1), 0.2)
    assert np.isclose(pinball_loss(y_true, np.array([12.0]), 0.1), 1.8)
    assert np.isclose(pinball_loss(y_true, np.array([8.0]), 0.9), 1.8)
    assert np.isclose(pinball_loss(y_true, np.array([12.0]), 0.9), 0.2)


def test_pinball_loss_median_is_half_mae():
    """At alpha 0.5 the pinball loss reduces to half the absolute error."""
    y_true = np.array([1.0, 5.0, 10.0])
    y_pred = np.array([2.0, 3.0, 10.0])
    assert np.isclose(pinball_loss(y_true, y_pred, 0.5), mae(y_true, y_pred) / 2)


def test_interval_coverage_and_width_known_values():
    """Hand-checked coverage percentage (boundaries inclusive) and mean width."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 3.0, 2.0, 5.0])
    upper = np.array([2.0, 4.0, 4.0, 6.0])
    assert interval_coverage(y_true, lower, upper) == 50.0
    assert np.isclose(interval_width(lower, upper), 1.5)


def test_sort_quantile_predictions_fixes_crossing():
    """Crossed rows get reordered, well-ordered rows pass through unchanged."""
    preds = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    fixed = sort_quantile_predictions(preds)
    assert (fixed == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])).all()
    assert (np.diff(fixed, axis=1) >= 0).all()


def test_calculate_quantile_metrics_known_values():
    """Hand-checked pinball, coverage, and width on a tiny sorted matrix."""
    y_true = np.array([10.0, 20.0])
    preds = np.array([
        [8.0, 10.0, 14.0],
        [22.0, 25.0, 30.0],
    ])
    result = calculate_quantile_metrics(y_true, preds)

    # first actual inside [8, 14], second outside [22, 30]
    assert result["coverage_80"] == 50.0
    assert np.isclose(result["interval_width"], 7.0)
    assert np.isclose(result["pinball_10"], 1.0)
    assert np.isclose(result["pinball_50"], 1.25)
    assert np.isclose(result["pinball_90"], 0.7)


def test_calculate_quantile_metrics_masks_nan_rows():
    """A NaN in the actual or any quantile column drops that whole row."""
    y_true = np.array([10.0, np.nan, 30.0])
    preds = np.array([
        [8.0, 10.0, 14.0],
        [1.0, 2.0, 3.0],
        [np.nan, 29.0, 33.0],
    ])
    result = calculate_quantile_metrics(y_true, preds)
    clean = calculate_quantile_metrics(np.array([10.0]), np.array([[8.0, 10.0, 14.0]]))
    assert result == clean


def test_calculate_quantile_metrics_key_naming_rounds():
    """Alpha 0.95 must produce pinball_95, float truncation would give 94."""
    y_true = np.array([10.0])
    preds = np.array([[8.0, 10.0, 12.0]])
    result = calculate_quantile_metrics(y_true, preds, alphas=(0.05, 0.5, 0.95))
    expected_keys = {
        "pinball_5", "pinball_50", "pinball_95",
        "below_q5", "below_q50", "below_q95",
        "coverage_90", "interval_width", "wis",
    }
    assert set(result) == expected_keys


def test_calculate_quantile_metrics_per_quantile_calibration():
    """Hand-checked per-quantile calibration on constant quantile columns."""
    y_true = np.arange(10, dtype=float)
    # 2 of 10 actuals at or below q10, 5 at or below q50, 9 at or below q90
    preds = np.tile([1.0, 4.0, 8.0], (10, 1))
    result = calculate_quantile_metrics(y_true, preds)

    assert result["below_q10"] == 20.0
    assert result["below_q50"] == 50.0
    assert result["below_q90"] == 90.0


def test_calculate_quantile_metrics_all_nan_returns_none_dict():
    """Nothing surviving the mask returns the documented all-None dict."""
    y_true = np.array([np.nan, np.nan])
    preds = np.ones((2, 3))
    expected = {
        "pinball_10": None, "pinball_50": None, "pinball_90": None,
        "below_q10": None, "below_q50": None, "below_q90": None,
        "coverage_80": None, "interval_width": None, "wis": None,
    }
    assert calculate_quantile_metrics(y_true, preds) == expected


def test_weighted_interval_score_hand_computed():
    """
    Single row, y=10 against quantiles (5, 8, 12). The Bracher form gives
    (0.5 * |10-8| + 0.1 * interval score) / 1.5 with IS = (12-5) = 7 since
    the actual falls inside, so WIS = 1.7 / 1.5. The pinball identity must
    produce the same number.
    """
    y_true = np.array([10.0])
    preds = np.array([[5.0, 8.0, 12.0]])
    assert np.isclose(weighted_interval_score(y_true, preds), 1.7 / 1.5)


def test_weighted_interval_score_point_forecast_equals_mae():
    """A degenerate forecast with all quantiles equal scores its MAE."""
    y_true = np.array([10.0, 0.0])
    preds = np.array([[7.0, 7.0, 7.0], [2.0, 2.0, 2.0]])
    assert np.isclose(weighted_interval_score(y_true, preds), 2.5)
    assert np.isclose(weighted_interval_score(y_true, preds), mae(y_true, preds[:, 1]))


def test_calculate_quantile_metrics_wis_matches_pinball_identity():
    """The wis key equals twice the mean of the three pinball losses."""
    y_true = np.arange(10, dtype=float)
    preds = np.tile([1.0, 4.0, 8.0], (10, 1))
    result = calculate_quantile_metrics(y_true, preds)
    pinballs = [result["pinball_10"], result["pinball_50"], result["pinball_90"]]
    assert np.isclose(result["wis"], 2.0 * np.mean(pinballs))


def test_weighted_interval_score_rejects_asymmetric_alphas():
    """Levels that do not pair to 1.0 break the pinball identity and must raise."""
    y_true = np.array([10.0])
    preds = np.array([[5.0, 8.0, 12.0]])
    with pytest.raises(ValueError):
        weighted_interval_score(y_true, preds, alphas=(0.1, 0.5, 0.8))


def test_calculate_quantile_metrics_wis_none_for_asymmetric_alphas():
    """Asymmetric levels set wis to None while the pinball keys stay populated."""
    y_true = np.array([10.0])
    preds = np.array([[5.0, 8.0, 12.0]])
    result = calculate_quantile_metrics(y_true, preds, alphas=(0.1, 0.5, 0.8))
    assert result["wis"] is None
    assert np.isclose(result["pinball_10"], 0.5)
    assert np.isclose(result["pinball_50"], 1.0)
    assert np.isclose(result["pinball_80"], 0.4)
