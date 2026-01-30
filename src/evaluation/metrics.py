"""
Evaluation metrics for delay prediction models.
"""

from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error - penalizes large errors more than MAE."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in the same units as the target."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean absolute percentage error.
    Skips values where |actual| < epsilon since MAPE blows up near zero.
    Returns NaN if more than half the values are filtered out.
    """
    valid_mask = np.abs(y_true) > epsilon

    if valid_mask.sum() < len(y_true) * 0.5:
        return np.nan

    return np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared - fraction of variance explained by the model."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # drop NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "n_samples": len(y_true)
    }


def calculate_delay_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            delay_threshold: float = 15) -> Dict[str, float]:
    """
    Delay-specific metrics for model evaluation.

    Primary: MAE, RMSE, within-15 (% of predictions within 15 min of actual)
    Secondary: directional accuracy, median error, R²
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "mae": None, "within_15": None, "rmse": None,
            "mape": None, "median_ae": None, "directional": None, "r2": None
        }

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae_val = float(np.mean(abs_errors))
    within_15 = float(np.mean(abs_errors <= 15) * 100)
    rmse_val = float(np.sqrt(np.mean(errors ** 2)))

    # mape: skip near-zero actuals (|actual| < 1 min) to avoid division issues
    valid_mask = np.abs(y_true) > 1.0
    if np.sum(valid_mask) > len(y_true) * 0.5:
        mape_val = float(np.mean(np.abs(errors[valid_mask] / y_true[valid_mask])) * 100)
    else:
        mape_val = None

    median_ae = float(np.median(abs_errors))

    # directional: did we correctly classify as delayed (> threshold) or not?
    actual_delayed = y_true > delay_threshold
    pred_delayed = y_pred > delay_threshold
    directional = float(np.mean(actual_delayed == pred_delayed) * 100)

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_val = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": mae_val,
        "within_15": within_15,
        "rmse": rmse_val,
        "mape": mape_val,
        "median_ae": median_ae,
        "directional": directional,
        "r2": r2_val
    }


def calculate_metrics_by_segment(y_true: np.ndarray, y_pred: np.ndarray,
                                  segments: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Calculate metrics separately for each segment value."""
    results = {}

    for segment in np.unique(segments):
        mask = segments == segment
        if np.sum(mask) > 0:
            results[str(segment)] = calculate_metrics(
                y_true[mask], y_pred[mask]
            )

    return results
