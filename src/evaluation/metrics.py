import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred, epsilon=1.0):
    """
    Skips values where actual is near zero since MAPE blows up there.
    Returns NaN if too many values get filtered out.
    """
    valid_mask = np.abs(y_true) > epsilon

    if valid_mask.sum() < len(y_true) * 0.5:
        return np.nan

    return np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100


def r_squared(y_true, y_pred):
    """Coefficient of determination, returns 0 if total variance is zero."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def calculate_metrics(y_true, y_pred):
    """Basic regression metrics (RMSE, MAE, MAPE, R2)."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

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


def calculate_delay_metrics(y_true, y_pred, delay_threshold=15):
    """MAE, RMSE, hit rate (within 15 min), threshold accuracy, R-squared."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "mae": None, "within_15": None, "rmse": None,
            "mape": None, "median_ae": None, "threshold_acc": None, "r2": None
        }

    abs_errors = np.abs(y_pred - y_true)

    mae_val = float(mae(y_true, y_pred))
    rmse_val = float(rmse(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    mape_val = float(mape_val) if mape_val is not None and not np.isnan(mape_val) else None
    r2_val = float(r_squared(y_true, y_pred))

    within_15 = float(np.mean(abs_errors <= delay_threshold) * 100)
    median_ae = float(np.median(abs_errors))

    actual_delayed = y_true > delay_threshold
    pred_delayed = y_pred > delay_threshold
    threshold_acc = float(np.mean(actual_delayed == pred_delayed) * 100)

    return {
        "mae": mae_val,
        "within_15": within_15,
        "rmse": rmse_val,
        "mape": mape_val,
        "median_ae": median_ae,
        "threshold_acc": threshold_acc,
        "r2": r2_val
    }


def calculate_metrics_by_segment(y_true, y_pred, segments):
    """Computes metrics separately for each unique segment value."""
    results = {}

    for segment in np.unique(segments):
        mask = segments == segment
        if np.sum(mask) > 0:
            results[str(segment)] = calculate_metrics(
                y_true[mask], y_pred[mask]
            )

    return results
