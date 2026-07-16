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


def pinball_loss(y_true, y_pred, alpha):
    """Quantile (pinball) loss, penalizes over- and under-prediction asymmetrically."""
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))


def interval_coverage(y_true, lower, upper):
    """Percentage of actuals falling inside [lower, upper]."""
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100


def interval_width(lower, upper):
    """Average width of the prediction interval."""
    return np.mean(upper - lower)


def _wis_alphas_valid(alphas, tol=1e-9):
    """
    True when the sorted quantile levels pair to 1.0 around a present 0.5
    median, the precondition for the 2x-mean-pinball WIS identity.
    """
    levels = sorted(float(a) for a in alphas)
    if not any(abs(a - 0.5) <= tol for a in levels):
        return False
    return all(abs(lo + hi - 1.0) <= tol for lo, hi in zip(levels, reversed(levels), strict=True))


def weighted_interval_score(y_true, quantile_preds, alphas=(0.1, 0.5, 0.9)):
    """
    Weighted interval score (Bracher et al. 2021) computed as twice the mean
    pinball loss across the set. That identity only holds when the quantile
    levels are symmetric around 0.5 and include the median, anything else
    raises ValueError. WIS generalizes absolute error, so a point forecast
    (all quantiles equal) scores its MAE.
    """
    if not _wis_alphas_valid(alphas):
        raise ValueError(
            "weighted_interval_score needs quantile levels symmetric around 0.5 "
            f"including the median, got {tuple(alphas)}"
        )
    y_true = np.asarray(y_true, dtype=float).flatten()
    preds = np.asarray(quantile_preds, dtype=float)
    losses = [pinball_loss(y_true, preds[:, i], alpha) for i, alpha in enumerate(alphas)]
    return 2.0 * float(np.mean(losses))


def sort_quantile_predictions(quantile_preds):
    """
    Sorts each row's quantile predictions into ascending order. Independently
    trained quantile models can cross (q10 above q90 on individual rows), and
    row-wise sorting is the standard non-crossing fix.
    """
    return np.sort(np.asarray(quantile_preds, dtype=float), axis=1)


def calculate_quantile_metrics(y_true, quantile_preds, alphas=(0.1, 0.5, 0.9)):
    """
    Pinball loss per quantile, coverage and width of the outer interval, and
    per-quantile calibration (the empirical fraction of actuals at or below
    each predicted quantile, ideally 100 * alpha).
    quantile_preds is (n, len(alphas)) with columns ordered like alphas.
    Same conventions as calculate_delay_metrics: rows with a NaN actual or a
    NaN in any quantile column are masked, empty input returns all-None.
    The wis key is None when the alphas are not symmetric around a present
    median, since the pinball identity behind it does not hold there.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    preds = np.asarray(quantile_preds, dtype=float)
    wis_valid = _wis_alphas_valid(alphas)

    pinball_keys = [f"pinball_{int(round(a * 100))}" for a in alphas]
    below_keys = [f"below_q{int(round(a * 100))}" for a in alphas]
    coverage_key = f"coverage_{int(round((alphas[-1] - alphas[0]) * 100))}"

    mask = ~(np.isnan(y_true) | np.isnan(preds).any(axis=1))
    y_true = y_true[mask]
    preds = preds[mask]

    if len(y_true) == 0:
        result = {key: None for key in pinball_keys + below_keys}
        result[coverage_key] = None
        result["interval_width"] = None
        result["wis"] = None
        return result

    result = {
        key: float(pinball_loss(y_true, preds[:, i], alpha))
        for i, (key, alpha) in enumerate(zip(pinball_keys, alphas, strict=True))
    }
    result["wis"] = (
        2.0 * float(np.mean([result[key] for key in pinball_keys])) if wis_valid else None
    )
    for i, key in enumerate(below_keys):
        result[key] = float(np.mean(y_true <= preds[:, i]) * 100)
    result[coverage_key] = float(interval_coverage(y_true, preds[:, 0], preds[:, -1]))
    result["interval_width"] = float(interval_width(preds[:, 0], preds[:, -1]))
    return result


def calculate_classification_metrics(y_true, y_score, threshold=15):
    """
    Decision-framed metrics for flagging bad-delay days, where the day's mean
    delay exceeds the threshold. y_true is the continuous delay (binarized
    here), y_score is the predicted probability of exceeding the threshold.
    PR-AUC is the headline because the positive class is the minority and a
    missed bad-delay day costs more than a false alarm, base rate is reported
    so the lift over chance is legible. Returns an all-None dict when a fold
    has a single class.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=float).flatten()
    y_score = np.asarray(y_score, dtype=float).flatten()

    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_bin = (y_true[mask] > threshold).astype(int)
    y_score = y_score[mask]

    if len(y_bin) == 0 or y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"pr_auc": None, "roc_auc": None, "base_rate": None, "brier": None}

    return {
        "pr_auc": float(average_precision_score(y_bin, y_score)),
        "roc_auc": float(roc_auc_score(y_bin, y_score)),
        "base_rate": float(y_bin.mean() * 100),
        "brier": float(brier_score_loss(y_bin, y_score)),
    }


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
