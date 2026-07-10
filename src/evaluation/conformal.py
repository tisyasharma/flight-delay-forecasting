"""
Split-conformal calibration for quantile intervals (Conformalized Quantile
Regression, Romano et al. 2019). The three quantile models are trained
independently and their raw 80% interval under-covers; wrapping them with a
held-out calibration set restores a finite-sample, distribution-free coverage
guarantee while keeping the interval adaptive to each row.

The calibration set must come from the period after training and before the
evaluation window so the guarantee is honest.
"""

import numpy as np


def cqr_offset(y_cal, lower_cal, upper_cal, alpha=0.2):
    """
    Additive offset that restores at least 1 - alpha marginal coverage to the
    interval [lower, upper]. The per-point conformity score is
    max(lower - y, y - upper) (positive when the point falls outside the
    interval, negative when comfortably inside); the offset is the conformal
    quantile of those scores with the finite-sample correction. A negative
    offset legitimately tightens intervals that were over-covering.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    lower_cal = np.asarray(lower_cal, dtype=float)
    upper_cal = np.asarray(upper_cal, dtype=float)

    mask = ~(np.isnan(y_cal) | np.isnan(lower_cal) | np.isnan(upper_cal))
    scores = np.maximum(lower_cal[mask] - y_cal[mask], y_cal[mask] - upper_cal[mask])
    n = len(scores)
    if n == 0:
        return 0.0

    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_cqr(lower, upper, offset):
    """Shifts the interval bounds outward by the conformal offset."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return lower - offset, upper + offset


def split_calibration_window(val_df, date_col="date"):
    """
    Splits a validation frame by date into an early-stopping half and a
    later, dedicated calibration half. Fitting the conformal offset on the
    same rows that drove early stopping reuses data across model selection
    and calibration, which quietly weakens the guarantee; the split keeps
    the calibration set untouched by any selection decision. The later half
    is used for calibration because it sits closest to the evaluation window.
    """
    dates = val_df[date_col]
    midpoint = dates.min() + (dates.max() - dates.min()) / 2
    early = val_df[dates < midpoint]
    late = val_df[dates >= midpoint]
    return early, late
