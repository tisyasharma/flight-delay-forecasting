"""
Population stability index with month-matched references. Seasonal series
score high PSI against a naive full-history reference from seasonality alone,
comparing each month against the same calendar month of prior years keeps the
score sensitive to genuine level shifts (like the Dec 2024 wind seam) while
staying quiet through ordinary seasonal swings.
"""

import numpy as np
import pandas as pd


def quantile_bin_edges(reference, n_bins=10):
    """
    Interior bin edges at reference quantiles. The outermost bins are open,
    so every value falls into some bin. Duplicate edges from degenerate
    distributions are collapsed, a constant reference yields a single bin.
    """
    reference = np.asarray(reference, dtype=float)
    reference = reference[~np.isnan(reference)]
    if len(reference) == 0:
        return np.array([])
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    return edges[1:-1]


def psi(reference, actual, n_bins=10, epsilon=1e-4):
    """
    PSI between a reference and an actual sample over quantile bins of the
    reference. Proportions are smoothed with epsilon so empty bins never
    produce infinities. Note a constant reference collapses to one bin and
    scores 0 by construction.
    """
    reference = np.asarray(reference, dtype=float)
    reference = reference[~np.isnan(reference)]
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]
    if len(reference) == 0 or len(actual) == 0:
        return float("nan")

    edges = quantile_bin_edges(reference, n_bins)
    n_total_bins = len(edges) + 1

    ref_counts = np.bincount(
        np.searchsorted(edges, reference, side="right"), minlength=n_total_bins
    )
    act_counts = np.bincount(
        np.searchsorted(edges, actual, side="right"), minlength=n_total_bins
    )

    ref_prop = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * n_total_bins)
    act_prop = (act_counts + epsilon) / (act_counts.sum() + epsilon * n_total_bins)

    return float(np.sum((act_prop - ref_prop) * np.log(act_prop / ref_prop)))


def month_matched_psi(dates, values, window_start, window_end, n_bins=10,
                      min_reference=30, min_actual=30):
    """
    Scores each calendar month inside [window_start, window_end) against the
    same calendar month from all earlier years. Months are skipped when either
    the reference or the actual side is thinner than its minimum, so a partial
    edge month cannot produce a noisy score. Returns per-month scores plus
    their mean and max.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(dates))
    series = series.dropna()
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    window = series[(series.index >= window_start) & (series.index < window_end)]

    months = {}
    for period, month_values in window.groupby(window.index.to_period("M")):
        reference = series[
            (series.index.month == period.month) & (series.index < period.to_timestamp())
        ]
        if len(reference) < min_reference or len(month_values) < min_actual:
            continue
        months[str(period)] = psi(reference.values, month_values.values, n_bins=n_bins)

    scores = list(months.values())
    return {
        "months": months,
        "mean": float(np.mean(scores)) if scores else float("nan"),
        "max": float(np.max(scores)) if scores else float("nan"),
    }
