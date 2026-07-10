"""
The PSI module must flag genuine level shifts while staying quiet through
pure seasonality, and never blow up on degenerate inputs.
"""

import numpy as np
import pandas as pd

from src.monitoring.psi import month_matched_psi, psi, quantile_bin_edges


def _seasonal_frame(shift_after=None, shift=0.0):
    """
    Four years of a daily seasonal series observed at 10 stations. Multiple
    observations per day give each month enough points for stable bins, the
    same reason the real checks pool across airports and routes.
    """
    rng = np.random.default_rng(7)
    days = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    dates = days.repeat(10)
    seasonal = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    values = seasonal + rng.normal(0, 2, len(dates))
    if shift_after is not None:
        values = values + np.where(dates >= pd.Timestamp(shift_after), shift, 0.0)
    return dates, values


def test_quantile_bin_edges_known_values():
    """Interior quintile edges of 1..10 are hand-checkable."""
    edges = quantile_bin_edges(np.arange(1.0, 11.0), n_bins=5)
    assert np.allclose(edges, [2.8, 4.6, 6.4, 8.2])


def test_psi_identical_samples_is_zero():
    values = np.arange(1000, dtype=float)
    assert psi(values, values) < 1e-6


def test_psi_degenerate_inputs_stay_finite():
    """Constant series, single points, and empty bins must never raise."""
    assert np.isfinite(psi(np.ones(100), np.ones(100)))
    assert np.isfinite(psi(np.ones(100), np.full(100, 5.0)))
    assert np.isfinite(psi(np.arange(100.0), np.full(50, 1e9)))
    assert np.isnan(psi(np.array([]), np.arange(10.0)))


def test_month_matched_psi_quiet_through_seasonality():
    """A purely seasonal series must score near zero month-matched."""
    dates, values = _seasonal_frame()
    result = month_matched_psi(dates, values, "2024-07-01", "2025-01-01")
    assert len(result["months"]) == 6
    assert result["mean"] < 0.1


def test_month_matched_psi_catches_level_shift():
    """An injected upstream-style level shift must breach the 0.25 threshold."""
    dates, values = _seasonal_frame(shift_after="2024-07-01", shift=-8.0)
    result = month_matched_psi(dates, values, "2024-07-01", "2025-01-01")
    assert result["mean"] > 0.25


def test_month_matched_psi_skips_thin_references():
    """A month with no prior-year history is skipped, not scored on noise."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    values = np.arange(len(dates), dtype=float)
    result = month_matched_psi(dates, values, "2024-01-01", "2024-04-01")
    assert result["months"] == {}
    assert np.isnan(result["mean"])
