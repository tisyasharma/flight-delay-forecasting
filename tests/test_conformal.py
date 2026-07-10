"""
Conformalized quantile regression must restore the nominal coverage guarantee
on a held-out set even when the raw quantile interval is miscalibrated.
"""

import numpy as np

from src.evaluation.conformal import apply_cqr, cqr_offset


def _under_covering_split(seed):
    """
    Two independent samples from the same distribution with an interval that is
    deliberately too narrow (covers well below 80 percent before calibration).
    """
    rng = np.random.default_rng(seed)
    y_cal = rng.normal(0, 10, 4000)
    y_test = rng.normal(0, 10, 4000)
    # a symmetric interval of +/- 6 covers ~55% of a N(0,10), well under 80%
    return y_cal, np.full(4000, -6.0), np.full(4000, 6.0), y_test


def test_cqr_restores_nominal_coverage():
    y_cal, lo_cal, hi_cal, y_test = _under_covering_split(0)
    raw_cov = np.mean((y_test >= -6.0) & (y_test <= 6.0)) * 100
    assert raw_cov < 70

    offset = cqr_offset(y_cal, lo_cal, hi_cal, alpha=0.2)
    assert offset > 0

    lo, hi = apply_cqr(np.full_like(y_test, -6.0), np.full_like(y_test, 6.0), offset)
    cal_cov = np.mean((y_test >= lo) & (y_test <= hi)) * 100
    assert cal_cov >= 79


def test_cqr_tightens_over_covering_interval():
    """An interval far wider than needed yields a negative offset."""
    rng = np.random.default_rng(1)
    y_cal = rng.normal(0, 5, 3000)
    offset = cqr_offset(y_cal, np.full(3000, -40.0), np.full(3000, 40.0), alpha=0.2)
    assert offset < 0


def test_cqr_offset_masks_nans_and_handles_empty():
    y = np.array([1.0, np.nan, 3.0])
    lo = np.array([0.0, 0.0, np.nan])
    hi = np.array([2.0, 2.0, 2.0])
    # only the first row is complete, so the call must not raise
    assert np.isfinite(cqr_offset(y, lo, hi))
    assert cqr_offset(np.array([]), np.array([]), np.array([])) == 0.0
