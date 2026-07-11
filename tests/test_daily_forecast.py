"""
Tests for the daily serving pipeline's guards: the offsets-vs-base floor,
the validated-depth ceiling, and the one-vintage-per-day check that keeps
the published dashboard and the append-only graded record identical. These
are the checks that keep a broken deployment from serving, so they need
regression coverage independent of the scheduled run.
"""

import json

import pandas as pd
import pytest

from src.pipelines.daily_forecast import (
    ensure_depth_validated,
    validate_offsets,
    vintage_logged,
)


def test_validate_offsets_requires_dominating_base():
    offsets = {1: 1.5, 2: 2.0, 3: 4.5}

    validate_offsets(offsets, floor=1.2)

    with pytest.raises(RuntimeError, match="below the serving base"):
        validate_offsets(offsets, floor=1.8)


def test_depth_beyond_validated_offsets_refuses_to_serve():
    offsets = {k: 1.5 + 0.05 * k for k in range(1, 51)}

    ensure_depth_validated(50, offsets)

    with pytest.raises(RuntimeError, match="beyond the deepest"):
        ensure_depth_validated(51, offsets)


def test_vintage_logged_matches_only_todays_date(tmp_path):
    log_path = tmp_path / "2026-07.ndjson"
    today = pd.Timestamp("2026-07-11")

    assert not vintage_logged(log_path, today)

    with open(log_path, "w") as f:
        f.write(json.dumps({"generated_at": "2026-07-10T09:35:00+00:00"}) + "\n")
    assert not vintage_logged(log_path, today)

    with open(log_path, "a") as f:
        f.write(json.dumps({"generated_at": "2026-07-11T09:35:00+00:00"}) + "\n")
    assert vintage_logged(log_path, today)
