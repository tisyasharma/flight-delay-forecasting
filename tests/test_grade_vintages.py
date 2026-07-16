"""
Pins the vintage grader's join and scoring behavior: gradeable rows score
against the latest actuals, unpublished dates count as pending, and the
persistence comparator derives each row's origin from date minus k.
"""

import numpy as np
import pandas as pd

from src.evaluation.grade_vintages import grade


def _vintage_frame(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["vintage"] = df["generated_at"].str[:10]
    return df


def _actuals(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_grade_scores_hand_computed_values():
    vintages = _vintage_frame([{
        "generated_at": "2026-07-11T09:30:00+00:00", "model_version": "abc123",
        "route": "A-B", "date": "2026-07-12", "k": 1,
        "q50": 10.0, "lo": 5.0, "hi": 15.0,
    }])
    actuals = _actuals([
        {"route": "A-B", "date": "2026-07-11", "avg_arr_delay": 8.0},
        {"route": "A-B", "date": "2026-07-12", "avg_arr_delay": 12.0},
    ])

    report = grade(vintages, actuals)
    totals = report["totals"]

    assert totals["graded_rows"] == 1 and totals["pending_rows"] == 0
    assert totals["mae"] == 2.0
    assert totals["coverage_80"] == 100.0
    # pinball at (0.1, 0.5, 0.9) for actual 12 against (5, 10, 15) is
    # (0.7, 1.0, 0.3), and WIS doubles the mean
    assert np.isclose(totals["wis"], round(2 * (0.7 + 1.0 + 0.3) / 3, 2))
    # persistence carries the origin actual (date minus k = 07-11) forward
    assert totals["persistence_mae"] == 4.0


def test_grade_counts_unpublished_dates_as_pending():
    vintages = _vintage_frame([
        {"generated_at": "2026-07-11T09:30:00+00:00", "model_version": "abc123",
         "route": "A-B", "date": "2026-07-12", "k": 1,
         "q50": 10.0, "lo": 5.0, "hi": 15.0},
        {"generated_at": "2026-07-11T09:30:00+00:00", "model_version": "abc123",
         "route": "A-B", "date": "2026-08-30", "k": 50,
         "q50": 10.0, "lo": 0.0, "hi": 20.0},
    ])
    actuals = _actuals([
        {"route": "A-B", "date": "2026-07-11", "avg_arr_delay": 8.0},
        {"route": "A-B", "date": "2026-07-12", "avg_arr_delay": 12.0},
    ])

    report = grade(vintages, actuals)

    assert report["totals"]["logged_rows"] == 2
    assert report["totals"]["graded_rows"] == 1
    assert report["totals"]["pending_rows"] == 1
    assert [d["k"] for d in report["by_depth"]] == [1]


def test_grade_with_nothing_gradeable_keeps_structure():
    vintages = _vintage_frame([{
        "generated_at": "2026-07-11T09:30:00+00:00", "model_version": "abc123",
        "route": "A-B", "date": "2026-08-30", "k": 50,
        "q50": 10.0, "lo": 0.0, "hi": 20.0,
    }])
    actuals = _actuals([
        {"route": "A-B", "date": "2026-05-31", "avg_arr_delay": 8.0},
    ])

    report = grade(vintages, actuals)

    assert report["totals"]["graded_rows"] == 0
    assert report["totals"]["pending_rows"] == 1
    assert report["totals"]["vintage_days"] == 1
    assert report["vintages"] == [] and report["by_depth"] == []
    assert "mae" not in report["totals"]


def test_grade_empty_log():
    report = grade(pd.DataFrame(), _actuals([
        {"route": "A-B", "date": "2026-05-31", "avg_arr_delay": 8.0},
    ]))
    assert report["totals"]["logged_rows"] == 0
    assert report["vintages"] == []
