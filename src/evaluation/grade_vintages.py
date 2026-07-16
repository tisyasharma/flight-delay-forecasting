"""
Grades the append-only prediction log against the latest published actuals,
the same as-of discipline the log was designed for: every vintage is scored
exactly as issued, against ground truth that arrived months later. A row is
gradeable once BTS has published its date. Until the first forecasted month
lands, the output carries the pending counts the monitoring page renders as
designed empty states.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import interval_coverage, weighted_interval_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "live" / "predictions"
DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "daily_route_demand.csv"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "data" / "monitoring.json"

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)


def load_vintages(log_dir=LOG_DIR):
    """Every logged forecast row across all monthly log files, as one frame."""
    rows = []
    for path in sorted(glob.glob(str(Path(log_dir) / "*.ndjson"))):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["vintage"] = df["generated_at"].str[:10]
    return df


def grade(vintages, actuals):
    """
    Joins logged forecasts against actuals on (route, date) and scores what
    has become gradeable. The persistence comparator carries each vintage's
    last known actual forward, derived per row as the actual at date minus k
    days, the same baseline the recursion backtest reports.
    """
    if vintages.empty:
        return {"vintages": [], "by_depth": [], "totals": empty_totals(0)}

    actual_by_key = actuals.set_index(["route", "date"])["avg_arr_delay"]

    keys = pd.MultiIndex.from_frame(vintages[["route", "date"]])
    vintages = vintages.assign(actual=actual_by_key.reindex(keys).values)

    origin_dates = vintages["date"] - pd.to_timedelta(vintages["k"], unit="D")
    origin_keys = pd.MultiIndex.from_arrays([vintages["route"], origin_dates])
    vintages = vintages.assign(persistence=actual_by_key.reindex(origin_keys).values)

    graded = vintages.dropna(subset=["actual"])
    pending = int(len(vintages) - len(graded))

    per_vintage = [
        summarize(g, {"vintage": vintage, "model_version": g["model_version"].iloc[0]})
        for vintage, g in graded.groupby("vintage")
    ]
    per_depth = [
        summarize(g, {"k": int(k)}) for k, g in graded.groupby("k")
    ]

    totals = empty_totals(pending)
    totals["logged_rows"] = int(len(vintages))
    totals["graded_rows"] = int(len(graded))
    totals["first_vintage"] = str(vintages["vintage"].min())
    totals["last_vintage"] = str(vintages["vintage"].max())
    totals["vintage_days"] = int(vintages["vintage"].nunique())
    if len(graded):
        totals.update(scores(graded))
    return {"vintages": per_vintage, "by_depth": per_depth, "totals": totals}


def empty_totals(pending):
    return {"logged_rows": 0, "graded_rows": 0, "pending_rows": pending,
            "first_vintage": None, "last_vintage": None, "vintage_days": 0}


def scores(g):
    """MAE, persistence MAE, empirical coverage, and WIS for one grade slice."""
    preds = np.column_stack([g["lo"], g["q50"], g["hi"]])
    out = {
        "n": int(len(g)),
        "mae": round(float(np.abs(g["q50"] - g["actual"]).mean()), 2),
        "coverage_80": round(float(interval_coverage(
            g["actual"].values, g["lo"].values, g["hi"].values)), 1),
        "wis": round(float(weighted_interval_score(
            g["actual"].values, preds, alphas=QUANTILE_ALPHAS)), 2),
    }
    persist = g.dropna(subset=["persistence"])
    if len(persist):
        out["persistence_mae"] = round(
            float(np.abs(persist["persistence"] - persist["actual"]).mean()), 2)
    return out


def summarize(g, base):
    base.update(scores(g))
    return base


def main():
    parser = argparse.ArgumentParser(description="Grade logged vintages against actuals")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    vintages = load_vintages()
    actuals = pd.read_csv(DEMAND_PATH, usecols=["route", "date", "avg_arr_delay"],
                          parse_dates=["date"])

    report = grade(vintages, actuals)
    report["actuals_through"] = str(actuals["date"].max().date())
    report["graded_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=1)
    print(f"graded {report['totals']['graded_rows']} of {report['totals']['logged_rows']} "
          f"logged rows, {report['totals']['pending_rows']} pending publication")


if __name__ == "__main__":
    raise SystemExit(main())
