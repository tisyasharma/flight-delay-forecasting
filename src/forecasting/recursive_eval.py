"""
Backtests the recursive engine under serving conditions: from a forecast
origin, every delay lag beyond the origin comes from the model's own rolled
predictions, exactly as it would live where BTS actuals arrive one to two
months late. Walk-forward MAE answers "how good are one-step predictions
with real lags"; this answers the harder, honest question of how accuracy
and coverage decay with recursion depth.

Per fold, quantile models are trained with the same discipline as the
walk-forward harness (early stopping on the first half of the val window),
then rolled K days from weekly origins across the fold's test window. The
per-horizon conformal offsets are fit on folds 1-3 and validated on fold 4's
untouched window, mirroring the project's calibration-split discipline.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import tracking
from src.config import TABULAR_FEATURES, WALK_FORWARD_FOLDS
from src.evaluation.conformal import cqr_offset, split_calibration_window
from src.forecasting.recursive import MIN_WINDOW, RecursiveForecaster
from src.training.fold_features import RAW_PATH, build_fold_features, load_raw
from src.training.walk_forward import (
    OUTPUTS_DIR,
    QUANTILE_ALPHAS,
    load_params,
)

TARGET = "avg_arr_delay"

# deep enough to cover live serving across the whole monthly refresh cycle:
# right after new BTS actuals land the depth is the publication lag plus the
# 7-day horizon (~45), but the last actual then stays fixed while the clock
# runs, so the day before the next month publishes it reaches lag + ~31 + 7
# (~75-83). every depth the daily job can serve must be validated, not
# extrapolated, and the job refuses depths beyond the deepest fitted offset
MAX_DEPTH = 90
ORIGIN_SPACING_DAYS = 7
VALIDATION_FOLD = 3  # index into WALK_FORWARD_FOLDS, the last fold

OUTPUT_PATH = OUTPUTS_DIR / "recursive_eval.json"


def train_fold_quantile_models(fold_df, features, fold):
    """Fits the three quantile models with the walk-forward discipline."""
    params = load_params("lightgbm")
    base_params = {
        "n_estimators": params.get("n_estimators", 500),
        "num_leaves": params.get("num_leaves", 63),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "subsample_freq": params.get("subsample_freq", 0),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 1e-6),
        "reg_lambda": params.get("reg_lambda", 1e-6),
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    train_df = fold_df[fold_df["date"] < fold["train_end"]].dropna(subset=features + [TARGET])
    val_df = fold_df[
        (fold_df["date"] >= fold["train_end"]) & (fold_df["date"] < fold["val_end"])
    ].dropna(subset=features + [TARGET])
    es_df, _ = split_calibration_window(val_df)

    models = {}
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
        model.fit(
            train_df[features].values, train_df[TARGET].values,
            eval_set=[(es_df[features].values, es_df[TARGET].values)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        models[alpha] = model
    return models


def roll_fold(fold_idx, fold, fold_df, features, state):
    """
    Rolls forecasts from weekly origins across one fold's test window and
    returns a record frame: fold, origin, depth k, actual, raw quantiles,
    and the frozen-persistence baseline (the last actual carried forward).
    """
    models = train_fold_quantile_models(fold_df, features, fold)
    forecaster = RecursiveForecaster(models, state, features=features)

    actuals = fold_df.set_index(["route", "date"])[TARGET]
    test_end = pd.Timestamp(fold["test_end"])

    records = []
    origins = pd.date_range(
        pd.Timestamp(fold["test_start"]) - pd.Timedelta(days=1),
        test_end - pd.Timedelta(days=1),
        freq=f"{ORIGIN_SPACING_DAYS}D",
    )
    for origin in origins:
        n_days = min(MAX_DEPTH, (test_end - origin).days)
        out = forecaster.forecast(fold_df, origin, n_days=n_days)

        keyed = pd.MultiIndex.from_arrays([out["route"], out["date"]])
        out["actual"] = actuals.reindex(keyed).values
        persist = actuals.reindex(
            pd.MultiIndex.from_arrays([out["route"], np.repeat(origin, len(out))])
        ).values
        out["persistence"] = persist
        out["fold"] = fold_idx
        out["origin"] = origin
        records.append(out.dropna(subset=["actual"]))

    print(f"  fold {fold_idx + 1}: {len(origins)} origins rolled")
    return pd.concat(records, ignore_index=True)


def fit_horizon_offsets(records):
    """Per-depth conformal offsets from the fitting folds' rolled intervals."""
    offsets = {}
    for k, group in records.groupby("k"):
        offsets[int(k)] = cqr_offset(
            group["actual"].values, group["q10"].values, group["q90"].values, alpha=0.2
        )
    return offsets


def by_depth(records, offsets=None):
    """MAE, coverage, and width per recursion depth, optionally widened."""
    table = {}
    for k, g in records.groupby("k"):
        lo, hi = g["q10"].values, g["q90"].values
        if offsets is not None:
            off = offsets[min(int(k), max(offsets))]
            lo, hi = lo - off, hi + off
        y = g["actual"].values
        table[int(k)] = {
            "n": int(len(g)),
            "mae": round(float(np.abs(y - g["q50"].values).mean()), 2),
            "persistence_mae": round(float(np.abs(y - g["persistence"].values).mean()), 2),
            "coverage_80": round(float(((y >= lo) & (y <= hi)).mean() * 100), 2),
            "interval_width": round(float((hi - lo).mean()), 2),
        }
    return table


def main():
    np.random.seed(42)
    raw_df = load_raw()

    all_records = []
    for fold_idx, fold in enumerate(WALK_FORWARD_FOLDS):
        print(f"fold {fold_idx + 1}: building features at {fold['train_end']}...")
        fold_df = build_fold_features(raw_df, fold["train_end"])
        features = [c for c in TABULAR_FEATURES if c in fold_df.columns]

        # the engine only reads the route set from the state, build a minimal
        # one so route mismatches between frame and models still fail loudly
        state = {"route_codes": {r: i for i, r in enumerate(sorted(fold_df["route"].unique()))}}

        all_records.append(roll_fold(fold_idx, fold, fold_df, features, state))

    records = pd.concat(all_records, ignore_index=True)

    fit_records = records[records["fold"] != VALIDATION_FOLD]
    val_records = records[records["fold"] == VALIDATION_FOLD]
    offsets = fit_horizon_offsets(fit_records)

    results = {
        "protocol": {
            "max_depth": MAX_DEPTH,
            "origin_spacing_days": ORIGIN_SPACING_DAYS,
            "trailing_window_days": MIN_WINDOW,
            "recursion_feed": "q50",
            "models": "per-fold LightGBM quantile triple, walk-forward discipline",
            "offsets_fit_on": "folds 1-3",
            "validated_on": f"fold {VALIDATION_FOLD + 1} (untouched by the offset fit)",
            "weather_note": (
                "backtest uses observed weather across the horizon; a live run "
                "would substitute forecast weather and be somewhat worse"
            ),
        },
        "all_folds_raw": by_depth(records),
        "offsets_by_k": {k: round(v, 3) for k, v in offsets.items()},
        "validation_fold4_raw": by_depth(val_records),
        "validation_fold4_calibrated": by_depth(val_records, offsets=offsets),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    cal = results["validation_fold4_calibrated"]
    ks = sorted(cal)
    print("\ndepth k=1 vs deep-k (fold-4 validation, calibrated):")
    for k in (ks[0], 7, 14, 30, ks[-1]):
        if k in cal:
            row = cal[k]
            print(f"  k={k:>2}: mae {row['mae']:.2f} "
                  f"(persistence {row['persistence_mae']:.2f}), "
                  f"coverage {row['coverage_80']:.1f}%, n={row['n']}")

    provenance = tracking.provenance_tags(features_df=raw_df, features_path=RAW_PATH)
    with tracking.start_run(run_name="recursive_eval", tags={"stage": "recursive_eval",
                                                             **provenance}):
        tracking.log_params({
            "max_depth": MAX_DEPTH,
            "origin_spacing_days": ORIGIN_SPACING_DAYS,
            "n_records": len(records),
        })
        first, last = min(cal), max(cal)
        tracking.log_metrics({
            "mae_k1": cal[first]["mae"],
            "mae_deepest": cal[last]["mae"],
            "coverage_k1_calibrated": cal[first]["coverage_80"],
            "coverage_deepest_calibrated": cal[last]["coverage_80"],
        })
        tracking.log_artifact(OUTPUT_PATH)


if __name__ == "__main__":
    main()
