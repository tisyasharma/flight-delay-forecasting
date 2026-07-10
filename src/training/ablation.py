"""
Feature-set ablation over the walk-forward folds: v1 features, plus aviation
weather, plus hub state, and an all-inbound hub variant that uses every BTS
arrival rather than just the modeled routes. The last one is backtest-only
evidence (a live pipeline cannot observe unmodeled inbound traffic), included
to quantify what the serve-consistent hub feature leaves on the table.

Metrics are reported overall and on severe-weather days separately, since
squared error concentrates heavily on the severe tail. All sets share the
same tuned hyperparameters, which mildly favors the full set they were tuned
on; noted in the output.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import tracking
from src.config import WALK_FORWARD_FOLDS
from src.evaluation.conformal import apply_cqr, cqr_offset, split_calibration_window
from src.evaluation.metrics import (
    calculate_delay_metrics,
    calculate_quantile_metrics,
    interval_coverage,
    interval_width,
    sort_quantile_predictions,
)
from src.features.registry import FEATURE_GROUPS, TABULAR_FEATURES
from src.training.fold_features import RAW_PATH, build_fold_features, load_raw

PROJECT_ROOT = Path(__file__).parent.parent.parent
HUB_INBOUND_PATH = PROJECT_ROOT / "data" / "processed" / "hub_inbound_daily.csv"
PARAMS_PATH = PROJECT_ROOT / "configs" / "best_params_lightgbm.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ablation.json"

TARGET = "avg_arr_delay"
QUANTILE_ALPHAS = (0.1, 0.5, 0.9)

# worst-airport severity of rain-or-worse, the same threshold the codebase
# uses for its is_adverse flag
SEVERE_THRESHOLD = 3

V1_FEATURES = [
    f for f in TABULAR_FEATURES
    if f not in set(FEATURE_GROUPS["aviation_weather"]) | set(FEATURE_GROUPS["hub_state"])
]

ALLINBOUND_HUB_COLS = ["hub_allinbound_lag_1", "hub_allinbound_roll_7"]


def load_params():
    """Loads tuned LightGBM params and overlays the fixed training settings."""
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    params.update({"max_depth": -1, "random_state": 42, "n_jobs": -1, "verbose": -1})
    return params


def add_allinbound_hub(df):
    """
    Joins hub state derived from ALL inbound BTS arrivals per destination
    airport, lagged one day and rolled over the trailing week.
    """
    hub = pd.read_csv(HUB_INBOUND_PATH, parse_dates=["date"])
    hub = hub.sort_values(["airport", "date"])

    grouped = hub.groupby("airport")["inbound_avg_arr_delay"]
    hub["hub_allinbound_lag_1"] = grouped.shift(1)
    hub["hub_allinbound_roll_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    out = df.assign(_dest=df["route"].str.split("-").str[1])
    out = out.merge(
        hub[["airport", "date"] + ALLINBOUND_HUB_COLS],
        left_on=["_dest", "date"],
        right_on=["airport", "date"],
        how="left",
    ).drop(columns=["_dest", "airport"])
    return out


def evaluate_fold(df, features, params, fold, per_route=False):
    """
    Fits point and quantile models on one fold, returns overall and severe
    metrics, optionally with per-route interval coverage.
    """
    train_df = df[df["date"] < fold["train_end"]].dropna(subset=features + [TARGET])
    val_df = df[
        (df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])
    ].dropna(subset=features + [TARGET])
    test_df = df[
        (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    ].dropna(subset=features + [TARGET])

    point = lgb.LGBMRegressor(**params)
    point.fit(
        train_df[features].values, train_df[TARGET].values,
        eval_set=[(val_df[features].values, val_df[TARGET].values)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    point_preds = point.predict(test_df[features].values)

    # quantile models early-stop on the first half of the val window only,
    # the later half is the dedicated conformal calibration set
    es_df, cal_df = split_calibration_window(val_df)

    quantile_preds = []
    cal_preds = []
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        model.fit(
            train_df[features].values, train_df[TARGET].values,
            eval_set=[(es_df[features].values, es_df[TARGET].values)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        quantile_preds.append(model.predict(test_df[features].values))
        cal_preds.append(model.predict(cal_df[features].values))
    sorted_preds = sort_quantile_predictions(np.column_stack(quantile_preds))

    sorted_cal = sort_quantile_predictions(np.column_stack(cal_preds))
    offset = cqr_offset(cal_df[TARGET].values, sorted_cal[:, 0], sorted_cal[:, -1], alpha=0.2)
    lo_cqr, hi_cqr = apply_cqr(sorted_preds[:, 0], sorted_preds[:, -1], offset)

    y = test_df[TARGET].values
    severe = (test_df["weather_severity_max"] >= SEVERE_THRESHOLD).values

    overall = calculate_delay_metrics(y, point_preds)
    severe_point = calculate_delay_metrics(y[severe], point_preds[severe])
    quantile = calculate_quantile_metrics(y, sorted_preds, alphas=QUANTILE_ALPHAS)
    severe_quantile = calculate_quantile_metrics(
        y[severe], sorted_preds[severe], alphas=QUANTILE_ALPHAS
    )

    result = {
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "severe_mae": severe_point["mae"],
        "severe_rmse": severe_point["rmse"],
        "severe_n": int(severe.sum()),
        "coverage_80": quantile["coverage_80"],
        "interval_width": quantile["interval_width"],
        "severe_coverage_80": severe_quantile["coverage_80"],
        # conditional coverage after conformal: marginal calibration can hide
        # subgroup under-coverage, so the calibrated interval is scored on the
        # severe-day slice as well
        "coverage_80_cqr": float(interval_coverage(y, lo_cqr, hi_cqr)),
        "interval_width_cqr": float(interval_width(lo_cqr, hi_cqr)),
        "severe_coverage_80_cqr": float(
            interval_coverage(y[severe], lo_cqr[severe], hi_cqr[severe])
        ),
        "cqr_offset": float(offset),
    }

    if per_route:
        routes = test_df["route"].values
        in_interval = (y >= sorted_preds[:, 0]) & (y <= sorted_preds[:, -1])
        in_interval_cqr = (y >= lo_cqr) & (y <= hi_cqr)
        result["per_route_coverage_80"] = {
            route: round(float(in_interval[routes == route].mean() * 100), 2)
            for route in np.unique(routes)
        }
        result["per_route_coverage_80_cqr"] = {
            route: round(float(in_interval_cqr[routes == route].mean() * 100), 2)
            for route in np.unique(routes)
        }

    return result


def summarize(fold_metrics):
    """Means each metric across folds, skipping Nones."""
    summary = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics if m.get(key) is not None]
        summary[key] = round(float(np.mean(values)), 2) if values else None
    return summary


def main():
    raw_df = load_raw()
    params = load_params()

    feature_sets = {
        "v1": V1_FEATURES,
        "v1_plus_aviation": V1_FEATURES + FEATURE_GROUPS["aviation_weather"],
        "full": list(TABULAR_FEATURES),
        "full_allinbound_hub": [
            f for f in TABULAR_FEATURES if f not in FEATURE_GROUPS["hub_state"]
        ] + ALLINBOUND_HUB_COLS,
    }

    results = {
        "target": TARGET,
        "severe_definition": f"weather_severity_max >= {SEVERE_THRESHOLD} on test rows",
        "params_note": (
            "all sets share the params tuned on the full feature set; "
            "full_allinbound_hub is backtest-only and not servable live"
        ),
        "methodology": "feature statistics rebuilt per fold at each fold's train_end",
        "n_folds": len(WALK_FORWARD_FOLDS),
        "sets": {},
    }

    # folds are the outer loop so each fold frame is built once and shared
    # by every feature set, the all-inbound hub join is pure shift/roll and
    # carries no train-window statistics
    set_fold_metrics = {name: [] for name in feature_sets}
    for fold in WALK_FORWARD_FOLDS:
        print(f"\nfold to {fold['test_end']}: building fold features...")
        fold_df = add_allinbound_hub(build_fold_features(raw_df, fold["train_end"]))

        for name, features in feature_sets.items():
            # per-route coverage only for the served feature set, it feeds the
            # decision on whether interval widening needs a per-route term
            metrics = evaluate_fold(fold_df, features, params, fold, per_route=(name == "full"))
            set_fold_metrics[name].append(metrics)
            print(f"  {name}: mae {metrics['mae']:.2f}, "
                  f"severe mae {metrics['severe_mae']:.2f}")

    for name, features in feature_sets.items():
        fold_metrics = set_fold_metrics[name]
        per_route_keys = ["per_route_coverage_80", "per_route_coverage_80_cqr"]
        per_route_folds = {
            key: [m.pop(key, None) for m in fold_metrics] for key in per_route_keys
        }
        entry = {
            "n_features": len(features),
            "folds": fold_metrics,
            "mean": summarize(fold_metrics),
        }

        for key, folds in per_route_folds.items():
            folds = [d for d in folds if d]
            if not folds:
                continue
            routes = sorted(set().union(*folds))
            per_route = {}
            for route in routes:
                values = [d[route] for d in folds if route in d]
                per_route[route] = {
                    "mean": round(float(np.mean(values)), 2),
                    "min": round(float(np.min(values)), 2),
                    "max": round(float(np.max(values)), 2),
                    "n_folds": len(values),
                }
            entry[key] = per_route

            worst = sorted(per_route.items(), key=lambda kv: kv[1]["mean"])[:5]
            print(f"\nworst-covered routes ({name} set, mean {key} across folds):")
            for route, stats in worst:
                print(f"  {route}: {stats['mean']:.1f} (min {stats['min']:.1f})")

        results["sets"][name] = entry

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    # tracking runs after the results file is safe on disk, no-op without mlflow
    provenance = tracking.provenance_tags(features_df=raw_df, features_path=RAW_PATH)
    with tracking.start_run(run_name="ablation", tags={"stage": "ablation", **provenance}):
        tracking.log_params({"n_folds": len(WALK_FORWARD_FOLDS), "sets": ",".join(feature_sets)})
        for name, entry in results["sets"].items():
            with tracking.start_run(run_name=f"set_{name}", nested=True):
                tracking.log_params({"n_features": entry["n_features"]})
                tracking.log_metrics(entry["mean"])
        tracking.log_artifact(OUTPUT_PATH)

    header = ["set", "features", "mae", "severe_mae", "rmse", "severe_rmse",
              "coverage_80", "severe_coverage_80", "interval_width"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name, entry in results["sets"].items():
        m = entry["mean"]
        row = [name, str(entry["n_features"])] + [
            str(m[k]) for k in ["mae", "severe_mae", "rmse", "severe_rmse",
                                "coverage_80", "severe_coverage_80", "interval_width"]
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
