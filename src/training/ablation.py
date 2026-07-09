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

from src.config import WALK_FORWARD_FOLDS
from src.evaluation.metrics import (
    calculate_delay_metrics,
    calculate_quantile_metrics,
    sort_quantile_predictions,
)
from src.features.registry import FEATURE_GROUPS, TABULAR_FEATURES

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
HUB_INBOUND_PATH = PROJECT_ROOT / "data" / "processed" / "hub_inbound_daily.csv"
PARAMS_PATH = PROJECT_ROOT / "trained_models" / "best_params_lightgbm.json"
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


def evaluate_fold(df, features, params, fold):
    """Fits point and quantile models on one fold, returns overall and severe metrics."""
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

    quantile_preds = []
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        model.fit(
            train_df[features].values, train_df[TARGET].values,
            eval_set=[(val_df[features].values, val_df[TARGET].values)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        quantile_preds.append(model.predict(test_df[features].values))
    sorted_preds = sort_quantile_predictions(np.column_stack(quantile_preds))

    y = test_df[TARGET].values
    severe = (test_df["weather_severity_max"] >= SEVERE_THRESHOLD).values

    overall = calculate_delay_metrics(y, point_preds)
    severe_point = calculate_delay_metrics(y[severe], point_preds[severe])
    quantile = calculate_quantile_metrics(y, sorted_preds, alphas=QUANTILE_ALPHAS)
    severe_quantile = calculate_quantile_metrics(
        y[severe], sorted_preds[severe], alphas=QUANTILE_ALPHAS
    )

    return {
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "severe_mae": severe_point["mae"],
        "severe_rmse": severe_point["rmse"],
        "severe_n": int(severe.sum()),
        "coverage_80": quantile["coverage_80"],
        "interval_width": quantile["interval_width"],
        "severe_coverage_80": severe_quantile["coverage_80"],
    }


def summarize(fold_metrics):
    """Means each metric across folds, skipping Nones."""
    summary = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics if m.get(key) is not None]
        summary[key] = round(float(np.mean(values)), 2) if values else None
    return summary


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = add_allinbound_hub(df)

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
        "n_folds": len(WALK_FORWARD_FOLDS),
        "sets": {},
    }

    for name, features in feature_sets.items():
        print(f"\n{name} ({len(features)} features)")
        fold_metrics = []
        for fold in WALK_FORWARD_FOLDS:
            metrics = evaluate_fold(df, features, params, fold)
            fold_metrics.append(metrics)
            print(f"  fold to {fold['test_end']}: mae {metrics['mae']:.2f}, "
                  f"severe mae {metrics['severe_mae']:.2f}")

        results["sets"][name] = {
            "n_features": len(features),
            "folds": fold_metrics,
            "mean": summarize(fold_metrics),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

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
