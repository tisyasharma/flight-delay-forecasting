import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src import tracking
from src.config import TRAIN_END, VAL_END, TEST_START, FINAL_TEST_START, TABULAR_FEATURES
from src.evaluation.metrics import calculate_delay_metrics

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
}


def load_tuned_params():
    """Loads Optuna best params if available, otherwise returns defaults."""
    params_path = CONFIGS_DIR / "best_params_xgboost.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"Using Optuna-tuned params from {params_path.name}")
        return tuned
    print("No tuned params found, using defaults")
    return DEFAULT_PARAMS.copy()


def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost point model")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="also evaluate the locked 2025 H1 holdout, release gate use only",
    )
    args = parser.parse_args()

    np.random.seed(42)

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"{len(df):,} samples, {len(available_features)} features")

    train_df = df[df["date"] < TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].dropna(subset=available_features + [target_col])

    # the historical devtest and locked-final-test spans predate the serving
    # generation's train_end, so they are in-sample now and skipped; the live
    # public record is this generation's blind test
    holdouts_valid = pd.Timestamp(TEST_START) >= pd.Timestamp(TRAIN_END)
    devtest_df = df[
        (df["date"] >= TEST_START) & (df["date"] < FINAL_TEST_START)
    ].dropna(subset=available_features + [target_col]) if holdouts_valid else df.iloc[:0]

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_devtest = devtest_df[available_features].values
    y_devtest = devtest_df[target_col].values

    print(f"train={len(X_train):,}  val={len(X_val):,}  devtest={len(X_devtest):,}")

    xgb_params = load_tuned_params()
    xgb_params.update({
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    })

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    print(f"Best iteration: {model.best_iteration}")

    val_metrics = calculate_delay_metrics(y_val, model.predict(X_val))
    print(f"\nVal MAE: {val_metrics['mae']:.2f} min  RMSE: {val_metrics['rmse']:.2f} min")
    metrics = {f"val_{k}": v for k, v in val_metrics.items()}

    if holdouts_valid:
        devtest_metrics = calculate_delay_metrics(y_devtest, model.predict(X_devtest))
        print(f"Devtest MAE: {devtest_metrics['mae']:.2f} min")
        print(f"Devtest RMSE: {devtest_metrics['rmse']:.2f} min")
        print(f"Devtest R2: {devtest_metrics['r2']:.3f}")
        print(f"Devtest within 15min: {devtest_metrics['within_15']:.1f}%")
        metrics.update({f"devtest_{k}": v for k, v in devtest_metrics.items()})
    else:
        print("historical devtest/final-test spans are inside the training window, skipping")

    if args.final_test and holdouts_valid:
        final_df = df[df["date"] >= FINAL_TEST_START].dropna(subset=available_features + [target_col])
        final_metrics = calculate_delay_metrics(
            final_df[target_col].values, model.predict(final_df[available_features].values)
        )
        print(f"\nLOCKED FINAL TEST ({FINAL_TEST_START} onward, one-shot release gate)")
        print(f"Final MAE: {final_metrics['mae']:.2f} min  RMSE: {final_metrics['rmse']:.2f} min")
        print(f"Final within 15min: {final_metrics['within_15']:.1f}%")
        metrics.update({f"finaltest_{k}": v for k, v in final_metrics.items()})

    joblib.dump(model, MODELS_DIR / "xgboost_delay.pkl")
    joblib.dump(available_features, MODELS_DIR / "xgboost_features.pkl")

    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    importance_df.to_csv(MODELS_DIR / "xgboost_feature_importance.csv", index=False)

    provenance = tracking.provenance_tags(features_df=df, features_path=DATA_DIR / "features.csv")
    with tracking.start_run(run_name="train_xgboost", tags=provenance):
        tracking.log_params(xgb_params)
        tracking.log_metrics(metrics)
        tracking.log_artifact(MODELS_DIR / "xgboost_delay.pkl")


if __name__ == "__main__":
    main()
