import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

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
    "num_leaves": 63,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
}


def load_tuned_params():
    """Loads Optuna best params if available, otherwise returns defaults."""
    params_path = CONFIGS_DIR / "best_params_lightgbm.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"Using Optuna-tuned params from {params_path.name}")
        return tuned
    print("No tuned params found, using defaults")
    return DEFAULT_PARAMS.copy()


def main():
    parser = argparse.ArgumentParser(description="Train the LightGBM point model")
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
    # the dev-test window stops before the locked final holdout, which is
    # evaluated exactly once at release behind --final-test
    devtest_df = df[
        (df["date"] >= TEST_START) & (df["date"] < FINAL_TEST_START)
    ].dropna(subset=available_features + [target_col])

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_devtest = devtest_df[available_features].values
    y_devtest = devtest_df[target_col].values

    print(f"train={len(X_train):,}  val={len(X_val):,}  devtest={len(X_devtest):,}")

    lgb_params = load_tuned_params()
    lgb_params.update({
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=20),
        ],
    )

    print(f"Best iteration: {model.best_iteration_}")

    val_metrics = calculate_delay_metrics(y_val, model.predict(X_val))
    devtest_metrics = calculate_delay_metrics(y_devtest, model.predict(X_devtest))

    print(f"\nVal MAE: {val_metrics['mae']:.2f} min  RMSE: {val_metrics['rmse']:.2f} min")
    print(f"Devtest MAE: {devtest_metrics['mae']:.2f} min")
    print(f"Devtest RMSE: {devtest_metrics['rmse']:.2f} min")
    print(f"Devtest R2: {devtest_metrics['r2']:.3f}")
    print(f"Devtest within 15min: {devtest_metrics['within_15']:.1f}%")

    metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    metrics.update({f"devtest_{k}": v for k, v in devtest_metrics.items()})

    if args.final_test:
        final_df = df[df["date"] >= FINAL_TEST_START].dropna(subset=available_features + [target_col])
        final_metrics = calculate_delay_metrics(
            final_df[target_col].values, model.predict(final_df[available_features].values)
        )
        print(f"\nLOCKED FINAL TEST ({FINAL_TEST_START} onward, one-shot release gate)")
        print(f"Final MAE: {final_metrics['mae']:.2f} min  RMSE: {final_metrics['rmse']:.2f} min")
        print(f"Final within 15min: {final_metrics['within_15']:.1f}%")
        metrics.update({f"finaltest_{k}": v for k, v in final_metrics.items()})

    joblib.dump(model, MODELS_DIR / "lightgbm_delay.pkl")
    joblib.dump(available_features, MODELS_DIR / "lightgbm_features.pkl")

    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.0f}")

    importance_df.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)

    provenance = tracking.provenance_tags(features_df=df, features_path=DATA_DIR / "features.csv")
    with tracking.start_run(run_name="train_lightgbm", tags=provenance):
        tracking.log_params(lgb_params)
        tracking.log_metrics(metrics)
        val_curve = model.evals_result_.get("valid_0", {}).get("l1", [])
        for i in range(0, len(val_curve), 5):
            tracking.log_metrics({"val_mae_curve": val_curve[i]}, step=i)
        tracking.log_artifact(MODELS_DIR / "lightgbm_delay.pkl")


if __name__ == "__main__":
    main()
