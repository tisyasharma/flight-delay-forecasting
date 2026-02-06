import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_END, VAL_END, TEST_START, TABULAR_FEATURES
from src.evaluation.metrics import calculate_delay_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    params_path = MODELS_DIR / "best_params_xgboost.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"Using Optuna-tuned params from {params_path.name}")
        return tuned
    print("No tuned params found, using defaults")
    return DEFAULT_PARAMS.copy()


def main():
    np.random.seed(42)

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"{len(df):,} samples, {len(available_features)} features")

    train_df = df[df["date"] < TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].dropna(subset=available_features + [target_col])
    test_df = df[df["date"] >= TEST_START].dropna(subset=available_features + [target_col])

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_test = test_df[available_features].values
    y_test = test_df[target_col].values

    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

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

    y_pred = model.predict(X_test)
    metrics = calculate_delay_metrics(y_test, y_pred)

    print(f"\nTest MAE: {metrics['mae']:.2f} min")
    print(f"Test RMSE: {metrics['rmse']:.2f} min")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"Within 15min: {metrics['within_15']:.1f}%")

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


if __name__ == "__main__":
    main()
