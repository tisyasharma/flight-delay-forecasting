"""
Train XGBoost model for delay prediction.
Saves model, feature list, and feature importance to trained_models/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_END, VAL_END, TABULAR_FEATURES
from src.evaluation.metrics import calculate_delay_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Training XGBoost Delay Model...")

    np.random.seed(42)

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    # full 57 features with explicit lags (XGBoost can't learn from sequence position)
    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"Total samples: {len(df):,}")
    print(f"Features: {len(available_features)}")

    train_df = df[df["date"] < TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].dropna(subset=available_features + [target_col])
    test_df = df[df["date"] >= VAL_END].dropna(subset=available_features + [target_col])

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_test = test_df[available_features].values
    y_test = test_df[target_col].values

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50
    }

    print("\nTraining XGBoost with early stopping...")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    print(f"\nBest iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    metrics = calculate_delay_metrics(y_test, y_pred)

    print("\nTest Set Metrics:")
    print(f"  MAE:          {metrics['mae']:.2f} min")
    print(f"  RMSE:         {metrics['rmse']:.2f} min")
    print(f"  R2:           {metrics['r2']:.3f}")
    print(f"  Within 15min: {metrics['within_15']:.1f}%")

    # save model and feature list so generate_forecasts.py can load them later
    model_path = MODELS_DIR / "xgboost_delay.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    joblib.dump(available_features, MODELS_DIR / "xgboost_features.pkl")
    print("Feature list saved.")

    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 Feature Importances:")
    for _, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:35s} {row['importance']:.4f}")

    importance_path = MODELS_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")


if __name__ == "__main__":
    main()
