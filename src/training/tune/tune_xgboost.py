import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_END, VAL_END, TABULAR_FEATURES
from src.evaluation.metrics import calculate_delay_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    """Loads features.csv and splits into train/val arrays."""
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    train_df = df[df["date"] < TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].dropna(
        subset=available_features + [target_col]
    )

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values

    return X_train, y_train, X_val, y_val


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: train XGBoost with sampled params, return val MAE."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    metrics = calculate_delay_metrics(y_val, y_pred)

    return metrics["mae"]


def main():
    """Runs Optuna study for XGBoost and saves best params."""
    np.random.seed(42)

    X_train, y_train, X_val, y_val = load_splits()
    print(f"train={len(X_train):,}  val={len(X_val):,}")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"\nBest val MAE: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    output_path = MODELS_DIR / "best_params_xgboost.json"
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
