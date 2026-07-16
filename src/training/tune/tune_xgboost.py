import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from src import tracking
from src.evaluation.metrics import calculate_delay_metrics
from src.training.common import load_splits
from src.training.fold_features import RAW_PATH

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


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

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"\nBest val MAE: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    output_path = CONFIGS_DIR / "best_params_xgboost.json"
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saved to {output_path}")

    provenance = tracking.provenance_tags(features_path=RAW_PATH)
    with tracking.start_run(run_name="tune_xgboost", tags=provenance):
        tracking.log_params({**study.best_params, "sampler_seed": 42})
        tracking.log_metrics({"best_val_mae": study.best_value, "n_trials": len(study.trials)})
        tracking.log_artifact(output_path)


if __name__ == "__main__":
    main()
