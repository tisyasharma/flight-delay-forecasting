from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import TRAIN_END, VAL_END, TEST_START, TABULAR_FEATURES
from src.evaluation.metrics import calculate_quantile_metrics, sort_quantile_predictions
from src.training.train.train_lightgbm import load_tuned_params

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)


def export_point_model():
    """
    Saves the published point model as a native text booster so serving can
    load it without sklearn or joblib. The pickle stays the source of truth,
    retraining here would silently detach the artifact from its published
    metrics.
    """
    pkl_path = MODELS_DIR / "lightgbm_delay.pkl"
    if not pkl_path.exists():
        print("lightgbm_delay.pkl not found, skipping point model export")
        return None

    model = joblib.load(pkl_path)
    out_path = MODELS_DIR / "lightgbm_point.txt"
    model.booster_.save_model(str(out_path))
    print(f"Exported point model to {out_path.name}")
    return out_path


def main():
    np.random.seed(42)

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"{len(df):,} samples, {len(available_features)} features")

    train_df = df[df["date"] < TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[
        (df["date"] >= TRAIN_END) & (df["date"] < VAL_END)
    ].dropna(subset=available_features + [target_col])
    test_df = df[df["date"] >= TEST_START].dropna(subset=available_features + [target_col])

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_test = test_df[available_features].values
    y_test = test_df[target_col].values

    print(f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    base_params = load_tuned_params()
    base_params.update({
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    quantile_preds = []
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        name = f"lightgbm_q{int(round(alpha * 100))}"
        model.booster_.save_model(str(MODELS_DIR / f"{name}.txt"))
        print(f"{name}: best iteration {model.best_iteration_}")

        quantile_preds.append(model.predict(X_test))

    sorted_preds = sort_quantile_predictions(np.column_stack(quantile_preds))
    metrics = calculate_quantile_metrics(y_test, sorted_preds, alphas=QUANTILE_ALPHAS)

    print(f"\nTest coverage_80: {metrics['coverage_80']:.1f}% (nominal 80%)")
    print(f"Test interval width: {metrics['interval_width']:.1f} min")
    for alpha in QUANTILE_ALPHAS:
        key = f"pinball_{int(alpha * 100)}"
        print(f"Test {key}: {metrics[key]:.3f}")

    export_point_model()


if __name__ == "__main__":
    main()
