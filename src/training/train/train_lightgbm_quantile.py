import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import tracking
from src.config import TRAIN_END, VAL_END, TEST_START, FINAL_TEST_START, TABULAR_FEATURES
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
    parser = argparse.ArgumentParser(description="Train the LightGBM quantile models")
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
    val_df = df[
        (df["date"] >= TRAIN_END) & (df["date"] < VAL_END)
    ].dropna(subset=available_features + [target_col])
    # the dev-test window stops before the locked final holdout, which is
    # evaluated exactly once at release behind --final-test
    devtest_df = df[
        (df["date"] >= TEST_START) & (df["date"] < FINAL_TEST_START)
    ].dropna(subset=available_features + [target_col])
    final_df = df[df["date"] >= FINAL_TEST_START].dropna(subset=available_features + [target_col])

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values
    X_devtest = devtest_df[available_features].values
    y_devtest = devtest_df[target_col].values

    print(f"train={len(X_train):,}  val={len(X_val):,}  devtest={len(X_devtest):,}")

    base_params = load_tuned_params()
    base_params.update({
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    quantile_preds = []
    final_preds = []
    val_curves = {}
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

        val_curves[name] = model.evals_result_.get("valid_0", {}).get("quantile", [])
        quantile_preds.append(model.predict(X_devtest))
        if args.final_test:
            final_preds.append(model.predict(final_df[available_features].values))

    sorted_preds = sort_quantile_predictions(np.column_stack(quantile_preds))
    devtest_metrics = calculate_quantile_metrics(y_devtest, sorted_preds, alphas=QUANTILE_ALPHAS)

    print(f"\nDevtest coverage_80: {devtest_metrics['coverage_80']:.1f}% (nominal 80%)")
    print(f"Devtest interval width: {devtest_metrics['interval_width']:.1f} min")
    for alpha in QUANTILE_ALPHAS:
        key = f"pinball_{int(round(alpha * 100))}"
        print(f"Devtest {key}: {devtest_metrics[key]:.3f}")

    metrics = {f"devtest_{k}": v for k, v in devtest_metrics.items()}

    if args.final_test:
        final_sorted = sort_quantile_predictions(np.column_stack(final_preds))
        final_metrics = calculate_quantile_metrics(
            final_df[target_col].values, final_sorted, alphas=QUANTILE_ALPHAS
        )
        print(f"\nLOCKED FINAL TEST ({FINAL_TEST_START} onward, one-shot release gate)")
        print(f"Final coverage_80: {final_metrics['coverage_80']:.1f}% (nominal 80%)")
        print(f"Final interval width: {final_metrics['interval_width']:.1f} min")
        metrics.update({f"finaltest_{k}": v for k, v in final_metrics.items()})

    export_point_model()

    provenance = tracking.provenance_tags(features_df=df, features_path=DATA_DIR / "features.csv")
    with tracking.start_run(run_name="train_lightgbm_quantile", tags=provenance):
        tracking.log_params({**base_params, "alphas": str(QUANTILE_ALPHAS)})
        tracking.log_metrics(metrics)
        for name, curve in val_curves.items():
            for i in range(0, len(curve), 5):
                tracking.log_metrics({f"val_pinball_curve_{name.split('_')[-1]}": curve[i]}, step=i)
        for alpha in QUANTILE_ALPHAS:
            tracking.log_artifact(MODELS_DIR / f"lightgbm_q{int(round(alpha * 100))}.txt")
        tracking.log_artifact(MODELS_DIR / "lightgbm_point.txt")


if __name__ == "__main__":
    main()
