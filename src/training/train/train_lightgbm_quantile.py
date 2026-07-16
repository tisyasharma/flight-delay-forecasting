import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import tracking
from src.config import FINAL_TEST_START, TABULAR_FEATURES, TEST_START, TRAIN_END, VAL_END
from src.evaluation.conformal import apply_cqr, cqr_offset, split_calibration_window
from src.evaluation.metrics import (
    calculate_quantile_metrics,
    interval_coverage,
    interval_width,
    sort_quantile_predictions,
)
from src.training.common import load_tuned_params
from src.training.train.train_lightgbm import DEFAULT_PARAMS

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)


def export_point_model():
    """
    Saves the published point model as a native text booster so the release
    ships one artifact set that can reproduce the point-model numbers. It is
    not on the serving path, the daily forecast loads only the quantile
    boosters. The pickle stays the source of truth, retraining here would
    silently detach the artifact from its published metrics.
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

    # the historical devtest and locked-final-test spans predate the serving
    # generation's train_end, so they are in-sample now and skipped. the live
    # public record is this generation's blind test
    holdouts_valid = pd.Timestamp(TEST_START) >= pd.Timestamp(TRAIN_END)
    devtest_df = df[
        (df["date"] >= TEST_START) & (df["date"] < FINAL_TEST_START)
    ].dropna(subset=available_features + [target_col]) if holdouts_valid else df.iloc[:0]
    final_df = df[df["date"] >= FINAL_TEST_START].dropna(
        subset=available_features + [target_col]) if holdouts_valid else df.iloc[:0]

    # the early half of the val window drives early stopping, the later half
    # is a dedicated calibration set for the conformal offset
    es_df, cal_df = split_calibration_window(val_df)

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_es = es_df[available_features].values
    y_es = es_df[target_col].values
    X_cal = cal_df[available_features].values
    y_cal = cal_df[target_col].values
    X_devtest = devtest_df[available_features].values
    y_devtest = devtest_df[target_col].values

    print(f"train={len(X_train):,}  earlystop={len(X_es):,}  "
          f"calibration={len(X_cal):,}  devtest={len(X_devtest):,}")
    if not holdouts_valid:
        print("historical devtest/final-test spans are inside the training window, skipping")

    base_params = load_tuned_params("lightgbm", DEFAULT_PARAMS)
    base_params.update({
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    quantile_preds = []
    cal_preds = []
    final_preds = []
    val_curves = {}
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_es, y_es)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        name = f"lightgbm_q{int(round(alpha * 100))}"
        model.booster_.save_model(str(MODELS_DIR / f"{name}.txt"))
        print(f"{name}: best iteration {model.best_iteration_}")

        val_curves[name] = model.evals_result_.get("valid_0", {}).get("quantile", [])
        cal_preds.append(model.predict(X_cal))
        if holdouts_valid:
            quantile_preds.append(model.predict(X_devtest))
        if args.final_test and holdouts_valid:
            final_preds.append(model.predict(final_df[available_features].values))

    # the conformal offset ships with the boosters so any consumer widens the
    # raw interval the same way the published numbers do
    sorted_cal = sort_quantile_predictions(np.column_stack(cal_preds))
    offset = cqr_offset(y_cal, sorted_cal[:, 0], sorted_cal[:, -1], alpha=0.2)

    devtest_metrics = {}
    if holdouts_valid:
        sorted_preds = sort_quantile_predictions(np.column_stack(quantile_preds))
        devtest_metrics = calculate_quantile_metrics(y_devtest, sorted_preds,
                                                     alphas=QUANTILE_ALPHAS)
        lo_dev, hi_dev = apply_cqr(sorted_preds[:, 0], sorted_preds[:, -1], offset)
        devtest_metrics["coverage_80_cqr"] = float(interval_coverage(y_devtest, lo_dev, hi_dev))
        devtest_metrics["interval_width_cqr"] = float(interval_width(lo_dev, hi_dev))

    conformal_path = MODELS_DIR / "conformal.json"
    with open(conformal_path, "w") as f:
        json.dump({
            "offset": float(offset),
            "alpha": 0.2,
            # the daily job matches this against the repo feature state so a
            # release and a checkout from different generations refuse to serve
            "train_end_date": TRAIN_END,
            "calibration_window": [
                str(cal_df["date"].min().date()), str(cal_df["date"].max().date()),
            ],
            "n_calibration": int(len(y_cal)),
        }, f, indent=2)
    print(f"Saved conformal offset {offset:.3f} to {conformal_path.name}")

    if holdouts_valid:
        print(f"\nDevtest coverage_80: {devtest_metrics['coverage_80']:.1f}% (nominal 80%)")
        print(f"Devtest coverage_80_cqr: {devtest_metrics['coverage_80_cqr']:.1f}%")
        print(f"Devtest interval width: {devtest_metrics['interval_width']:.1f} min "
              f"(cqr {devtest_metrics['interval_width_cqr']:.1f})")
        for alpha in QUANTILE_ALPHAS:
            key = f"pinball_{int(round(alpha * 100))}"
            print(f"Devtest {key}: {devtest_metrics[key]:.3f}")

    metrics = {f"devtest_{k}": v for k, v in devtest_metrics.items()}
    metrics["conformal_offset"] = float(offset)

    if args.final_test and not holdouts_valid:
        print("final test span already consumed by the previous generation, skipping")
    if args.final_test and holdouts_valid:
        final_sorted = sort_quantile_predictions(np.column_stack(final_preds))
        y_final = final_df[target_col].values
        final_metrics = calculate_quantile_metrics(y_final, final_sorted, alphas=QUANTILE_ALPHAS)
        lo_fin, hi_fin = apply_cqr(final_sorted[:, 0], final_sorted[:, -1], offset)
        final_metrics["coverage_80_cqr"] = float(interval_coverage(y_final, lo_fin, hi_fin))
        final_metrics["interval_width_cqr"] = float(interval_width(lo_fin, hi_fin))
        print(f"\nLOCKED FINAL TEST ({FINAL_TEST_START} onward, one-shot release gate)")
        print(f"Final coverage_80: {final_metrics['coverage_80']:.1f}% "
              f"(cqr {final_metrics['coverage_80_cqr']:.1f}, nominal 80%)")
        print(f"Final interval width: {final_metrics['interval_width']:.1f} min "
              f"(cqr {final_metrics['interval_width_cqr']:.1f})")
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
        tracking.log_artifact(conformal_path)


if __name__ == "__main__":
    main()
