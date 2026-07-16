import argparse
import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from src import tracking
from src.config import (
    DATA_START,
    HPO_TRAIN_END,
    HPO_VAL_END,
    SEQUENCE_LENGTH,
    SEQUENCE_MODEL_FEATURES,
    TABULAR_FEATURES,
    WALK_FORWARD_FOLDS,
)
from src.evaluation.conformal import apply_cqr, cqr_offset, split_calibration_window
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_delay_metrics,
    calculate_quantile_metrics,
    interval_coverage,
    interval_width,
    sort_quantile_predictions,
)
from src.training.fold_features import RAW_PATH, build_fold_features, load_raw

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)

PROJECT_ROOT = Path(__file__).parent.parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_params(model_name):
    """Loads tuned params JSON for a model, returns empty dict if missing."""
    params_path = CONFIGS_DIR / f"best_params_{model_name}.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return {}


def train_eval_xgboost(df, features, target_col, fold):
    """Trains XGBoost on one fold and returns test metrics."""
    params = load_params("xgboost")

    train_df = df[df["date"] < fold["train_end"]].dropna(subset=features + [target_col])
    val_df = df[
        (df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])
    ].dropna(subset=features + [target_col])
    test_df = df[
        (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    ].dropna(subset=features + [target_col])

    if len(test_df) == 0:
        return None

    xgb_params = {
        "n_estimators": params.get("n_estimators", 500),
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_weight": params.get("min_child_weight", 3),
        "reg_alpha": params.get("reg_alpha", 1e-6),
        "reg_lambda": params.get("reg_lambda", 1e-6),
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        train_df[features].values, train_df[target_col].values,
        eval_set=[(val_df[features].values, val_df[target_col].values)],
        verbose=False,
    )

    preds = model.predict(test_df[features].values)
    return calculate_delay_metrics(test_df[target_col].values, preds)


def train_eval_lightgbm(df, features, target_col, fold):
    """Trains LightGBM on one fold and returns test metrics."""
    params = load_params("lightgbm")

    train_df = df[df["date"] < fold["train_end"]].dropna(subset=features + [target_col])
    val_df = df[
        (df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])
    ].dropna(subset=features + [target_col])
    test_df = df[
        (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    ].dropna(subset=features + [target_col])

    if len(test_df) == 0:
        return None

    lgb_params = {
        "n_estimators": params.get("n_estimators", 500),
        "num_leaves": params.get("num_leaves", 63),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        # subsample only takes effect when subsample_freq is nonzero
        "subsample_freq": params.get("subsample_freq", 0),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 1e-6),
        "reg_lambda": params.get("reg_lambda", 1e-6),
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        train_df[features].values, train_df[target_col].values,
        eval_set=[(val_df[features].values, val_df[target_col].values)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    preds = model.predict(test_df[features].values)
    return calculate_delay_metrics(test_df[target_col].values, preds)


def train_eval_lightgbm_quantile(df, features, target_col, fold):
    """
    Trains one LightGBM per quantile on a fold, returns point metrics for the
    median plus pinball/coverage for the sorted interval.
    """
    params = load_params("lightgbm")

    train_df = df[df["date"] < fold["train_end"]].dropna(subset=features + [target_col])
    val_df = df[
        (df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])
    ].dropna(subset=features + [target_col])
    test_df = df[
        (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    ].dropna(subset=features + [target_col])

    if len(test_df) == 0:
        return None

    # the early half of the val window drives early stopping, the later half
    # is a dedicated calibration set the models never selected against
    es_df, cal_df = split_calibration_window(val_df)

    base_params = {
        "n_estimators": params.get("n_estimators", 500),
        "num_leaves": params.get("num_leaves", 63),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "subsample_freq": params.get("subsample_freq", 0),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 1e-6),
        "reg_lambda": params.get("reg_lambda", 1e-6),
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    test_preds = []
    cal_preds = []
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
        model.fit(
            train_df[features].values, train_df[target_col].values,
            eval_set=[(es_df[features].values, es_df[target_col].values)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        test_preds.append(model.predict(test_df[features].values))
        cal_preds.append(model.predict(cal_df[features].values))

    sorted_preds = sort_quantile_predictions(np.column_stack(test_preds))
    y_test = test_df[target_col].values

    metrics = calculate_delay_metrics(y_test, sorted_preds[:, len(QUANTILE_ALPHAS) // 2])
    metrics.update(calculate_quantile_metrics(y_test, sorted_preds, alphas=QUANTILE_ALPHAS))

    # split-conformal calibration on the dedicated calibration half, applied
    # to the held-out test interval to restore nominal 80% coverage
    sorted_cal = sort_quantile_predictions(np.column_stack(cal_preds))
    offset = cqr_offset(cal_df[target_col].values, sorted_cal[:, 0], sorted_cal[:, -1], alpha=0.2)
    lo_cal, hi_cal = apply_cqr(sorted_preds[:, 0], sorted_preds[:, -1], offset)
    metrics["coverage_80_cqr"] = float(interval_coverage(y_test, lo_cal, hi_cal))
    metrics["interval_width_cqr"] = float(interval_width(lo_cal, hi_cal))
    metrics["cqr_offset"] = float(offset)
    return metrics


def train_eval_lstm(df, features, target_col, fold, device):
    """Trains LSTM on one fold and returns test metrics."""
    # imported lazily so the module stays usable in torch-free environments
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.lstm import LSTMTrainer, RouteDelayLSTM
    from src.training.sequence_utils import create_sequences_by_date, evaluate_model

    params = load_params("lstm")

    df_clean = df.dropna(subset=features + [target_col])

    train_mask = df_clean["date"] < fold["train_end"]
    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, features].values)

    target_scaler = StandardScaler()
    target_scaler.fit(df_clean.loc[train_mask, target_col].values.reshape(-1, 1))

    train_X, train_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=DATA_START, end_date=fold["train_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )
    val_X, val_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=fold["train_end"], end_date=fold["val_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )
    test_X, test_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=fold["test_start"], end_date=fold["test_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )

    if len(test_X) == 0:
        return None

    batch_size = params.get("batch_size", 64)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                       torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                       torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32),
                       torch.tensor(test_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    model = RouteDelayLSTM(
        input_size=len(features),
        hidden_size=params.get("hidden_size", 64),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.3),
    )

    lr = params.get("lr", 0.001)
    weight_decay = params.get("weight_decay", 0.01)
    trainer = LSTMTrainer(model, learning_rate=lr, device=device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    trainer.fit(train_loader, val_loader, epochs=50, early_stopping_patience=10, verbose=False)

    actuals, preds = evaluate_model(model, test_loader, device, target_scaler=target_scaler)
    return calculate_delay_metrics(actuals, preds)


def train_eval_tcn(df, features, target_col, fold, device):
    """Trains TCN on one fold and returns test metrics."""
    # imported lazily so the module stays usable in torch-free environments
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.tcn import RouteDelayTCN, TCNTrainer
    from src.training.sequence_utils import create_sequences_by_date, evaluate_model
    from src.training.train.train_tcn import build_channel_list

    params = load_params("tcn")

    df_clean = df.dropna(subset=features + [target_col])

    train_mask = df_clean["date"] < fold["train_end"]
    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, features].values)

    target_scaler = StandardScaler()
    target_scaler.fit(df_clean.loc[train_mask, target_col].values.reshape(-1, 1))

    train_X, train_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=DATA_START, end_date=fold["train_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )
    val_X, val_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=fold["train_end"], end_date=fold["val_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )
    test_X, test_y = create_sequences_by_date(
        df_clean, features, target_col, scaler,
        start_date=fold["test_start"], end_date=fold["test_end"],
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )

    if len(test_X) == 0:
        return None

    batch_size = params.get("batch_size", 32)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                       torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                       torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32),
                       torch.tensor(test_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    depth = params.get("depth", 3)
    width = params.get("width", 32)
    num_channels = build_channel_list(depth, width)

    model = RouteDelayTCN(
        input_size=len(features),
        num_channels=num_channels,
        kernel_size=params.get("kernel_size", 3),
        dropout=params.get("dropout", 0.2),
    )

    lr = params.get("lr", 0.001)
    weight_decay = params.get("weight_decay", 0.01)
    trainer = TCNTrainer(model, learning_rate=lr, device=device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    trainer.fit(train_loader, val_loader, epochs=50, early_stopping_patience=10, verbose=False)

    actuals, preds = evaluate_model(model, test_loader, device, target_scaler=target_scaler)
    return calculate_delay_metrics(actuals, preds)


def eval_naive(df, target_col, fold):
    """Evaluates naive baseline (yesterday's delay) on one fold."""
    from src.models.simple_baselines import NaiveBaseline

    train_df = df[df["date"] < fold["train_end"]]
    test_mask = (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])

    if not test_mask.any():
        return None

    model = NaiveBaseline(target_col=target_col)
    model.fit(train_df)
    # predict over the full frame so the first test day sees the prior
    # actual instead of the median fallback, then score test rows only
    preds = model.predict(df)[test_mask]
    y_test = df.loc[test_mask, target_col]

    valid_mask = preds.notna()
    return calculate_delay_metrics(y_test[valid_mask].values, preds[valid_mask].values)


def eval_moving_average(df, target_col, fold):
    """Evaluates 7-day moving average baseline on one fold."""
    from src.models.simple_baselines import MovingAverageBaseline

    train_df = df[df["date"] < fold["train_end"]]
    test_mask = (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])

    if not test_mask.any():
        return None

    model = MovingAverageBaseline(window=7, target_col=target_col)
    model.fit(train_df)
    # full-frame predict for trailing context across the fold boundary
    preds = model.predict(df)[test_mask]
    y_test = df.loc[test_mask, target_col]

    valid_mask = preds.notna()
    return calculate_delay_metrics(y_test[valid_mask].values, preds[valid_mask].values)


def eval_lightgbm_classifier(df, features, target_col, fold):
    """
    Trains a P(mean delay > 15 min) classifier on one fold and returns
    decision-framed metrics (PR-AUC, ROC-AUC, base rate, Brier). This is the
    metric that maps to an actual decision, unlike MAE on a noisy aggregate.
    """
    params = load_params("lightgbm")
    threshold = 15

    train_df = df[df["date"] < fold["train_end"]].dropna(subset=features + [target_col])
    val_df = df[
        (df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])
    ].dropna(subset=features + [target_col])
    test_df = df[
        (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    ].dropna(subset=features + [target_col])

    y_train = (train_df[target_col].values > threshold).astype(int)
    if len(test_df) == 0 or y_train.sum() == 0 or y_train.sum() == len(y_train):
        return None

    clf_params = {
        "n_estimators": params.get("n_estimators", 500),
        "num_leaves": params.get("num_leaves", 63),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "subsample_freq": params.get("subsample_freq", 0),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 1e-6),
        "reg_lambda": params.get("reg_lambda", 1e-6),
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(objective="binary", **clf_params)
    model.fit(
        train_df[features].values, y_train,
        eval_set=[(val_df[features].values, (val_df[target_col].values > threshold).astype(int))],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    proba = model.predict_proba(test_df[features].values)[:, 1]
    return calculate_classification_metrics(test_df[target_col].values, proba, threshold)


def eval_seasonal_naive(df, target_col, fold):
    """Evaluates the seasonal naive baseline (same weekday last week) on one fold."""
    from src.models.simple_baselines import SeasonalNaiveBaseline

    train_df = df[df["date"] < fold["train_end"]]
    test_mask = (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])

    if not test_mask.any():
        return None

    model = SeasonalNaiveBaseline(seasonality=7, target_col=target_col)
    model.fit(train_df)
    # full-frame predict for trailing context across the fold boundary
    preds = model.predict(df)[test_mask]
    y_test = df.loc[test_mask, target_col]

    valid_mask = preds.notna()
    return calculate_delay_metrics(y_test[valid_mask].values, preds[valid_mask].values)


def eval_climatology_quantile(df, target_col, fold):
    """
    Per-route trailing empirical quantiles as the no-skill benchmark for the
    quantile models. Quantiles use trailing actuals only (shift by one, then
    a 90-day window) computed across the whole frame, scored on test rows.
    """
    test_mask = (df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])
    if not test_mask.any():
        return None

    grouped = df.groupby("route")[target_col]
    quantile_cols = []
    for alpha in QUANTILE_ALPHAS:
        trailing = grouped.transform(
            lambda s, a=alpha: s.shift(1).rolling(90, min_periods=30).quantile(a)
        )
        quantile_cols.append(trailing[test_mask].values)

    preds = np.column_stack(quantile_cols)
    y_test = df.loc[test_mask, target_col].values
    valid = ~np.isnan(preds).any(axis=1) & ~np.isnan(y_test)
    if not valid.any():
        return None

    sorted_preds = sort_quantile_predictions(preds[valid])
    y_valid = y_test[valid]

    metrics = calculate_delay_metrics(y_valid, sorted_preds[:, len(QUANTILE_ALPHAS) // 2])
    metrics.update(calculate_quantile_metrics(y_valid, sorted_preds, alphas=QUANTILE_ALPHAS))
    return metrics


def aggregate_fold_metrics(all_fold_metrics):
    """Computes mean and std for each metric across folds."""
    metric_keys = [
        "mae", "rmse", "r2", "within_15", "median_ae",
        "coverage_80", "interval_width", "coverage_80_cqr", "interval_width_cqr",
        "pinball_10", "pinball_50", "pinball_90", "wis",
        "below_q10", "below_q50", "below_q90",
        "pr_auc", "roc_auc", "base_rate", "brier",
    ]
    result = {}

    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics if m is not None and m.get(key) is not None]
        if values:
            result[key] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "folds": [round(v, 2) for v in values],
            }

    return result


def parse_model_selection(models, raw):
    """Validates a comma-separated --models value against the known model names."""
    if raw is None:
        return list(models)
    selected = [name.strip() for name in raw.split(",") if name.strip()]
    if not selected:
        raise SystemExit(f"no models selected, available: {list(models)}")
    unknown = [name for name in selected if name not in models]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}, available: {list(models)}")
    return selected


def main():
    """Runs walk-forward validation across all folds for the selected models."""
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument(
        "--models",
        default="naive,ma,seasonal_naive,climatology_q,xgboost,lightgbm,lightgbm_q,lightgbm_clf",
        help="comma-separated subset of models to run (default: every model except "
             "the torch comparators; pass lstm,tcn explicitly to include them)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    raw_df = load_raw()
    target_col = "avg_arr_delay"

    models = {
        "naive": {"func": eval_naive, "features": None},
        "ma": {"func": eval_moving_average, "features": None},
        "seasonal_naive": {"func": eval_seasonal_naive, "features": None},
        "climatology_q": {"func": eval_climatology_quantile, "features": None},
        "xgboost": {"func": train_eval_xgboost, "features": "tabular"},
        "lightgbm": {"func": train_eval_lightgbm, "features": "tabular"},
        "lightgbm_q": {"func": train_eval_lightgbm_quantile, "features": "tabular"},
        "lightgbm_clf": {"func": eval_lightgbm_classifier, "features": "tabular"},
        "lstm": {"func": train_eval_lstm, "features": "seq"},
        "tcn": {"func": train_eval_tcn, "features": "seq"},
    }

    selected = parse_model_selection(models, args.models)
    models = {name: models[name] for name in selected}

    # torch only enters the process when a sequence model was selected
    device = None
    if any(name in models for name in ("lstm", "tcn")):
        import torch

        from src.models.device import get_device

        torch.manual_seed(42)
        device = get_device()
        print(f"device: {device}")

    print(f"{len(raw_df):,} route-day rows")
    print(f"Models: {', '.join(models)}")
    print(f"{len(WALK_FORWARD_FOLDS)} folds\n")

    results = {name: [] for name in models}
    feature_lists = {"tabular": [], "seq": []}

    for fold_idx, fold in enumerate(WALK_FORWARD_FOLDS):
        print(f"Fold {fold_idx + 1}: test {fold['test_start']} to {fold['test_end']}")

        # feature statistics are rebuilt at this fold's own train_end so no
        # post-cutoff target information reaches the fold through route stats
        # or fill medians
        print("  building fold features...", flush=True)
        fold_df = build_fold_features(raw_df, fold["train_end"])
        feature_lists["tabular"] = [c for c in TABULAR_FEATURES if c in fold_df.columns]
        feature_lists["seq"] = [c for c in SEQUENCE_MODEL_FEATURES if c in fold_df.columns]

        for model_name, config in models.items():
            print(f"  {model_name}...", end=" ", flush=True)

            features = feature_lists[config["features"]] if config["features"] else None
            if config["features"] is None:
                metrics = config["func"](fold_df, target_col, fold)
            elif model_name in ("lstm", "tcn"):
                metrics = config["func"](fold_df, features, target_col, fold, device)
            else:
                metrics = config["func"](fold_df, features, target_col, fold)

            if metrics:
                results[model_name].append(metrics)
                if metrics.get("mae") is not None:
                    print(f"MAE={metrics['mae']:.2f}, within_15={metrics['within_15']:.1f}%")
                elif metrics.get("pr_auc") is not None:
                    print(f"PR-AUC={metrics['pr_auc']:.3f}, base_rate={metrics['base_rate']:.1f}%")
                else:
                    print("done")
            else:
                print("skipped (no test data)")

        print()

    summary = {}
    for model_name, fold_metrics in results.items():
        summary[model_name] = aggregate_fold_metrics(fold_metrics)

    print("Walk-forward summary:")
    for model_name, agg in summary.items():
        mae = agg.get("mae", {})
        print(f"  {model_name:12s}: MAE {mae.get('mean', 'N/A')} +/- {mae.get('std', 'N/A')}")

    # provenance is stamped per model, not once at the top, so a subset run
    # cannot relabel entries it did not recompute with the current methodology
    provenance = tracking.provenance_tags(features_df=raw_df, features_path=RAW_PATH)
    run_stamp = {
        "feature_state": "rebuilt per fold at each fold's train_end",
        "hpo_window": f"{HPO_TRAIN_END} to {HPO_VAL_END}",
        "params": "configs/best_params_*.json",
        "git_sha": provenance.get("git_sha", "unknown"),
    }

    output = {
        "n_folds": len(WALK_FORWARD_FOLDS),
        "folds": WALK_FORWARD_FOLDS,
        "models": {},
        "runs": {},
    }

    # a subset run must extend the published results, not clobber them
    output_path = OUTPUTS_DIR / "walk_forward_results.json"
    if output_path.exists():
        with open(output_path) as f:
            prior = json.load(f)
        output["models"] = prior.get("models", {})
        output["runs"] = prior.get("runs", {})

    output["models"].update(summary)
    for model_name in summary:
        output["runs"][model_name] = run_stamp

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # tracking runs last so a store failure can never cost the results file,
    # one parent run per model with a nested run per fold, no-op without mlflow
    for model_name, fold_metrics in results.items():
        with tracking.start_run(
            run_name=f"walk_forward_{model_name}", tags={"stage": "walk_forward", **provenance}
        ):
            features_kind = models[model_name]["features"]
            tracking.log_params({
                "model": model_name,
                "n_folds": len(WALK_FORWARD_FOLDS),
                "n_features": len(feature_lists[features_kind]) if features_kind else 0,
            })
            for fold_idx, metrics in enumerate(fold_metrics):
                with tracking.start_run(run_name=f"fold_{fold_idx + 1}", nested=True):
                    tracking.log_metrics(metrics)

            aggregates = {}
            for key, agg in summary[model_name].items():
                aggregates[f"{key}_mean"] = agg["mean"]
                aggregates[f"{key}_std"] = agg["std"]
            tracking.log_metrics(aggregates)


if __name__ == "__main__":
    main()
