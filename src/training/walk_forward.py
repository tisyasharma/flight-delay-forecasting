import json
import random
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import RouteDelayLSTM, LSTMTrainer
from src.models.tcn import RouteDelayTCN, TCNTrainer
from src.config import (
    DATA_START, SEQUENCE_LENGTH, WALK_FORWARD_FOLDS,
    TABULAR_FEATURES, SEQUENCE_MODEL_FEATURES, get_device,
)
from src.evaluation.metrics import calculate_delay_metrics
from src.training.sequence_utils import create_sequences_by_date, evaluate_model
from src.training.train.train_tcn import build_channel_list

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_params(model_name):
    """Loads tuned params JSON for a model, returns empty dict if missing."""
    params_path = MODELS_DIR / f"best_params_{model_name}.json"
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


def train_eval_lstm(df, features, target_col, fold, device):
    """Trains LSTM on one fold and returns test metrics."""
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


def aggregate_fold_metrics(all_fold_metrics):
    """Computes mean and std for each metric across folds."""
    metric_keys = ["mae", "rmse", "r2", "within_15", "median_ae"]
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


def main():
    """Runs walk-forward validation across all folds for all models."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print(f"device: {device}")

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    tabular_features = [c for c in TABULAR_FEATURES if c in df.columns]
    seq_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"{len(df):,} samples")
    print(f"Tabular features: {len(tabular_features)}, Sequence features: {len(seq_features)}")
    print(f"{len(WALK_FORWARD_FOLDS)} folds\n")

    models = {
        "xgboost": {"func": train_eval_xgboost, "features": tabular_features},
        "lightgbm": {"func": train_eval_lightgbm, "features": tabular_features},
        "lstm": {"func": train_eval_lstm, "features": seq_features},
        "tcn": {"func": train_eval_tcn, "features": seq_features},
    }

    results = {name: [] for name in models}

    for fold_idx, fold in enumerate(WALK_FORWARD_FOLDS):
        print(f"Fold {fold_idx + 1}: test {fold['test_start']} to {fold['test_end']}")

        for model_name, config in models.items():
            print(f"  {model_name}...", end=" ", flush=True)

            if model_name in ("lstm", "tcn"):
                metrics = config["func"](df, config["features"], target_col, fold, device)
            else:
                metrics = config["func"](df, config["features"], target_col, fold)

            if metrics:
                results[model_name].append(metrics)
                print(f"MAE={metrics['mae']:.2f}, within_15={metrics['within_15']:.1f}%")
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

    output = {
        "n_folds": len(WALK_FORWARD_FOLDS),
        "folds": WALK_FORWARD_FOLDS,
        "models": summary,
    }

    output_path = MODELS_DIR / "walk_forward_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
