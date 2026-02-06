import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import RouteDelayLSTM, LSTMTrainer
from src.config import (
    TRAIN_END, VAL_END, DATA_START,
    SEQUENCE_LENGTH, SEQUENCE_MODEL_FEATURES, get_device,
)
from src.training.sequence_utils import create_sequences_by_date, evaluate_model

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Loads features.csv, builds sequences for train and val."""
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    df_clean = df.dropna(subset=available_features + [target_col])

    train_mask = df_clean["date"] < TRAIN_END
    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, available_features].values)

    target_scaler = StandardScaler()
    target_scaler.fit(df_clean.loc[train_mask, target_col].values.reshape(-1, 1))

    train_X, train_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=DATA_START, end_date=TRAIN_END,
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )
    val_X, val_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=TRAIN_END, end_date=VAL_END,
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
    )

    return train_X, train_y, val_X, val_y, len(available_features), target_scaler


def objective(trial, train_X, train_y, val_X, val_y, input_size, target_scaler, device):
    """Optuna objective: train LSTM with sampled params, return val MAE."""
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False,
    )

    model = RouteDelayLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    trainer = LSTMTrainer(model, learning_rate=lr, device=device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    trainer.fit(
        train_loader, val_loader, epochs=50,
        early_stopping_patience=8, verbose=False, trial=trial,
    )

    actuals, preds = evaluate_model(model, val_loader, device, target_scaler=target_scaler)
    mae = float(np.mean(np.abs(actuals - preds)))

    return mae


def main():
    """Runs Optuna study for LSTM and saves best params."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print(f"device: {device}")

    train_X, train_y, val_X, val_y, input_size, target_scaler = load_data()
    print(f"sequences: train={len(train_X):,}  val={len(val_X):,}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda trial: objective(
            trial, train_X, train_y, val_X, val_y,
            input_size, target_scaler, device,
        ),
        n_trials=50,
        show_progress_bar=True,
    )

    print(f"\nBest val MAE: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    output_path = MODELS_DIR / "best_params_lstm.json"
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
