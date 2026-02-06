import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import RouteDelayLSTM, LSTMTrainer
from src.config import (
    TRAIN_END, VAL_END, TEST_END, DATA_START,
    SEQUENCE_LENGTH, SEQUENCE_MODEL_FEATURES,
)
from src.evaluation.metrics import calculate_delay_metrics
from src.training.sequence_utils import create_sequences_by_date, evaluate_model

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 64,
    "weight_decay": 0.01,
}


def load_tuned_params():
    """Loads Optuna best params if available, otherwise returns defaults."""
    params_path = MODELS_DIR / "best_params_lstm.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"Using Optuna-tuned params from {params_path.name}")
        return tuned
    print("No tuned params found, using defaults")
    return DEFAULT_PARAMS.copy()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    from src.config import get_device
    device = get_device()
    print(f"device: {device}")

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    available_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    df_clean = df.dropna(subset=available_features + [target_col])
    print(f"{len(df_clean):,} samples, {len(available_features)} features")

    train_mask = df_clean["date"] < TRAIN_END
    val_mask = (df_clean["date"] >= TRAIN_END) & (df_clean["date"] < VAL_END)
    test_mask = df_clean["date"] >= VAL_END

    print(f"train={train_mask.sum():,}  val={val_mask.sum():,}  test={test_mask.sum():,}")

    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, available_features].values)

    # fit target scaler on training targets for normalized loss computation
    train_targets_raw = df_clean.loc[train_mask, target_col].values
    target_scaler = StandardScaler()
    target_scaler.fit(train_targets_raw.reshape(-1, 1))

    train_X, train_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=DATA_START, end_date=TRAIN_END,
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler
    )
    val_X, val_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=TRAIN_END, end_date=VAL_END,
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler
    )
    test_X, test_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=VAL_END, end_date=TEST_END,
        sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler
    )

    print(f"sequences: train={len(train_X):,}  val={len(val_X):,}  test={len(test_X):,}")

    hp = load_tuned_params()
    batch_size = hp.get("batch_size", 64)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    hidden_size = hp.get("hidden_size", 64)
    num_layers = hp.get("num_layers", 2)
    dropout = hp.get("dropout", 0.3)
    lr = hp.get("lr", 0.001)
    weight_decay = hp.get("weight_decay", 0.01)

    model = RouteDelayLSTM(
        input_size=len(available_features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    trainer = LSTMTrainer(model, learning_rate=lr, device=device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=100, early_stopping_patience=10,
        verbose=True
    )

    print(f"\n{len(history['train_loss'])} epochs, best val loss: {min(history['val_loss']):.4f}")

    actuals, preds = evaluate_model(model, test_loader, device, target_scaler=target_scaler)
    metrics = calculate_delay_metrics(actuals, preds)

    print(f"Test MAE: {metrics['mae']:.2f} min")
    print(f"Test RMSE: {metrics['rmse']:.2f} min")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"Within 15min: {metrics['within_15']:.1f}%")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "history": history,
        "input_size": len(available_features),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "feature_list": available_features
    }, MODELS_DIR / "best_lstm_arr_delay.pt")

    joblib.dump(scaler, MODELS_DIR / "feature_scaler_arr_delay.pkl")
    joblib.dump(target_scaler, MODELS_DIR / "target_scaler_lstm.pkl")
    print("saved model + scalers")


if __name__ == "__main__":
    main()
