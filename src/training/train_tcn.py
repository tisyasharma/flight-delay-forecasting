"""
Train TCN model for delay prediction.
Same 22-feature set and splits as LSTM for a fair architecture comparison.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn import FlightDelayTCN, TCNTrainer
from src.config import TRAIN_END, VAL_END, SEQUENCE_MODEL_FEATURES
from src.evaluation.metrics import calculate_delay_metrics

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 28


def create_sequences_by_date(df, feature_cols, target_col, scaler,
                             start_date, end_date, sequence_length):
    """Create sequences for TCN. Each sequence uses only past data (no leakage)."""
    sequences_X = []
    sequences_y = []

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for route in df["route"].unique():
        route_df = df[df["route"] == route].copy()
        route_df = route_df.sort_values("date").reset_index(drop=True)

        features = scaler.transform(route_df[feature_cols].values)
        targets = route_df[target_col].values
        dates = route_df["date"].values

        for idx in range(len(route_df)):
            current_date = pd.Timestamp(dates[idx])

            if current_date < start_ts or current_date >= end_ts:
                continue
            if idx < sequence_length:
                continue

            sequences_X.append(features[idx - sequence_length:idx])
            sequences_y.append(targets[idx])

    return np.array(sequences_X), np.array(sequences_y)


def evaluate_model(model, data_loader, device):
    """Run inference and return actuals + predictions."""
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.extend(preds.cpu().numpy())
            all_actuals.extend(y_batch.numpy())

    return np.array(all_actuals), np.array(all_preds)


def main():
    print("Training TCN Delay Model (experimental)...")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)

    # same 22-feature set as LSTM for a fair comparison
    available_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    print(f"Total features: {len(available_features)}")

    df_clean = df.dropna(subset=available_features + [target_col])
    print(f"Total samples after dropping NaN: {len(df_clean):,}")

    train_mask = df_clean["date"] < TRAIN_END
    val_mask = (df_clean["date"] >= TRAIN_END) & (df_clean["date"] < VAL_END)
    test_mask = df_clean["date"] >= VAL_END

    print(f"\nTrain samples: {train_mask.sum():,}")
    print(f"Val samples:   {val_mask.sum():,}")
    print(f"Test samples:  {test_mask.sum():,}")

    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, available_features].values)

    train_X, train_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date="2019-01-01",
        end_date=TRAIN_END,
        sequence_length=SEQUENCE_LENGTH
    )

    val_X, val_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=TRAIN_END,
        end_date=VAL_END,
        sequence_length=SEQUENCE_LENGTH
    )

    test_X, test_y = create_sequences_by_date(
        df_clean, available_features, target_col, scaler,
        start_date=VAL_END,
        end_date="2025-07-01",
        sequence_length=SEQUENCE_LENGTH
    )

    print(f"\nSequences created:")
    print(f"  Train: {len(train_X):,}")
    print(f"  Val:   {len(val_X):,}")
    print(f"  Test:  {len(test_X):,}")

    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_X, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FlightDelayTCN(
        input_size=len(available_features),
        num_channels=[32, 64, 64],
        kernel_size=3,
        dropout=0.2
    )

    print(f"\nModel architecture:")
    print(f"  Input size:      {len(available_features)}")
    print(f"  Channels:        [32, 64, 64]")
    print(f"  Kernel size:     3")
    print(f"  Receptive field: {model.receptive_field}")
    print(f"  Dropout:         0.2")

    trainer = TCNTrainer(model, learning_rate=0.001, device=device)

    print("\nTraining TCN...")
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=50,
        early_stopping_patience=10,
        verbose=True
    )

    print(f"\nTraining completed after {len(history['train_loss'])} epochs")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")

    actuals, preds = evaluate_model(model, test_loader, device)
    metrics = calculate_delay_metrics(actuals, preds)

    print("\nTest Set Metrics:")
    print(f"  MAE:          {metrics['mae']:.2f} min")
    print(f"  RMSE:         {metrics['rmse']:.2f} min")
    print(f"  R2:           {metrics['r2']:.3f}")
    print(f"  Within 15min: {metrics['within_15']:.1f}%")

    model_path = MODELS_DIR / "best_tcn_arr_delay.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "history": history,
        "input_size": len(available_features),
        "num_channels": [32, 64, 64],
        "kernel_size": 3,
        "dropout": 0.2,
        "feature_list": available_features
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    print("\nTCN training complete.")


if __name__ == "__main__":
    main()
