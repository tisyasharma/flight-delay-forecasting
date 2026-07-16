import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DATA_START,
    FINAL_TEST_START,
    SEQUENCE_LENGTH,
    SEQUENCE_MODEL_FEATURES,
    TEST_END,
    TRAIN_END,
    VAL_END,
)
from src.evaluation.metrics import calculate_delay_metrics
from src.models.tcn import RouteDelayTCN, TCNTrainer
from src.training.common import load_tuned_params
from src.training.sequence_utils import create_sequences_by_date, evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "depth": 3,
    "width": 32,
    "kernel_size": 3,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 32,
    "weight_decay": 0.01,
}


def build_channel_list(depth, width):
    """Creates a channel list that doubles width up to the final layer."""
    if depth == 2:
        return [width, width * 2]
    elif depth == 3:
        return [width, width * 2, width * 2]
    else:
        return [width, width, width * 2, width * 2]


def main():
    parser = argparse.ArgumentParser(description="Train the sequence model")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="also evaluate the locked 2025 H1 holdout, release gate use only",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    from src.models.device import get_device
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
    devtest_mask = (df_clean["date"] >= VAL_END) & (df_clean["date"] < FINAL_TEST_START)

    # the historical devtest and locked-final-test spans predate the serving
    # generation's train_end, so they are in-sample now and skipped
    holdouts_valid = pd.Timestamp(FINAL_TEST_START) > pd.Timestamp(VAL_END)

    print(f"train={train_mask.sum():,}  val={val_mask.sum():,}  devtest={devtest_mask.sum():,}")
    if not holdouts_valid:
        print("historical devtest/final-test spans are inside the training window, skipping")

    scaler = StandardScaler()
    scaler.fit(df_clean.loc[train_mask, available_features].values)

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
    if holdouts_valid:
        test_X, test_y = create_sequences_by_date(
            df_clean, available_features, target_col, scaler,
            start_date=VAL_END, end_date=FINAL_TEST_START,
            sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler
        )
    else:
        test_X, test_y = train_X[:0], train_y[:0]

    print(f"sequences: train={len(train_X):,}  val={len(val_X):,}  devtest={len(test_X):,}")

    hp = load_tuned_params("tcn", DEFAULT_PARAMS)
    batch_size = hp.get("batch_size", 32)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_X, dtype=torch.float32),
            torch.tensor(test_y, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False
    )

    depth = hp.get("depth", 3)
    width = hp.get("width", 32)
    kernel_size = hp.get("kernel_size", 3)
    dropout = hp.get("dropout", 0.2)
    lr = hp.get("lr", 0.001)
    weight_decay = hp.get("weight_decay", 0.01)
    num_channels = build_channel_list(depth, width)

    model = RouteDelayTCN(
        input_size=len(available_features),
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    print(f"receptive field: {model.receptive_field}")

    trainer = TCNTrainer(model, learning_rate=lr, device=device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=50, early_stopping_patience=10,
        verbose=True
    )

    print(f"\n{len(history['train_loss'])} epochs, best val loss: {min(history['val_loss']):.4f}")

    if holdouts_valid:
        actuals, preds = evaluate_model(model, test_loader, device, target_scaler=target_scaler)
        metrics = calculate_delay_metrics(actuals, preds)

        print(f"Devtest MAE: {metrics['mae']:.2f} min")
        print(f"Devtest RMSE: {metrics['rmse']:.2f} min")
        print(f"Devtest R2: {metrics['r2']:.3f}")
        print(f"Devtest within 15min: {metrics['within_15']:.1f}%")

    if args.final_test and not holdouts_valid:
        print("final test span already consumed by the previous generation, skipping")
    if args.final_test and holdouts_valid:
        final_X, final_y = create_sequences_by_date(
            df_clean, available_features, target_col, scaler,
            start_date=FINAL_TEST_START, end_date=TEST_END,
            sequence_length=SEQUENCE_LENGTH, target_scaler=target_scaler,
        )
        final_loader = DataLoader(
            TensorDataset(torch.tensor(final_X, dtype=torch.float32),
                          torch.tensor(final_y, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )
        final_actuals, final_preds = evaluate_model(
            model, final_loader, device, target_scaler=target_scaler
        )
        final_metrics = calculate_delay_metrics(final_actuals, final_preds)
        print(f"\nLOCKED FINAL TEST ({FINAL_TEST_START} onward, one-shot release gate)")
        print(f"Final MAE: {final_metrics['mae']:.2f} min  RMSE: {final_metrics['rmse']:.2f} min")
        print(f"Final within 15min: {final_metrics['within_15']:.1f}%")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "history": history,
        "input_size": len(available_features),
        "num_channels": num_channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "feature_list": available_features
    }, MODELS_DIR / "best_tcn_arr_delay.pt")

    joblib.dump(scaler, MODELS_DIR / "feature_scaler_tcn_arr_delay.pkl")
    joblib.dump(target_scaler, MODELS_DIR / "target_scaler_tcn.pkl")
    print("saved model + scalers")


if __name__ == "__main__":
    main()
