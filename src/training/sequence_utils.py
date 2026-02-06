import numpy as np
import pandas as pd
import torch


def create_sequences_by_date(df, feature_cols, target_col, scaler,
                             start_date, end_date, sequence_length,
                             target_scaler=None):
    """Builds sliding window sequences per route within the given date range."""
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
            y_val = targets[idx]
            if target_scaler is not None:
                y_val = target_scaler.transform([[y_val]])[0][0]
            sequences_y.append(y_val)

    return np.array(sequences_X), np.array(sequences_y)


def evaluate_model(model, data_loader, device, target_scaler=None):
    """Runs inference and returns (actuals, predictions) arrays in original scale."""
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.extend(preds.cpu().numpy())
            all_actuals.extend(y_batch.numpy())

    actuals = np.array(all_actuals)
    predictions = np.array(all_preds)

    if target_scaler is not None:
        actuals = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return actuals, predictions
