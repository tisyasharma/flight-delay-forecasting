"""
Generates delay_forecasts.json for the frontend dashboard.
Runs all models (baselines, LSTM, XGBoost, LightGBM) on test data.
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import FlightDelayLSTM
from src.evaluation.metrics import calculate_delay_metrics
from src.config import TRAIN_END, TEST_START, SEQUENCE_MODEL_FEATURES

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"

TEST_END = "2025-06-30"
SEQUENCE_LENGTH = 28


def load_data():
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)
    return df


def calc_metrics(actual, pred, delay_threshold=15):
    """Calculate metrics with rounding."""
    metrics = calculate_delay_metrics(actual, pred, delay_threshold)

    return {
        "mae": round(metrics["mae"], 1),
        "within_15": round(metrics["within_15"], 1),
        "rmse": round(metrics["rmse"], 1),
        "mape": round(metrics["mape"], 1) if metrics["mape"] else None,
        "median_ae": round(metrics["median_ae"], 1),
        "directional": round(metrics["directional"], 1),
        "r2": round(metrics["r2"], 3)
    }


def create_sequences(df, route, scaler, test_start, features, target_col):
    route_df = df[df["route"] == route].sort_values("date").reset_index(drop=True)
    X = route_df[features].values
    y = route_df[target_col].values
    dates = route_df["date"].values

    X_scaled = scaler.transform(X)
    test_mask = route_df["date"] >= test_start
    test_idx = np.where(test_mask)[0]

    X_test, y_test, test_dates = [], [], []
    for idx in test_idx:
        if idx >= SEQUENCE_LENGTH:
            X_test.append(X_scaled[idx - SEQUENCE_LENGTH:idx])
            y_test.append(y[idx])
            test_dates.append(dates[idx])

    return np.array(X_test), np.array(y_test), test_dates


def generate_baseline_predictions(df, test_start, target_col):
    """Generate naive and moving average baseline predictions."""
    routes = df["route"].unique()
    lag_1_col = "lag_1_arr_delay"
    ma_col = "rolling_mean_7_arr_delay"

    all_preds = {}
    for route in routes:
        route_df = df[df["route"] == route].sort_values("date")
        test_df = route_df[route_df["date"] >= test_start]

        preds = []
        for _, row in test_df.iterrows():
            actual = float(row[target_col])
            preds.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "actual": round(actual, 1),
                "naive": round(float(row[lag_1_col]) if pd.notna(row[lag_1_col]) else actual, 1),
                "ma": round(float(row[ma_col]) if pd.notna(row[ma_col]) else actual, 1)
            })
        all_preds[route] = preds

    return all_preds


def generate_lstm_predictions(df, model, scaler, features, test_start, target_col):
    """Generate LSTM predictions."""
    routes = df["route"].unique()
    all_preds = {}

    model.eval()
    device = torch.device("mps")
    model = model.to(device)

    for route in routes:
        X_test, y_test, test_dates = create_sequences(df, route, scaler, test_start, features, target_col)
        if len(X_test) == 0:
            continue

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()

        route_preds = []
        for i, date in enumerate(test_dates):
            route_preds.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "actual": round(float(y_test[i]), 1),
                "lstm": round(float(preds[i]), 1)
            })
        all_preds[route] = route_preds

    return all_preds


def generate_tabular_predictions(df, model, features, test_start, target_col, model_key):
    """Generate predictions for a tabular model (XGBoost or LightGBM)."""
    routes = df["route"].unique()
    all_preds = {}

    for route in routes:
        route_df = df[df["route"] == route].sort_values("date")
        test_df = route_df[route_df["date"] >= test_start].dropna(subset=features + [target_col])

        if len(test_df) == 0:
            continue

        X_test = test_df[features].values
        y_test = test_df[target_col].values
        test_dates = test_df["date"].values

        preds = model.predict(X_test)

        route_preds = []
        for i, date in enumerate(test_dates):
            route_preds.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "actual": round(float(y_test[i]), 1),
                model_key: round(float(preds[i]), 1)
            })
        all_preds[route] = route_preds

    return all_preds


def merge_predictions(baseline_preds, lstm_preds, xgboost_preds, lightgbm_preds):
    """Merge all model predictions by route and date."""
    merged = {}

    for route in baseline_preds:
        baseline = baseline_preds.get(route, [])
        lstm = {p["date"]: p.get("lstm") for p in lstm_preds.get(route, [])}
        xgb = {p["date"]: p.get("xgboost") for p in xgboost_preds.get(route, [])}
        lgb = {p["date"]: p.get("lightgbm") for p in lightgbm_preds.get(route, [])}

        route_merged = []
        for p in baseline:
            p["lstm"] = lstm.get(p["date"], p["actual"])
            p["xgboost"] = xgb.get(p["date"], p["actual"])
            p["lightgbm"] = lgb.get(p["date"], p["actual"])
            route_merged.append(p)

        merged[route] = route_merged
    return merged


def generate_historical(df, test_start, target_col):
    historical = {}
    for route in df["route"].unique():
        route_df = df[df["route"] == route].sort_values("date")
        train_df = route_df[route_df["date"] < test_start]

        historical[route] = [
            {"date": row["date"].strftime("%Y-%m-%d"), "actual": round(float(row[target_col]), 1)}
            for _, row in train_df.iterrows()
        ]
    return historical


def main():
    print("Generating Delay Forecast JSON...")

    df = load_data()
    target_col = "avg_arr_delay"

    lstm_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]

    train_df = df[df["date"] < pd.Timestamp(TRAIN_END)]

    lstm_scaler = StandardScaler()
    lstm_scaler.fit(train_df[lstm_features].values)

    print("\nLoading models...")

    lstm_path = MODELS_DIR / "best_lstm_arr_delay.pt"
    checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
    lstm_model = FlightDelayLSTM(input_size=len(lstm_features), hidden_size=64, num_layers=2, dropout=0.3)
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    print("  LSTM loaded")

    xgb_path = MODELS_DIR / "xgboost_delay.pkl"
    xgb_model = joblib.load(xgb_path)
    xgb_features = joblib.load(MODELS_DIR / "xgboost_features.pkl")
    xgb_features = [f for f in xgb_features if f in df.columns]
    print(f"  XGBoost loaded ({len(xgb_features)} features)")

    lgb_path = MODELS_DIR / "lightgbm_delay.pkl"
    lgb_model = joblib.load(lgb_path)
    lgb_features = joblib.load(MODELS_DIR / "lightgbm_features.pkl")
    lgb_features = [f for f in lgb_features if f in df.columns]
    print(f"  LightGBM loaded ({len(lgb_features)} features)")

    test_start = pd.Timestamp(TEST_START)

    print("\nGenerating predictions...")
    baseline_preds = generate_baseline_predictions(df, test_start, target_col)
    lstm_preds = generate_lstm_predictions(df, lstm_model, lstm_scaler, lstm_features, test_start, target_col)
    xgboost_preds = generate_tabular_predictions(df, xgb_model, xgb_features, test_start, target_col, "xgboost")
    lightgbm_preds = generate_tabular_predictions(df, lgb_model, lgb_features, test_start, target_col, "lightgbm")

    merged = merge_predictions(baseline_preds, lstm_preds, xgboost_preds, lightgbm_preds)
    historical = generate_historical(df, test_start, target_col)

    print("\nCalculating metrics...")
    model_keys = ["naive", "ma", "lstm", "xgboost", "lightgbm"]

    all_metrics = {k: {"by_route": {}} for k in model_keys}
    all_actual = {k: [] for k in model_keys}
    all_pred = {k: [] for k in model_keys}

    for route, preds in merged.items():
        actuals = [p["actual"] for p in preds]
        for k in model_keys:
            model_preds = [p[k] for p in preds]
            all_metrics[k]["by_route"][route] = calc_metrics(actuals, model_preds)
            all_actual[k].extend(actuals)
            all_pred[k].extend(model_preds)

    for k in model_keys:
        all_metrics[k]["overall"] = calc_metrics(all_actual[k], all_pred[k])

    print("\nModel Comparison (Overall MAE in minutes):")
    for k, m in all_metrics.items():
        mae = m["overall"]["mae"]
        within = m["overall"]["within_15"]
        print(f"  {k.upper():12s}: MAE {mae:.2f} min, Hit Rate {within:.1f}%")

    models_output = {
        "naive": {"name": "Naive", "description": "Yesterday's delay (baseline)", "metrics": all_metrics["naive"]},
        "ma": {"name": "Moving Average", "description": "7-day rolling mean", "metrics": all_metrics["ma"]},
        "lstm": {"name": "LSTM", "description": "2-layer LSTM with attention mechanism", "metrics": all_metrics["lstm"]},
        "xgboost": {"name": "XGBoost", "description": "Gradient boosting with 57 engineered features", "metrics": all_metrics["xgboost"]},
        "lightgbm": {"name": "LightGBM", "description": "Leaf-wise gradient boosting with 57 engineered features", "metrics": all_metrics["lightgbm"]}
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target": "arrival_delay",
        "unit": "minutes",
        "test_period": {"start": TEST_START, "end": TEST_END},
        "training_period": {"start": df["date"].min().strftime("%Y-%m-%d"), "end": "2024-06-30"},
        "routes": list(merged.keys()),
        "historical": historical,
        "models": models_output,
        "predictions": merged
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "delay_forecasts.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
