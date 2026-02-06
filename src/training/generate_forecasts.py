import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import RouteDelayLSTM
from src.models.tcn import RouteDelayTCN
from src.evaluation.metrics import calculate_delay_metrics
from src.config import (
    TRAIN_END, TEST_START, TEST_END, SEQUENCE_LENGTH,
    SEQUENCE_MODEL_FEATURES, get_device,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "trained_models"
OUTPUT_DIR = PROJECT_ROOT / "frontend" / "public" / "data"

DISPLAY_ROUTES = 20


def load_data():
    """Loads features.csv sorted by route and date."""
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"]).reset_index(drop=True)
    return df


def calc_metrics(actual, pred, delay_threshold=15):
    """Rounds metrics for the JSON output, handles empty data."""
    metrics = calculate_delay_metrics(actual, pred, delay_threshold)

    if metrics["mae"] is None:
        return {k: None for k in ["mae", "within_15", "rmse", "mape", "median_ae", "threshold_acc", "r2"]}

    return {
        "mae": round(metrics["mae"], 1),
        "within_15": round(metrics["within_15"], 1),
        "rmse": round(metrics["rmse"], 1),
        "mape": round(metrics["mape"], 1) if metrics["mape"] else None,
        "median_ae": round(metrics["median_ae"], 1),
        "threshold_acc": round(metrics["threshold_acc"], 1),
        "r2": round(metrics["r2"], 3)
    }


def create_sequences(df, route, scaler, test_start, features, target_col):
    """Builds sliding window test sequences for one route."""
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
    """Generates naive lag-1 and 7-day moving average predictions."""
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
                "naive": round(float(row[lag_1_col]) if pd.notna(row[lag_1_col]) else 0.0, 1),
                "ma": round(float(row[ma_col]) if pd.notna(row[ma_col]) else 0.0, 1)
            })
        all_preds[route] = preds

    return all_preds


def generate_lstm_predictions(df, model, scaler, features, test_start, target_col, target_scaler=None):
    """Runs LSTM inference on test sequences for each route."""
    routes = df["route"].unique()
    all_preds = {}

    device = get_device()
    model.eval()
    model = model.to(device)

    for route in routes:
        X_test, y_test, test_dates = create_sequences(df, route, scaler, test_start, features, target_col)
        if len(X_test) == 0:
            continue

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()

        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

        route_preds = []
        for i, date in enumerate(test_dates):
            route_preds.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "actual": round(float(y_test[i]), 1),
                "lstm": round(float(preds[i]), 1)
            })
        all_preds[route] = route_preds

    return all_preds


def generate_tcn_predictions(df, model, scaler, features, test_start, target_col, target_scaler=None):
    """Runs TCN inference on test sequences for each route."""
    routes = df["route"].unique()
    all_preds = {}

    device = get_device()
    model.eval()
    model = model.to(device)

    for route in routes:
        X_test, y_test, test_dates = create_sequences(df, route, scaler, test_start, features, target_col)
        if len(X_test) == 0:
            continue

        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()

        if target_scaler is not None:
            preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

        route_preds = []
        for i, date in enumerate(test_dates):
            route_preds.append({
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "actual": round(float(y_test[i]), 1),
                "tcn": round(float(preds[i]), 1)
            })
        all_preds[route] = route_preds

    return all_preds


def generate_tabular_predictions(df, model, features, test_start, target_col, model_key):
    """Generates predictions for a tabular model (XGBoost or LightGBM)."""
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


def merge_predictions(baseline_preds, lstm_preds, tcn_preds, xgboost_preds, lightgbm_preds):
    """Merges all model predictions into one dict keyed by route."""
    merged = {}

    for route in baseline_preds:
        baseline = baseline_preds.get(route, [])
        lstm = {p["date"]: p.get("lstm") for p in lstm_preds.get(route, [])}
        tcn = {p["date"]: p.get("tcn") for p in tcn_preds.get(route, [])}
        xgb = {p["date"]: p.get("xgboost") for p in xgboost_preds.get(route, [])}
        lgb = {p["date"]: p.get("lightgbm") for p in lightgbm_preds.get(route, [])}

        route_merged = []
        for p in baseline:
            p["lstm"] = lstm.get(p["date"], None)
            p["tcn"] = tcn.get(p["date"], None)
            p["xgboost"] = xgb.get(p["date"], None)
            p["lightgbm"] = lgb.get(p["date"], None)
            route_merged.append(p)

        merged[route] = route_merged
    return merged


def generate_historical(df, test_start, target_col):
    """Gets pre-test data for the historical chart background."""
    historical = {}
    for route in df["route"].unique():
        route_df = df[df["route"] == route].sort_values("date")
        train_df = route_df[route_df["date"] < test_start].copy()
        train_df["date_str"] = train_df["date"].dt.strftime("%Y-%m-%d")
        train_df["actual"] = train_df[target_col].round(1)
        historical[route] = train_df[["date_str", "actual"]].rename(
            columns={"date_str": "date"}
        ).to_dict(orient="records")
    return historical


def get_top_display_routes(df, n=DISPLAY_ROUTES):
    """Returns the top n routes by total flight count for frontend display."""
    route_volume = df.groupby("route")["flight_count"].sum().sort_values(ascending=False)
    return route_volume.head(n).index.tolist()


def main():
    """Loads all models, generates predictions, computes metrics, saves JSON."""
    print("generating forecast JSON...")

    df = load_data()
    target_col = "avg_arr_delay"

    lstm_features = [c for c in SEQUENCE_MODEL_FEATURES if c in df.columns]

    lstm_scaler = joblib.load(MODELS_DIR / "feature_scaler_arr_delay.pkl")

    tcn_scaler_path = MODELS_DIR / "feature_scaler_tcn_arr_delay.pkl"
    tcn_scaler = joblib.load(tcn_scaler_path) if tcn_scaler_path.exists() else lstm_scaler

    lstm_target_scaler_path = MODELS_DIR / "target_scaler_lstm.pkl"
    lstm_target_scaler = joblib.load(lstm_target_scaler_path) if lstm_target_scaler_path.exists() else None

    tcn_target_scaler_path = MODELS_DIR / "target_scaler_tcn.pkl"
    tcn_target_scaler = joblib.load(tcn_target_scaler_path) if tcn_target_scaler_path.exists() else None

    # load all models
    checkpoint = torch.load(MODELS_DIR / "best_lstm_arr_delay.pt", map_location="cpu", weights_only=False)
    lstm_model = RouteDelayLSTM(
        input_size=checkpoint.get("input_size", len(lstm_features)),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.3)
    )
    lstm_model.load_state_dict(checkpoint["model_state_dict"])

    xgb_model = joblib.load(MODELS_DIR / "xgboost_delay.pkl")
    xgb_features = joblib.load(MODELS_DIR / "xgboost_features.pkl")
    xgb_features = [f for f in xgb_features if f in df.columns]

    tcn_checkpoint = torch.load(MODELS_DIR / "best_tcn_arr_delay.pt", map_location="cpu", weights_only=False)
    tcn_model = RouteDelayTCN(
        input_size=tcn_checkpoint.get("input_size", len(lstm_features)),
        num_channels=tcn_checkpoint.get("num_channels", [32, 64, 64]),
        kernel_size=tcn_checkpoint.get("kernel_size", 3),
        dropout=tcn_checkpoint.get("dropout", 0.2)
    )
    tcn_model.load_state_dict(tcn_checkpoint["model_state_dict"])

    lgb_model = joblib.load(MODELS_DIR / "lightgbm_delay.pkl")
    lgb_features = joblib.load(MODELS_DIR / "lightgbm_features.pkl")
    lgb_features = [f for f in lgb_features if f in df.columns]

    test_start = pd.Timestamp(TEST_START)

    baseline_preds = generate_baseline_predictions(df, test_start, target_col)
    lstm_preds = generate_lstm_predictions(df, lstm_model, lstm_scaler, lstm_features, test_start, target_col, target_scaler=lstm_target_scaler)
    tcn_preds = generate_tcn_predictions(df, tcn_model, tcn_scaler, lstm_features, test_start, target_col, target_scaler=tcn_target_scaler)
    xgboost_preds = generate_tabular_predictions(df, xgb_model, xgb_features, test_start, target_col, "xgboost")
    lightgbm_preds = generate_tabular_predictions(df, lgb_model, lgb_features, test_start, target_col, "lightgbm")

    merged = merge_predictions(baseline_preds, lstm_preds, tcn_preds, xgboost_preds, lightgbm_preds)
    historical = generate_historical(df, test_start, target_col)

    # compute metrics per model per route
    model_keys = ["naive", "ma", "lstm", "tcn", "xgboost", "lightgbm"]

    all_metrics = {k: {"by_route": {}} for k in model_keys}
    all_actual = {k: [] for k in model_keys}
    all_pred = {k: [] for k in model_keys}

    for route, preds in merged.items():
        for k in model_keys:
            paired = [(p["actual"], p[k]) for p in preds if p[k] is not None]
            if not paired:
                continue
            actuals, model_preds = zip(*paired)
            all_metrics[k]["by_route"][route] = calc_metrics(list(actuals), list(model_preds))
            all_actual[k].extend(actuals)
            all_pred[k].extend(model_preds)

    for k in model_keys:
        all_metrics[k]["overall"] = calc_metrics(all_actual[k], all_pred[k])

    for k, m in all_metrics.items():
        mae_val = m["overall"]["mae"]
        within = m["overall"]["within_15"]
        print(f"  {k:12s}: MAE {mae_val} min, hit rate {within}%")

    # filter to top routes by volume for frontend display
    display_routes = get_top_display_routes(df, n=DISPLAY_ROUTES)
    total_trained_routes = len(merged)
    display_merged = {r: merged[r] for r in display_routes if r in merged}
    display_historical = {r: historical[r] for r in display_routes if r in historical}

    # restrict by_route metrics to display routes (overall stays computed from all routes)
    for k in model_keys:
        all_metrics[k]["by_route"] = {
            r: all_metrics[k]["by_route"][r]
            for r in display_routes
            if r in all_metrics[k]["by_route"]
        }

    models_output = {
        "naive": {"name": "Naive", "description": "Yesterday's delay", "metrics": all_metrics["naive"]},
        "ma": {"name": "Moving Average", "description": "7-day rolling mean", "metrics": all_metrics["ma"]},
        "lstm": {"name": "LSTM", "description": "LSTM with attention", "metrics": all_metrics["lstm"]},
        "tcn": {"name": "TCN", "description": "Temporal convolutional network", "metrics": all_metrics["tcn"]},
        "xgboost": {"name": "XGBoost", "description": "Optuna-tuned gradient boosting", "metrics": all_metrics["xgboost"]},
        "lightgbm": {"name": "LightGBM", "description": "Optuna-tuned leaf-wise gradient boosting", "metrics": all_metrics["lightgbm"]}
    }

    # load walk-forward results if available
    wf_path = MODELS_DIR / "walk_forward_results.json"
    walk_forward_data = None
    if wf_path.exists():
        with open(wf_path) as f:
            walk_forward_data = json.load(f)
        print(f"Loaded walk-forward results ({walk_forward_data['n_folds']} folds)")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target": "avg_arrival_delay",
        "granularity": "daily_route_level",
        "unit": "minutes",
        "test_period": {"start": TEST_START, "end": TEST_END},
        "training_period": {"start": df["date"].min().strftime("%Y-%m-%d"), "end": pd.Timestamp(TRAIN_END).strftime("%Y-%m-%d")},
        "trained_on_routes": total_trained_routes,
        "routes": list(display_merged.keys()),
        "historical": display_historical,
        "models": models_output,
        "predictions": display_merged,
    }

    if walk_forward_data:
        output["walk_forward"] = walk_forward_data

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "delay_forecasts.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
