import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_models"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Reads features.csv and parses dates."""
    features_path = DATA_DIR / "processed" / "features.csv"
    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_weather_impact_data(df):
    """Delay stats grouped by weather severity, plus weather-delay correlations."""
    severity_labels = {
        0: "Clear",
        1: "Partly Cloudy",
        2: "Fog/Drizzle",
        3: "Rain",
        4: "Snow",
        5: "Thunderstorm"
    }

    def weighted_mean(group):
        weights = group["flight_count"]
        total = weights.sum()
        if total == 0:
            return 0.0
        return (group["avg_arr_delay"] * weights).sum() / total

    def weighted_high_delay_pct(group):
        weights = group["flight_count"]
        total = weights.sum()
        if total == 0:
            return 0.0
        return ((group["avg_arr_delay"] > 15) * weights).sum() / total

    weather_data = df.groupby("weather_severity_max").apply(
        lambda g: pd.Series({
            "avg_delay": weighted_mean(g),
            "n_flights": g["flight_count"].sum(),
            "high_delay_pct": weighted_high_delay_pct(g)
        }), include_groups=False
    ).reset_index()

    by_severity = []
    for _, row in weather_data.iterrows():
        sev = int(row["weather_severity_max"])
        by_severity.append({
            "severity": sev,
            "label": severity_labels.get(sev, f"Level {sev}"),
            "avg_delay": round(float(row["avg_delay"]), 1),
            "n_flights": int(row["n_flights"]),
            "high_delay_pct": round(float(row["high_delay_pct"]), 3)
        })

    by_severity = sorted(by_severity, key=lambda x: x["severity"])

    # weighted correlations between weather vars and delay
    weather_cols = {
        "apt1_temp_avg": "temp",
        "max_wind": "wind",
        "total_precip": "precip",
        "total_snowfall": "snowfall",
        "weather_severity_max": "severity"
    }

    correlation = {}
    for col, key in weather_cols.items():
        valid_data = df[[col, "avg_arr_delay", "flight_count"]].dropna()
        w = valid_data["flight_count"].values
        x = valid_data[col].values
        y = valid_data["avg_arr_delay"].values
        wx = x - np.average(x, weights=w)
        wy = y - np.average(y, weights=w)
        wcov = np.average(wx * wy, weights=w)
        corr = wcov / (np.sqrt(np.average(wx**2, weights=w)) * np.sqrt(np.average(wy**2, weights=w)))
        correlation[key] = round(float(corr), 3)

    return {
        "by_severity": by_severity,
        "correlation": correlation,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


def load_raw_bts_data():
    """Reads and concatenates all raw BTS CSV files."""
    raw_dir = DATA_DIR / "raw"
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    chunks = []
    for f in csv_files:
        chunk = pd.read_csv(f, low_memory=False)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def generate_carrier_performance_data():
    """Computes delay and cancellation stats for the top 10 airlines by volume."""
    REGIONAL_CODES = {"OO", "MQ", "OH", "YX", "9E", "YV", "QX", "ZW", "CP", "G7"}

    print("  loading raw BTS data...")
    raw = load_raw_bts_data()

    raw = raw[~raw["OP_UNIQUE_CARRIER"].isin(REGIONAL_CODES)].copy()

    carrier_counts = raw.groupby("OP_UNIQUE_CARRIER").size()
    top_carriers = carrier_counts.nlargest(10).index.tolist()
    raw = raw[raw["OP_UNIQUE_CARRIER"].isin(top_carriers)].copy()

    raw["FL_DATE"] = pd.to_datetime(raw["FL_DATE"], format="mixed")
    date_start = raw["FL_DATE"].min().strftime("%Y-%m-%d")
    date_end = raw["FL_DATE"].max().strftime("%Y-%m-%d")

    delay_base = raw[(raw["CANCELLED"] == 0) & raw["ARR_DELAY"].notna()].copy()

    delay_stats = delay_base.groupby("OP_UNIQUE_CARRIER").agg(
        avg_delay=("ARR_DELAY", "mean"),
        on_time_pct=("ARR_DELAY", lambda x: (x <= 15).mean()),
        n_flights=("ARR_DELAY", "count"),
        severe_delay_pct=("ARR_DELAY", lambda x: (x > 30).mean())
    )

    cancel_stats = raw.groupby("OP_UNIQUE_CARRIER").agg(
        cancel_rate=("CANCELLED", "mean")
    )

    stats = delay_stats.join(cancel_stats, how="left").reset_index()

    carriers = []
    for _, row in stats.iterrows():
        carriers.append({
            "code": row["OP_UNIQUE_CARRIER"],
            "avg_delay": round(float(row["avg_delay"]), 1),
            "on_time_pct": round(float(row["on_time_pct"]), 3),
            "n_flights": int(row["n_flights"]),
            "cancel_rate": round(float(row["cancel_rate"]), 4),
            "severe_delay_pct": round(float(row["severe_delay_pct"]), 3)
        })

    carriers = sorted(carriers, key=lambda x: x["on_time_pct"], reverse=True)

    return {
        "carriers": carriers,
        "date_range": {"start": str(date_start), "end": str(date_end)},
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


def generate_feature_importance_data():
    """Loads and normalizes feature importance from XGBoost and LightGBM."""
    model_configs = {
        "xgboost": {"name": "XGBoost", "file": "xgboost_feature_importance.csv"},
        "lightgbm": {"name": "LightGBM", "file": "lightgbm_feature_importance.csv"},
    }

    models = {}
    for key, cfg in model_configs.items():
        csv_path = MODELS_DIR / cfg["file"]
        imp_df = pd.read_csv(csv_path)
        total = imp_df["importance"].sum()

        features = [
            {"feature": row["feature"], "importance": float(row["importance"] / total)}
            for _, row in imp_df.iterrows()
        ]
        features.sort(key=lambda f: f["importance"], reverse=True)
        models[key] = {"name": cfg["name"], "features": features}

    total_features = len(next(iter(models.values()))["features"])

    return {
        "models": models,
        "total_features": total_features,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }


def main():
    """Generates all analysis JSON files for the frontend."""
    print("generating analysis data...")

    df = load_processed_data()
    print(f"{len(df):,} records loaded")

    weather_data = generate_weather_impact_data(df)
    with open(FRONTEND_DATA_DIR / "weather_impact.json", "w") as f:
        json.dump(weather_data, f, indent=2)

    carrier_data = generate_carrier_performance_data()
    with open(FRONTEND_DATA_DIR / "carrier_performance.json", "w") as f:
        json.dump(carrier_data, f, indent=2)

    feature_data = generate_feature_importance_data()
    with open(FRONTEND_DATA_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_data, f, indent=2)

    print("done")


if __name__ == "__main__":
    main()
