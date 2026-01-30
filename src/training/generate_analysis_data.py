"""
Generates JSON files for frontend visualizations.
Creates weather_impact.json, carrier_performance.json, feature_importance.json.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "trained_models"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """Load the features dataset."""
    features_path = DATA_DIR / "processed" / "features.csv"
    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_weather_impact_data(df):
    """Generate weather impact analysis data."""
    severity_labels = {
        0: "Clear",
        1: "Partly Cloudy",
        2: "Fog/Drizzle",
        3: "Rain",
        4: "Snow",
        5: "Thunderstorm"
    }

    weather_data = df.groupby("weather_severity_max").agg(
        avg_delay=("avg_arr_delay", "mean"),
        n_days=("avg_arr_delay", "count"),
        high_delay_pct=("avg_arr_delay", lambda x: (x > 15).mean())
    ).reset_index()

    by_severity = []
    for _, row in weather_data.iterrows():
        sev = int(row["weather_severity_max"])
        by_severity.append({
            "severity": sev,
            "label": severity_labels.get(sev, f"Level {sev}"),
            "avg_delay": round(float(row["avg_delay"]), 1),
            "n_days": int(row["n_days"]),
            "high_delay_pct": round(float(row["high_delay_pct"]), 3)
        })

    by_severity = sorted(by_severity, key=lambda x: x["severity"])

    # compute correlations between weather variables and delay
    weather_cols = {
        "apt1_temp_avg": "temp",
        "max_wind": "wind",
        "total_precip": "precip",
        "total_snowfall": "snowfall",
        "weather_severity_max": "severity"
    }

    correlation = {}
    for col, key in weather_cols.items():
        valid_data = df[[col, "avg_arr_delay"]].dropna()
        corr = valid_data[col].corr(valid_data["avg_arr_delay"])
        correlation[key] = round(float(corr), 3)

    return {
        "by_severity": by_severity,
        "correlation": correlation,
        "generated_at": datetime.now().isoformat()
    }


def generate_carrier_performance_data(df):
    """Generate carrier performance comparison data."""
    # the daily route data has carriers as a comma-separated string
    # we'll compute route-level stats instead
    route_stats = df.groupby("route").agg(
        avg_delay=("avg_arr_delay", "mean"),
        on_time_pct=("avg_arr_delay", lambda x: (x <= 15).mean()),
        n_days=("avg_arr_delay", "count"),
        severe_delay_pct=("avg_arr_delay", lambda x: (x > 30).mean()),
        cancel_rate=("cancel_rate", "mean")
    ).reset_index()

    routes = []
    for _, row in route_stats.iterrows():
        routes.append({
            "route": row["route"],
            "avg_delay": round(float(row["avg_delay"]), 1),
            "on_time_pct": round(float(row["on_time_pct"]), 3),
            "n_days": int(row["n_days"]),
            "cancel_rate": round(float(row["cancel_rate"]), 4),
            "severe_delay_pct": round(float(row["severe_delay_pct"]), 3)
        })

    routes = sorted(routes, key=lambda x: x["on_time_pct"], reverse=True)

    return {
        "routes": routes,
        "generated_at": datetime.now().isoformat()
    }


def generate_feature_importance_data():
    """Load feature importance from trained XGBoost model."""
    importance_csv = MODELS_DIR / "xgboost_feature_importance.csv"

    print("  Loading feature importance from trained model...")
    imp_df = pd.read_csv(importance_csv)
    total_importance = imp_df["importance"].sum()

    features = [
        {"feature": row["feature"], "importance": float(row["importance"] / total_importance)}
        for _, row in imp_df.iterrows()
    ]

    return {
        "features": features,
        "model": "XGBoost",
        "total_features": len(features),
        "generated_at": datetime.now().isoformat()
    }


def main():
    """Generate all frontend data files."""
    print("Generating Analysis Data for Frontend...")

    print("Loading processed data...")
    df = load_processed_data()
    print(f"Loaded {len(df):,} records")

    print("\nGenerating weather impact data...")
    weather_data = generate_weather_impact_data(df)
    weather_path = FRONTEND_DATA_DIR / "weather_impact.json"
    with open(weather_path, "w") as f:
        json.dump(weather_data, f, indent=2)
    print(f"  Saved to {weather_path}")

    print("\nGenerating route performance data...")
    route_data = generate_carrier_performance_data(df)
    route_path = FRONTEND_DATA_DIR / "carrier_performance.json"
    with open(route_path, "w") as f:
        json.dump(route_data, f, indent=2)
    print(f"  Saved to {route_path}")

    print("\nGenerating feature importance data...")
    feature_data = generate_feature_importance_data()
    feature_path = FRONTEND_DATA_DIR / "feature_importance.json"
    with open(feature_path, "w") as f:
        json.dump(feature_data, f, indent=2)
    print(f"  Saved to {feature_path}")

    print("\nAll data files generated successfully!")


if __name__ == "__main__":
    main()
