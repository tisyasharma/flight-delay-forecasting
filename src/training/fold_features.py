"""
Per-cutoff feature construction for walk-forward evaluation and tuning.
The canonical features.csv embeds train-window statistics (route stats,
fill medians) computed at the production cutoff, which postdates earlier
folds' training windows. Rebuilding the table at each fold's own train_end
keeps those statistics honest for that fold.
"""

from pathlib import Path

import pandas as pd

from src.build_features import FeatureBuilder

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "processed" / "daily_route_demand.csv"


def load_raw(path=RAW_PATH):
    """Loads the merged route-day table that FeatureBuilder consumes."""
    return pd.read_csv(path)


def build_fold_features(raw_df, train_end):
    """
    Builds the full feature table with every train-window statistic gated at
    train_end. Works on a copy, the raw frame can be reused across folds.
    """
    builder = FeatureBuilder(raw_df.copy(), train_end_date=train_end)
    df = builder.build()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["route", "date"]).reset_index(drop=True)
