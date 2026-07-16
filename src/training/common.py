"""Loading helpers shared by the tabular tuning and training scripts."""

import json
from pathlib import Path

from src.config import HPO_TRAIN_END, HPO_VAL_END, TABULAR_FEATURES
from src.training.fold_features import build_fold_features, load_raw

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_splits():
    """
    Builds features gated at the HPO cutoff and splits into train/val arrays.
    The canonical features.csv is not used here, its train-window statistics
    are computed at the production cutoff and would leak into the HPO holdout.
    """
    df = build_fold_features(load_raw(), HPO_TRAIN_END)

    available_features = [c for c in TABULAR_FEATURES if c in df.columns]
    target_col = "avg_arr_delay"

    train_df = df[df["date"] < HPO_TRAIN_END].dropna(subset=available_features + [target_col])
    val_df = df[(df["date"] >= HPO_TRAIN_END) & (df["date"] < HPO_VAL_END)].dropna(
        subset=available_features + [target_col]
    )

    X_train = train_df[available_features].values
    y_train = train_df[target_col].values
    X_val = val_df[available_features].values
    y_val = val_df[target_col].values

    return X_train, y_train, X_val, y_val


def load_tuned_params(model_name, defaults):
    """Loads Optuna best params for model_name if available, otherwise copies defaults."""
    params_path = CONFIGS_DIR / f"best_params_{model_name}.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"Using Optuna-tuned params from {params_path.name}")
        return tuned
    print("No tuned params found, using defaults")
    return defaults.copy()
