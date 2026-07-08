from src.features.registry import (
    FEATURE_GROUPS,
    SEQUENCE_MODEL_FEATURES,
    TABULAR_FEATURES,
    WEATHER_FEATURES,
)

__all__ = [
    "FEATURE_GROUPS",
    "SEQUENCE_MODEL_FEATURES",
    "TABULAR_FEATURES",
    "WEATHER_FEATURES",
    "TRAIN_END",
    "VAL_END",
    "TEST_START",
    "TEST_END",
    "DATA_START",
    "COVID_START",
    "COVID_PEAK_END",
    "COVID_RECOVERY_END",
    "SEQUENCE_LENGTH",
    "WALK_FORWARD_FOLDS",
]

TRAIN_END = "2024-01-01"
VAL_END = "2024-07-01"
TEST_START = "2024-07-01"
TEST_END = "2025-06-30"
DATA_START = "2019-01-01"

COVID_START = "2020-03-01"
COVID_PEAK_END = "2021-06-01"
COVID_RECOVERY_END = "2022-06-01"

SEQUENCE_LENGTH = 28

WALK_FORWARD_FOLDS = [
    {
        "train_end": "2022-07-01",
        "val_end": "2023-01-01",
        "test_start": "2023-01-01",
        "test_end": "2023-06-30",
    },
    {
        "train_end": "2023-01-01",
        "val_end": "2023-07-01",
        "test_start": "2023-07-01",
        "test_end": "2023-12-31",
    },
    {
        "train_end": "2023-07-01",
        "val_end": "2024-01-01",
        "test_start": "2024-01-01",
        "test_end": "2024-06-30",
    },
    {
        "train_end": "2024-01-01",
        "val_end": "2024-07-01",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
]
