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
    "HPO_TRAIN_END",
    "HPO_VAL_END",
    "FINAL_TEST_START",
    "COVID_START",
    "COVID_PEAK_END",
    "COVID_RECOVERY_END",
    "SEQUENCE_LENGTH",
    "WALK_FORWARD_FOLDS",
]

# serving-generation split. the previous generation (train_end 2024-01-01)
# ran the locked 2025-H1 final test exactly once on 2026-07-10; this
# generation trains through that consumed window and its blind test is the
# live public record, as the model card discloses
TRAIN_END = "2026-01-01"
VAL_END = "2026-06-01"
DATA_START = "2019-01-01"

# historical evaluation-protocol bounds, kept for the published study
# artifacts. both spans now predate TRAIN_END, so the trainers skip their
# devtest and final-test evaluations for the serving generation
TEST_START = "2024-07-01"
TEST_END = "2025-06-30"
FINAL_TEST_START = "2025-01-01"

# hyperparameter search holdout, fold 0's validation slice. every later
# half-year window in 2023-2024 is some walk-forward fold's test window and
# 2025 H1 was the locked final test, so this is the only clean choice
HPO_TRAIN_END = "2022-07-01"
HPO_VAL_END = "2023-01-01"

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
