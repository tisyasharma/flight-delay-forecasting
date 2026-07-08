"""
End-to-end smoke checks: a tiny model trained on the synthetic features must
carry real signal, and nothing in the suite may drag torch into the process.
"""

import sys

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae
from src.features.registry import TABULAR_FEATURES
from src.models.simple_baselines import NaiveBaseline, SeasonalNaiveBaseline


def test_tiny_lgbm_beats_train_mean(tiny_lgbm, train_end):
    """A 20-tree model on the synthetic features must clearly beat predicting the train mean."""
    model, features = tiny_lgbm
    train_mask = features["date"] < pd.Timestamp(train_end)
    test = features[~train_mask]
    y_test = test["avg_arr_delay"].to_numpy()

    model_mae = mae(y_test, model.predict(test[TABULAR_FEATURES]))
    train_mean = features.loc[train_mask, "avg_arr_delay"].mean()
    mean_mae = mae(y_test, np.full(len(test), train_mean))

    # the 0.85 margin absorbs float jitter without weakening the signal check
    assert model_mae < 0.85 * mean_mae


def test_seasonal_naive_beats_train_mean(synthetic_df, train_end):
    """Baselines predict sanely, and the seasonal one exploits the weekly cycle."""
    cutoff = pd.Timestamp(train_end)
    train_df = synthetic_df[synthetic_df["date"] < cutoff]
    test_mask = synthetic_df["date"] >= cutoff
    y_test = synthetic_df.loc[test_mask, "avg_arr_delay"].to_numpy()

    seasonal = SeasonalNaiveBaseline(seasonality=7).fit(train_df)
    seasonal_preds = seasonal.predict(synthetic_df)[test_mask].to_numpy()
    train_mean = train_df["avg_arr_delay"].mean()
    mean_mae = mae(y_test, np.full(test_mask.sum(), train_mean))

    # the weekly sinusoid in the fixture guarantees the seasonal naive wins
    assert mae(y_test, seasonal_preds) < mean_mae

    # no such guarantee for the plain naive, only that it predicts sanely
    naive_preds = NaiveBaseline().fit(train_df).predict(synthetic_df)
    assert len(naive_preds) == len(synthetic_df)
    assert np.isfinite(naive_preds.to_numpy()).all()


def test_torch_never_imported():
    """No import anywhere in the suite may drag torch into the process."""
    assert "torch" not in sys.modules
