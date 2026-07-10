"""
Generates the calibration figures the README leads with: interval coverage
before and after conformal calibration, and a reliability curve for the
bad-delay-day classifier. Run with `python -m src.evaluation.plots`.
"""

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.config import TABULAR_FEATURES, WALK_FORWARD_FOLDS
from src.training.fold_features import build_fold_features, load_raw
from src.training.walk_forward import load_params

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "walk_forward_results.json"
TARGET = "avg_arr_delay"
DELAY_THRESHOLD = 15


def _base_params():
    params = load_params("lightgbm")
    params.update({"max_depth": -1, "random_state": 42, "n_jobs": -1, "verbose": -1})
    return params


def _classifier_reliability(fold):
    """Retrains one fold's bad-delay-day classifier to recover per-sample probs."""
    df = build_fold_features(load_raw(), fold["train_end"])
    features = [c for c in TABULAR_FEATURES if c in df.columns]

    train = df[df["date"] < fold["train_end"]].dropna(subset=features + [TARGET])
    val = df[(df["date"] >= fold["train_end"]) & (df["date"] < fold["val_end"])].dropna(subset=features + [TARGET])
    test = df[(df["date"] >= fold["test_start"]) & (df["date"] < fold["test_end"])].dropna(subset=features + [TARGET])

    clf = lgb.LGBMClassifier(objective="binary", **_base_params())
    clf.fit(train[features].values, (train[TARGET].values > DELAY_THRESHOLD).astype(int),
            eval_set=[(val[features].values, (val[TARGET].values > DELAY_THRESHOLD).astype(int))],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    proba = clf.predict_proba(test[features].values)[:, 1]
    return proba, (test[TARGET].values > DELAY_THRESHOLD).astype(int)


def plot_calibration():
    IMAGES_DIR.mkdir(exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text())["models"]["lightgbm_q"]
    raw = results["coverage_80"]["folds"]
    cqr = results["coverage_80_cqr"]["folds"]
    proba, y_bin = _classifier_reliability(WALK_FORWARD_FOLDS[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(raw))
    ax1.bar(x - 0.2, raw, 0.4, label="raw quantiles", color="#c44")
    ax1.bar(x + 0.2, cqr, 0.4, label="conformalized", color="#4a7")
    ax1.axhline(80, ls="--", color="#333", lw=1, label="nominal 80%")
    ax1.set_xticks(x, [f"fold {i + 1}" for i in x])
    ax1.set_ylim(70, 85)
    ax1.set_ylabel("empirical coverage of the 80% interval (%)")
    ax1.set_title("Interval coverage: conformal calibration restores nominal")
    ax1.legend(fontsize=8)

    frac_pos, mean_pred = calibration_curve(y_bin, proba, n_bins=10, strategy="quantile")
    ax2.plot([0, 1], [0, 1], ls="--", color="#333", lw=1, label="perfect")
    ax2.plot(mean_pred, frac_pos, "o-", color="#36c", label="bad-delay-day classifier")
    ax2.set_xlabel("predicted P(mean delay > 15 min)")
    ax2.set_ylabel("observed frequency")
    ax2.set_title("Classifier reliability (fold 4)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = IMAGES_DIR / "calibration.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_calibration()
