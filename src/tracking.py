"""
Optional MLflow experiment tracking. Every helper degrades to a no-op when
mlflow is not installed, so training scripts behave identically with or
without the [track] extra. Store and experiment come from MLFLOW_TRACKING_URI
and MLFLOW_EXPERIMENT, defaulting to a local sqlite file (mlflow 3.x put the
old ./mlruns file store into maintenance mode and raises on it by default).
"""

import math
import numbers
import os
from contextlib import contextmanager
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "flight-delay-forecasting"


def _clean_metrics(metrics):
    """Keeps only finite numeric values, mlflow rejects everything else."""
    clean = {}
    for key, value in metrics.items():
        # numbers.Real admits numpy scalars, bool is excluded explicitly
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            continue
        if not math.isfinite(value):
            continue
        clean[key] = float(value)
    return clean


@contextmanager
def start_run(run_name=None, nested=False, tags=None):
    """
    Opens an mlflow run against the configured store. Yields None and records
    nothing when mlflow is absent, callers never need to guard.
    """
    if mlflow is None:
        yield None
        return

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
        yield run


def log_params(params):
    """Logs a params dict when mlflow is present."""
    if mlflow is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics, step=None):
    """Logs the finite numeric subset of a metrics dict when mlflow is present."""
    if mlflow is None:
        return
    clean = _clean_metrics(metrics)
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_artifact(path):
    """Logs a file artifact when mlflow is present and the file exists."""
    if mlflow is None:
        return
    if Path(path).exists():
        mlflow.log_artifact(str(path))
