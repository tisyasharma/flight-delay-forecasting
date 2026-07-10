"""
Optional MLflow experiment tracking. Every helper degrades to a no-op when
mlflow is not installed, so training scripts behave identically with or
without the [track] extra. Store and experiment come from MLFLOW_TRACKING_URI
and MLFLOW_EXPERIMENT, defaulting to a local sqlite file (mlflow 3.x put the
old ./mlruns file store into maintenance mode and raises on it by default).
"""

import hashlib
import math
import numbers
import os
import subprocess
import sys
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None

# anchored to the repo root so runs land in the same store regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_EXPERIMENT = "flight-delay-forecasting"

_TRACKED_LIBRARIES = ("lightgbm", "xgboost", "optuna", "scikit-learn", "pandas", "numpy")

_warned = False


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


def _git_state():
    """Returns (short sha, dirty flag) for the repo, or unknowns outside git."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        return sha, "true" if status.strip() else "false"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown", "unknown"


def provenance_tags(features_df=None, features_path=None):
    """
    Builds run tags recording code version, data fingerprint, and library
    versions so every tracked run is traceable to its exact inputs.
    """
    sha, dirty = _git_state()
    tags = {"git_sha": sha, "git_dirty": dirty}

    for lib in _TRACKED_LIBRARIES:
        try:
            tags[f"version_{lib.replace('-', '_')}"] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            continue

    if features_path is not None:
        path = Path(features_path)
        if path.exists():
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            tags["features_hash"] = digest.hexdigest()[:16]

    if features_df is not None and "date" in features_df.columns:
        tags["features_rows"] = str(len(features_df))
        tags["features_start"] = str(features_df["date"].min())[:10]
        tags["features_end"] = str(features_df["date"].max())[:10]

    return tags


@contextmanager
def start_run(run_name=None, nested=False, tags=None):
    """
    Opens an mlflow run against the configured store. Yields None and records
    nothing when mlflow is absent, callers never need to guard.
    """
    global _warned
    if mlflow is None:
        if not _warned:
            print(
                "tracking: mlflow is not installed, run will not be recorded "
                "(pip install -e '.[track]' to enable)",
                file=sys.stderr,
            )
            _warned = True
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
