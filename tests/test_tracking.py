"""
The tracking shim must be a complete no-op when mlflow is absent, which is
exactly how CI installs the package (no [track] extra).
"""

import numpy as np
import pytest

from src import tracking


def test_start_run_and_helpers_noop_without_mlflow(tmp_path, monkeypatch):
    """Without mlflow every helper returns quietly and nothing is written."""
    if tracking.mlflow is not None:
        pytest.skip("mlflow installed, the no-op path is not exercisable here")

    monkeypatch.chdir(tmp_path)
    with tracking.start_run(run_name="anything", tags={"stage": "test"}) as run:
        assert run is None
        tracking.log_params({"model": "lightgbm_q", "n_folds": 4})
        tracking.log_metrics({"mae": 11.25, "mape": None})
        tracking.log_artifact(tmp_path / "does_not_exist.txt")

    # no mlflow.db, mlruns/, or anything else may appear on the no-op path
    assert list(tmp_path.iterdir()) == []


def test_clean_metrics_keeps_only_finite_numbers():
    """None, bools, strings, NaN, and inf must never reach mlflow."""
    raw = {
        "mae": 11.25,
        "n_samples": 5,
        "mape": None,
        "converged": True,
        "model": "lightgbm",
        "bad_nan": float("nan"),
        "bad_inf": float("inf"),
        "np_float": np.float64(2.5),
        "np_int": np.int64(7),
    }
    expected = {"mae": 11.25, "n_samples": 5.0, "np_float": 2.5, "np_int": 7.0}
    assert tracking._clean_metrics(raw) == expected


def test_nested_run_noop_without_mlflow():
    """Nested runs follow the same no-op contract as top-level ones."""
    if tracking.mlflow is not None:
        pytest.skip("mlflow installed, the no-op path is not exercisable here")

    with tracking.start_run(run_name="parent"):
        with tracking.start_run(run_name="child", nested=True) as child:
            assert child is None


def test_missing_mlflow_warns_once_on_stderr(monkeypatch, capsys):
    """The disabled-tracking warning must be visible but fire only once."""
    monkeypatch.setattr(tracking, "mlflow", None)
    monkeypatch.setattr(tracking, "_warned", False)

    with tracking.start_run(run_name="first"):
        pass
    with tracking.start_run(run_name="second"):
        pass

    err = capsys.readouterr().err
    assert err.count("mlflow is not installed") == 1


def test_provenance_tags_fingerprint_code_data_and_env(tmp_path):
    """Tags must carry the git state, data fingerprint, and library versions."""
    pd = pytest.importorskip("pandas")

    data_file = tmp_path / "features.csv"
    data_file.write_text("date,route\n2024-01-01,A_B\n")
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-03-31"]), "route": ["A_B", "A_B"]})

    tags = tracking.provenance_tags(features_df=df, features_path=data_file)

    assert "git_sha" in tags and "git_dirty" in tags
    assert tags["features_rows"] == "2"
    assert tags["features_start"] == "2024-01-01"
    assert tags["features_end"] == "2024-03-31"
    assert len(tags["features_hash"]) == 16
    assert "version_pandas" in tags and "version_numpy" in tags
