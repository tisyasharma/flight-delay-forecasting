"""
The daily live forecast: rolls the recursive engine from the last published
BTS actual through seven days past today and publishes the result for the
dashboard, plus an append-only prediction log that can be graded against
actuals as BTS publishes them. Every day between the last actual and the
horizon is a genuine forward forecast, the model consuming its own predicted
lags, which is exactly the behavior the recursion-depth backtest quantified.

Failure policy: any uncovered airport-day or fetch failure aborts the run
with no output written. The dashboard keeps yesterday's file and shows its
staleness banner: stale and correct beats fresh and wrong.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.build_features import FeatureBuilder
from src.features.registry import TABULAR_FEATURES
from src.forecasting.recursive import RecursiveForecaster
from src.pipelines.live_weather import build_weather_span
from src.process import merge_aviation_data, merge_weather_data

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "daily_route_demand.csv"
STATE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_state.json"
MODELS_DIR = PROJECT_ROOT / "trained_models"
OFFSETS_PATH = PROJECT_ROOT / "outputs" / "recursive_eval.json"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "data" / "live_forecasts.json"
LOG_DIR = PROJECT_ROOT / "data" / "live" / "predictions"

TRAILING_DAYS = 130
HORIZON_DAYS_PAST_TODAY = 7
QUANTILE_ALPHAS = (0.1, 0.5, 0.9)

# weeks of observed history shipped with the payload so the dashboard can
# anchor the forecast against the actual series it continues from
PAYLOAD_ACTUAL_DAYS = 56


def validate_offsets(offsets, floor):
    """
    The applied widening is the per-horizon offsets from the recursion
    backtest, a protocol-level estimate from different model instances, and
    conformal.json carries the serving generation's own one-step base offset.
    The per-horizon offsets must dominate that base at every depth, or the
    applied intervals would be narrower than this generation's own one-step
    calibration demands.
    """
    if min(offsets.values()) < floor:
        raise RuntimeError(
            f"per-horizon offsets fall below the serving base offset {floor}; "
            "recursive_eval must be re-fit before serving"
        )


def ensure_depth_validated(n_days, offsets):
    """
    Serving must never extrapolate the widening: every depth the run needs
    has to carry a backtest-fitted offset. Depth grows one per day between
    monthly BTS refreshes, so this trips when a refresh is overdue.
    """
    deepest_validated = max(offsets)
    if n_days > deepest_validated:
        raise RuntimeError(
            f"serving would need recursion depth {n_days}, beyond the deepest "
            f"backtest-validated offset (k={deepest_validated}); refresh the BTS "
            "actuals or extend recursive_eval before serving"
        )


def vintage_logged(log_path, today):
    """Whether the append-only log already holds a vintage for today's date."""
    if not log_path.exists():
        return False
    with open(log_path) as f:
        for line in f:
            if json.loads(line)["generated_at"][:10] == str(today.date()):
                return True
    return False


def load_serving_artifacts():
    """
    Boosters, feature state, and the interval-widening offsets, with the
    consistency checks that keep a mismatched deployment from serving:
    the release-shipped conformal.json must belong to the same generation as
    the repo's feature state, and the per-horizon offsets must dominate the
    generation's own base offset.
    """
    models = {
        alpha: lgb.Booster(
            model_file=str(MODELS_DIR / f"lightgbm_q{int(round(alpha * 100))}.txt"))
        for alpha in QUANTILE_ALPHAS
    }
    with open(STATE_PATH) as f:
        state = json.load(f)
    with open(OFFSETS_PATH) as f:
        offsets = {int(k): v for k, v in json.load(f)["offsets_by_k"].items()}
    with open(MODELS_DIR / "conformal.json") as f:
        base = json.load(f)
    if base.get("train_end_date") != state["train_end_date"]:
        raise RuntimeError(
            f"generation mismatch: released conformal.json is trained to "
            f"{base.get('train_end_date')}, the repo feature state to "
            f"{state['train_end_date']}; the boosters and the feature state "
            "must ship from the same generation"
        )
    n_booster = models[0.5].num_feature()
    if n_booster != len(TABULAR_FEATURES):
        raise RuntimeError(
            f"released boosters expect {n_booster} features, the checked-out "
            f"registry defines {len(TABULAR_FEATURES)}; release and code have skewed"
        )
    validate_offsets(offsets, base["offset"])
    return models, state, offsets, base


def build_live_frame(demand, state, last_actual, horizon_end):
    """
    Trailing actual rows plus weather-merged horizon rows, built into a
    feature frame with the injected train-time state. The engine overwrites
    every target-dependent column on horizon rows, so only the static
    features of those rows matter here.
    """
    routes = sorted(state["route_codes"])
    horizon_dates = pd.date_range(
        last_actual + pd.Timedelta(days=1), horizon_end, freq="D")

    weather, aviation = build_weather_span(horizon_dates[0], horizon_dates[-1])

    grid = pd.MultiIndex.from_product(
        [horizon_dates, routes], names=["date", "route"]).to_frame(index=False)
    grid["avg_arr_delay"] = np.nan
    grid["flight_count"] = 0
    grid = merge_weather_data(grid, weather)
    grid = merge_aviation_data(grid, aviation)

    trailing = demand[demand["date"] > last_actual - pd.Timedelta(days=TRAILING_DAYS)]
    combined = pd.concat([trailing, grid], ignore_index=True)

    builder = FeatureBuilder(combined, state=state)
    return builder.build()


def serialize_forecast_row(row):
    """
    One forecast row as it appears in both the dashboard payload and the
    append-only log, so the graded record and the displayed numbers derive
    from a single place and cannot drift apart.
    """
    return {
        "date": str(row["date"].date()),
        "k": int(row["k"]),
        "q50": round(float(row["q50"]), 2),
        "lo": round(float(row["lo_cal"]), 2),
        "hi": round(float(row["hi_cal"]), 2),
    }


def main():
    demand = pd.read_csv(DEMAND_PATH, parse_dates=["date"])
    models, state, offsets, conformal_base = load_serving_artifacts()

    last_actual = demand["date"].max()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    horizon_end = today + pd.Timedelta(days=HORIZON_DAYS_PAST_TODAY)
    n_days = int((horizon_end - last_actual).days)
    if n_days <= 0:
        raise RuntimeError("nothing to forecast: actuals already reach the horizon")
    ensure_depth_validated(n_days, offsets)

    # append-only public record: one vintage per day. a same-day re-run
    # regenerates nothing, so the published dashboard can never diverge from
    # the vintage the log will be graded on
    log_path = LOG_DIR / f"{today.strftime('%Y-%m')}.ndjson"
    if vintage_logged(log_path, today):
        print("a vintage for today is already logged, nothing to regenerate")
        return

    print(f"last actual {last_actual.date()}, horizon {horizon_end.date()}, "
          f"{n_days} recursive steps")

    frame = build_live_frame(demand, state, last_actual, horizon_end)
    forecaster = RecursiveForecaster(models, state)
    out = forecaster.forecast(frame, last_actual, n_days, conformal_offset=offsets)

    # verification overlay: the last published month replayed from the prior
    # month's end, same engine and offsets. delay features roll from the
    # model's own predictions as in live serving, but weather stays archival,
    # so this bounds recursion cost rather than reproducing serving conditions.
    # purely retrospective and never logged, the prediction log stays vintages-only
    ver_origin = last_actual.replace(day=1) - pd.Timedelta(days=1)
    ver_days = int((last_actual - ver_origin).days)
    verification = forecaster.forecast(
        frame, ver_origin, ver_days, conformal_offset=offsets)

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    model_version = os.environ.get("GITHUB_SHA", "local")[:12]

    history = demand[
        demand["date"] > last_actual - pd.Timedelta(days=PAYLOAD_ACTUAL_DAYS)
    ].sort_values("date")
    actuals_by_route = {
        route: [
            {"date": str(row["date"].date()), "actual": round(float(row["avg_arr_delay"]), 2)}
            for _, row in g.iterrows()
        ]
        for route, g in history.groupby("route")
    }

    def forecast_rows(g):
        return [serialize_forecast_row(row) for _, row in g.iterrows()]

    verification_by_route = {
        route: forecast_rows(g) for route, g in verification.groupby("route")
    }
    routes_payload = {}
    for route, g in out.groupby("route"):
        routes_payload[route] = {
            "actuals": actuals_by_route.get(route, []),
            "forecast": forecast_rows(g),
            "verification": verification_by_route.get(route, []),
        }

    trained_through = str(
        (pd.Timestamp(state["train_end_date"]) - pd.Timedelta(days=1)).date())
    payload = {
        "generated_at": generated_at,
        "model_version": model_version,
        "trained_through": trained_through,
        "last_actual_date": str(last_actual.date()),
        "today": str(today.date()),
        "horizon_end": str(horizon_end.date()),
        "verification_origin": str(ver_origin.date()),
        "interval": ("80% target; per-horizon widening from the recursion backtest, "
                     f"base offset {conformal_base['offset']:.2f} from the serving "
                     "generation's calibration split"),
        "routes": routes_payload,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
          f"({OUTPUT_PATH.stat().st_size // 1024} KB)")

    # each day's run logs one vintage of its published 7-day horizon
    # (today..today+7), earlier vintages are never rewritten
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    future = out[out["date"] >= today]
    with open(log_path, "a") as f:
        for _, row in future.iterrows():
            f.write(json.dumps({
                "generated_at": generated_at,
                "model_version": model_version,
                "route": row["route"],
                **serialize_forecast_row(row),
            }) + "\n")
    print(f"logged {len(future)} forecasts to {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
