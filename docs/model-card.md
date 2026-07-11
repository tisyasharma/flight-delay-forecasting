# Model Card: Route-Level Delay Forecaster

## Model summary

- Model name: LightGBM point model (`lightgbm_point.txt` / `lightgbm_delay.pkl`) and LightGBM quantile triple (`lightgbm_q10/q50/q90.txt`)
- Version: 2026-07 serving generation (trained through 2025-12-31 for the live loop); the evaluated 2026-07 generation (cutoff 2024-01-01, clean HPO window, per-fold protocol) remains the basis of every published evaluation number below
- Owner: Tisya Sharma
- Date: 2026-07-11
- Task: daily mean arrival-delay forecasting per route (point and 10/50/90 quantiles), 50 busiest US directional routes, rolled forward recursively through 7 days past the run date. At BTS's monthly publication cadence the recursion depth cycles: roughly 45 days just after a month of actuals lands, climbing toward 80 the day before the next month publishes. The backtest validates every depth through 90, and the daily job refuses to serve deeper rather than extrapolate the widening
- Context of use: open evaluation and forecasting study; research-side comparators (XGBoost, LSTM, TCN) are documented but not part of the forecasting path

## Intended uses

- Supported uses: exploring daily route-average delay risk on the frozen 50-route set; methodology reference for walk-forward evaluation, conformal calibration, and recursive forward forecasting of route-level series
- Unsupported / out-of-scope uses: per-flight delay predictions; operational, booking, or crew decisions; routes or airports outside the frozen set; horizons beyond 7 days past the run date (recursion depths beyond the backtest's validated 90 days, which the daily job refuses at runtime); any safety-relevant application

## Training data

- Datasets used: `data/processed/features.csv` — see the [dataset card](dataset-card.md) for sources, rights, fill semantics, and known gaps
- Sampling window: 2019-01-01 to 2026-05-31; the serving generation trains to a 2026-01-01 cutoff with 2026-01..05 as its early-stopping and calibration window, while the evaluated generation used a 2024-01-01 cutoff (its devtest and locked-test spans predate the new cutoff, so the trainers skip them for the serving generation rather than report in-sample numbers)
- Labeling process: not applicable, targets are BTS-reported actual arrival delays
- Sensitive attributes involved: none
- Known gaps or biases: AK/HI aviation-variable coverage gap, COVID regime, lifted-index exclusion, resolved 2024-12 wind seam (details in the dataset card)

## Training procedure

- Preprocessing: scripted pipeline (`src/process.py`, `src/build_features.py`), train-window-gated statistics serialized to `data/processed/feature_state.json`, rebuilds gated by `src/data_checks.py`
- Features: 80 tabular features from `src/features/registry.py` (calendar, COVID flags, route stats, delay lags/rolling, daily/hourly weather, aviation weather, hub inbound-delay state)
- Hyperparameters: `configs/best_params_lightgbm.json` / `configs/best_params_xgboost.json`, selected by 50-trial Optuna TPE (seeded) minimizing validation MAE on 2022-07-01..2023-01-01 — fold 0's validation slice, chosen because every later half-year is some walk-forward fold's test window. Honest cost: parameters are selected on an older, smaller-train regime. Best search values: LightGBM 10.91, XGBoost 10.98 val MAE
- Compute / framework: LightGBM 4.6.0, XGBoost 3.3.0, Optuna 4.9.0, pandas 2.3.3, numpy 2.5.1 on Python 3.12 (pinned in `constraints.txt`); experiment records in a local MLflow store with git SHA, data fingerprint, and library-version tags per run
- Random seed strategy: numpy seed 42, model `random_state` 42, Optuna `TPESampler(seed=42)`

## Evaluation

- Validation protocol: 4 half-year walk-forward folds (2023-01..2024-12), feature statistics rebuilt at each fold's own train_end (`src/training/fold_features.py`); baselines run in the same harness
- Final test protocol: 2025-01-01..2025-06-30 is a locked holdout, evaluated exactly once at release via the trainers' `--final-test` flag. Disclosure: before the lock was declared (2026-07-10), earlier training runs printed metrics over this span, so it is not perfectly blind
- **Final test results (run once, 2026-07-10):** LightGBM point MAE 11.75 (77.3% within 15 min, R² 0.37), XGBoost MAE 11.91; quantile 80% interval covers 73.4% raw and **79.3% conformalized** at 32.9-minute width — the calibration fit on the 2024 split held on the untouched 2025 window. Final-test MAE runs ~0.7 min softer than the walk-forward mean, ordinary for a fresh half-year. Any model or method change after this evaluation re-opens the test cycle
- Primary metrics: the decision metric (bad-delay-day PR-AUC) and calibrated interval coverage below are the headline; MAE is secondary given the predictability ceiling. Point accuracy (walk-forward means, minutes):

| Model | MAE | Hit rate (within 15 min) |
|---|---|---|
| LightGBM quantile (q50) | 10.80 | 79.7% |
| LightGBM point | 11.05 | 77.9% |
| XGBoost | 11.16 | 77.6% |
| Climatology quantile (baseline) | 13.07 | 73.8% |
| Moving average, 7-day (baseline) | 13.53 | 70.0% |
| Naive (baseline) | 15.09 | 67.7% |

- Subgroup metrics: severe-weather days (worst-airport severity >= 3, 32% of test rows) run MAE 15.48 vs 11.05 overall for the full feature set. Conditional coverage after conformal calibration: severe-weather days reach 79.7% (from 76.3% raw), i.e. the marginal fix generalizes to the hard subgroup; per-route coverage narrows from a raw floor of 67.0% to a calibrated spread of 72.6% (LAS-LAX) to 87.2% (HNL-KOA) — the residual per-route gap that Mondrian conformal would target. Both slices published in `outputs/ablation.json`
- Decision metric: a P(mean delay > 15 min) classifier (`eval_lightgbm_classifier` in `src/training/walk_forward.py`) flags bad-delay days (22.6% base rate) with PR-AUC 0.61 (~2.7x base rate), ROC-AUC 0.82, Brier 0.13 — the metric that maps to an asymmetric-cost decision, reported alongside MAE
- Calibration / uncertainty checks: raw 80% quantile interval covers 74.99% at 29.2-minute mean width; conformalized quantile regression (`src/evaluation/conformal.py`) restores it to 79.64% (per fold 77.3-80.9) at 31.7-minute width — a distribution-free finite-sample guarantee (marginal, approximate on temporal data). The conformal offset is fit on a dedicated calibration slice (the later half of each fold's validation window) that early stopping never touches, so no data is reused between model selection and calibration. Per-quantile calibration: actuals fall at or below q10 12.1% (nominal 10) and below q90 87.1% (nominal 90). Reliability shown in `images/calibration.png`
- Known calibration limits, stated rather than hidden: split conformal assumes exchangeability, which temporal data violates, so the guarantee is approximate and marginal — it holds on average, not for every subgroup or regime. The techniques that would address the residual gaps are adaptive conformal inference (ACI) for drift over time and Mondrian (group-conditional) conformal for the per-route and severe-weather under-coverage; both are deliberate non-goals here, documented so the boundary of the claim is explicit
- Recursive forward evaluation: under serving conditions (all delay lags past the origin come from the model's own rolled predictions, `src/forecasting/recursive_eval.py`), pooled MAE decays from 10.7 at depth 1 to ~12.1 by depth 14 and stays below the 15.1 naive baseline through depth 90 — covering the deepest recursion live serving can reach across the monthly publication cycle — while raw interval coverage degrades from 74% to a floor near 62%; per-horizon conformal offsets (fit on folds 1-3, validated on fold 4) hold coverage at an 82.2% average across depths, per-depth range 78.5-85.5. The full-depth feature-parity test in `tests/test_recursive.py` is the correctness guard. Curves in `outputs/recursive_eval.json` and `images/recursive_degradation.png`
- Robustness / adversarial checks: month-matched PSI drift gates on the training data (`src/data_checks.py`, `src/monitoring/psi.py`); upstream-source pinning (`era5`, `gfs_seamless`); leakage tests with post-cutoff spike injection (`tests/test_leakage.py`)

## Limitations and risks

- Failure modes: error concentrates on severe-weather days (predictability-ceiling analysis: ~63% of squared error on ~32% of rows); intervals under-cover on a handful of routes; delay-lag features are recursively rolled forward (not actuals) at live scoring time
- Assumptions: frozen route set; BTS publishes with a 1-2 month lag; weather sources stay pinned
- Known weak cohorts: severe-weather days, worst-covered routes listed above
- Human oversight requirements: the data-quality and drift gates halt the build for human review; nothing retrains or republishes automatically

## Deployment

- Serving: a daily GitHub Actions run (`.github/workflows/daily-forecast.yml`) assembles live weather (`src/pipelines/live_weather.py` — era5 archive for settled days, the live forecast API with `gfs_seamless` pinned for the seam and the future), rolls the recursive engine from the last published BTS actual through today+7, and publishes `live_forecasts.json` to the public dashboard. Any uncovered airport-day aborts the run with no output: the dashboard keeps the previous honest file and shows a staleness banner
- Serving generation: trained through 2025-12-31 on the extended table (2019-2026), early-stopped and conformal-calibrated on disjoint halves of 2026-01..05 (base offset 1.203, shipped as `conformal.json`). The applied per-horizon widening comes from the recursion backtest's fold-fit offsets — a protocol-level estimate from different model instances — and the daily job verifies at startup that the released artifacts and the repo feature state belong to the same generation, that those offsets dominate the serving generation's own base offset at every depth (1.47-4.57 vs 1.20, a necessary consistency floor against its one-step calibration), and that the run's recursion depth stays within the validated 90. This generation's blind test is the live public record — the append-only prediction log can be graded as BTS actuals land
- Refresh cadence: monthly BTS backfill and validation-gated retrain via a manually dispatched workflow, never automatic; month-matched PSI drift gates (`src/monitoring/psi.py`, thresholds 0.1 moderate / 0.25 major) alert and halt, never auto-retrain — the 2024-12 wind seam (`docs/incidents/`) is the standing example of input drift that retraining would have baked in
- Artifacts: native LightGBM boosters (`lightgbm_q{10,50,90}.txt`, `lightgbm_point.txt`) plus `conformal.json`, shipped together as the `models-latest` GitHub Release the daily job downloads; the boosters and the offset ship together or not at all
