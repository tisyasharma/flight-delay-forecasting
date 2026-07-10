# Model Card: Route-Level Delay Forecaster

## Model summary

- Model name: LightGBM point model (`lightgbm_point.txt` / `lightgbm_delay.pkl`) and LightGBM quantile triple (`lightgbm_q10/q50/q90.txt`)
- Version: 2026-07 retune (clean HPO window, per-fold evaluation protocol)
- Owner: Tisya Sharma
- Date: 2026-07-10
- Task: daily mean arrival-delay forecasting per route (point and 10/50/90 quantiles), 50 busiest US directional routes
- Context of use: public portfolio forecasting system; research-side comparators (XGBoost, LSTM, TCN) are documented but not served

## Intended uses

- Supported uses: exploring daily route-average delay risk on the frozen 50-route set; the system's own public live track record; methodology reference for walk-forward evaluation of route-level forecasts
- Unsupported / out-of-scope uses: per-flight delay predictions; operational, booking, or crew decisions; routes or airports outside the frozen set; horizons beyond the published 7-day live window; any safety-relevant application

## Training data

- Datasets used: `data/processed/features.csv` — see the [dataset card](dataset-card.md) for sources, rights, fill semantics, and known gaps
- Sampling window: 2019-01-01 to 2025-06-30 (production training cutoff 2024-01-01 for the current artifacts)
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
- Final test protocol: 2025-01-01..2025-06-30 is a locked holdout, evaluated exactly once at release via the trainers' `--final-test` flag. Disclosure: before the lock was declared (2026-07-10), earlier training runs printed metrics over this span, so it is not perfectly blind; the live pipeline is the fully blind ongoing test after release
- Primary metrics: the decision metric (bad-delay-day PR-AUC) and calibrated interval coverage below are the headline; MAE is secondary given the predictability ceiling. Point accuracy (walk-forward means, minutes):

| Model | MAE | Hit rate (within 15 min) |
|---|---|---|
| LightGBM quantile (q50) | 10.80 | 79.7% |
| LightGBM point | 11.05 | 77.9% |
| XGBoost | 11.16 | 77.6% |
| Climatology quantile (baseline) | 13.07 | 73.8% |
| Moving average, 7-day (baseline) | 13.53 | 70.0% |
| Naive (baseline) | 15.09 | 67.7% |

- Subgroup metrics: severe-weather days (worst-airport severity >= 3, 32% of test rows) run MAE 15.48 vs 11.05 overall for the full feature set; per-route 80% coverage ranges from ~67% (LIH-HNL, LAS-LAX) up to roughly nominal (best route ORD-LGA at 80.5%), i.e. the whole spread sits at or below the 80% target — both slices published in `outputs/ablation.json`
- Decision metric: a P(mean delay > 15 min) classifier (`eval_lightgbm_classifier` in `src/training/walk_forward.py`) flags bad-delay days (22.6% base rate) with PR-AUC 0.61 (~2.7x base rate), ROC-AUC 0.82, Brier 0.13 — the metric that maps to an asymmetric-cost decision, reported alongside MAE
- Calibration / uncertainty checks: raw 80% quantile interval covers 74.97% at 29.2-minute mean width; conformalized quantile regression (`src/evaluation/conformal.py`, split-conformal offset fit on the val window) restores it to 79.68% (per fold 78.7-80.5) at 31.6-minute width — a distribution-free finite-sample guarantee (marginal, approximate on temporal data). Per-quantile calibration: actuals fall at or below q10 12.1% (nominal 10) and below q90 87.1% (nominal 90). Reliability shown in `images/calibration.png`
- Robustness / adversarial checks: month-matched PSI drift gates on the training data (`src/data_checks.py`, `src/monitoring/psi.py`); upstream-source pinning (`era5`, `gfs_seamless`); leakage tests with post-cutoff spike injection (`tests/test_leakage.py`)

## Limitations and risks

- Failure modes: error concentrates on severe-weather days (predictability-ceiling analysis: ~63% of squared error on ~32% of rows); intervals under-cover on a handful of routes; delay-lag features are recursively rolled forward (not actuals) at live scoring time
- Assumptions: frozen route set; BTS publishes with a 1-2 month lag; weather sources stay pinned
- Known weak cohorts: severe-weather days, worst-covered routes listed above
- Human oversight requirements: none in the loop; drift events raise a public banner and never auto-retrain

## Deployment

- Serving environment (planned): AWS Lambda container serving the native LightGBM boosters; daily batch forecasts via GitHub Actions
- Monitoring thresholds: month-matched PSI 0.1 moderate / 0.25 major, 7 consecutive days over 0.25 logs a drift event; monthly true-MAE and coverage backfill once BTS actuals land
- Retraining triggers: scheduled monthly retrain with a validation non-regression gate; never drift-triggered
- Rollback plan: deployment gated on the validation check, failure keeps the previous model release serving
