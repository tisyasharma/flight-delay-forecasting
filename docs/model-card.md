# Model card: route-level delay forecaster

## Model summary

- Model name: LightGBM quantile triple (`lightgbm_q10/q50/q90.txt`), the only serving path, and a LightGBM point model (`lightgbm_point.txt` / `lightgbm_delay.pkl`) kept as the accuracy comparator against XGBoost
- Version: two generations exist and carry distinct names here. The 2024-01 evaluated generation (training cutoff 2024-01-01, clean HPO window, per-fold protocol) produced every evaluation number in this card. The 2026-01 serving generation (trained through 2025-12-31) runs the live loop, and its own test is the public prediction log
- Owner: Tisya Sharma
- Date: 2026-07-16
- Task: daily mean arrival-delay forecasting per route (point and 10/50/90 quantiles), 50 busiest US directional routes, rolled forward recursively through 7 days past the run date. At BTS's monthly publication cadence the recursion depth cycles, from roughly 45 days just after a month of actuals lands to about 80 the day before the next month publishes. The backtest validates every depth through 90, and the daily job refuses to serve deeper rather than extrapolate the widening
- Context of use: open evaluation and forecasting study. Research-side comparators (XGBoost, LSTM, TCN) are documented but not part of the forecasting path

## Intended uses

- Supported uses: exploring daily route-average delay risk on the frozen 50-route set. Also a methodology reference for walk-forward evaluation, conformal calibration, and recursive forward forecasting of route-level series
- Unsupported / out-of-scope uses: per-flight delay predictions; operational, booking, or crew decisions; routes or airports outside the frozen set; horizons beyond 7 days past the run date (recursion depths beyond the backtest's validated 90 days, which the daily job refuses at runtime); any safety-relevant application

## Training data

- Datasets used: `data/processed/features.csv` (not tracked in git, rebuilt by `make features` from the tracked inputs), see the [dataset card](dataset-card.md) for sources, rights, fill semantics, and known gaps
- Sampling window: 2019-01-01 to 2026-05-31. The serving generation trains to a 2026-01-01 cutoff with 2026-01..05 as its early-stopping and calibration window, while the evaluated generation used a 2024-01-01 cutoff (its 2024-H2 devtest and 2025-H1 locked-test spans predate the new cutoff, so the trainers skip them for the serving generation rather than report in-sample numbers)
- Labeling process: not applicable, targets are BTS-reported actual arrival delays
- Sensitive attributes involved: none
- Known gaps or biases: AK/HI aviation-variable coverage gap, COVID regime, lifted-index exclusion, resolved 2024-12 wind seam (details in the dataset card)

## Training procedure

- Preprocessing: scripted pipeline (`src/process.py`, `src/build_features.py`), train-window-gated statistics serialized to `data/processed/feature_state.json`, rebuilds gated by `src/data_checks.py`
- Features: 80 tabular features from `src/features/registry.py` (calendar, COVID flags, route stats, delay lags/rolling, daily/hourly weather, aviation weather, hub inbound-delay state)
- Hyperparameters: `configs/best_params_lightgbm.json` / `configs/best_params_xgboost.json`, selected by 50-trial Optuna TPE (seeded) minimizing validation MAE on 2022-07-01..2023-01-01, fold 0's validation slice, chosen because every later half-year is some walk-forward fold's test window. The tradeoff: parameters are selected on an older, smaller-train regime. Best search values: LightGBM 10.91, XGBoost 10.98 val MAE
- Compute / framework: LightGBM 4.6.0, XGBoost 3.3.0, Optuna 4.9.0, pandas 2.3.3, numpy 2.5.1 on Python 3.12, with library versions pinned in `constraints.txt`. Experiment records live in a local MLflow store with git SHA, data fingerprint, and library-version tags per run
- Random seed strategy: numpy seed 42, model `random_state` 42, Optuna `TPESampler(seed=42)`

## Evaluation

- Validation protocol: 4 half-year walk-forward folds (2023-01..2024-12), feature statistics rebuilt at each fold's own train_end (`src/training/fold_features.py`). Baselines run in the same harness
- Final test protocol: 2025-01-01..2025-06-30 is a locked holdout, evaluated exactly once at release via the trainers' `--final-test` flag. Disclosure: before the lock was declared (2026-07-10), earlier training runs printed metrics over this span. No modeling choice was knowingly based on those numbers, but influence cannot be fully ruled out, which is why this card calls the span imperfectly blind rather than blind
- Final test results (run once, 2026-07-10, both artifact sets scored in the same run): LightGBM point MAE 11.75 (77.3% within 15 min, R² 0.37), XGBoost MAE 11.91. The quantile 80% interval covers 73.4% raw and 79.3% conformalized at 32.9-minute width, so the calibration fit on the 2024 split held on the untouched 2025 window. Final-test MAE runs about 0.7 min softer than the walk-forward mean, ordinary for a fresh half-year. Any model or method change after this evaluation re-opens the test cycle
- Primary metrics: the decision metric (bad-delay-day PR-AUC) and calibrated interval coverage below are the headline. MAE is secondary given the predictability ceiling. Point accuracy (secondary context, walk-forward means, minutes):

| Model | MAE | Hit rate (within 15 min) |
|---|---|---|
| LightGBM quantile (q50) | 10.80 | 79.7% |
| LightGBM point | 11.05 | 77.9% |
| XGBoost | 11.16 | 77.6% |
| Climatology quantile (baseline) | 13.07 | 73.8% |
| Moving average, 7-day (baseline) | 13.55 | 69.8% |
| Naive (baseline) | 15.12 | 67.7% |

- Subgroup metrics: severe-weather days (worst-airport severity >= 3, 32% of test rows) run MAE 15.48 vs 11.05 overall for the full feature set. Conditional coverage after conformal calibration: severe-weather days reach 79.7% (from 76.3% raw), so the marginal fix generalizes to the hard subgroup. Per-route coverage narrows from a raw floor of 67.0% to a calibrated spread of 72.6% (LAS-LAX) to 87.2% (HNL-KOA), the residual per-route gap that Mondrian conformal would target. Both slices published in `outputs/ablation.json`
- Decision metric: a P(mean delay > 15 min) classifier (`eval_lightgbm_classifier` in `src/training/walk_forward.py`) flags bad-delay days (22.6% base rate) with PR-AUC 0.61 (about 2.7x base rate), ROC-AUC 0.82, Brier 0.13, the metric that maps to an asymmetric-cost decision, reported alongside MAE
- Calibration / uncertainty checks: the raw 80% quantile interval covers 74.99% at 29.2-minute mean width. Conformalized quantile regression (Romano et al. 2019, `src/evaluation/conformal.py`) restores it to 79.64% (per fold 77.3-80.9) at 31.7-minute width, a distribution-free finite-sample guarantee (marginal, approximate on temporal data). The conformal offset is fit on a dedicated calibration slice (the later half of each fold's validation window) that early stopping never touches, so no data is reused between model selection and calibration. Per-quantile calibration: actuals fall at or below q10 12.1% (nominal 10) and below q90 87.1% (nominal 90). Reliability shown in `images/calibration.png`
- Interval score: the weighted interval score (Bracher et al. 2021, the epidemic Forecast Hub standard) averages 7.22 minutes across folds on the raw q10/50/90 set, against 8.87 for the climatology quantile baseline and 15.12 for the naive baseline (WIS generalizes absolute error, so a point forecast scores its MAE). With three quantile levels this is the K=1 form of the score, one central interval plus the median, computed as twice the mean pinball loss. It scores the raw quantiles as issued; the conformal widening is a coverage intervention applied on top
- Known calibration limits: split conformal assumes exchangeability, which temporal data violates, so the guarantee is approximate and marginal. It holds on average, not for every subgroup or regime. The techniques that would address the residual gaps are adaptive conformal inference (ACI) for drift over time and Mondrian (group-conditional) conformal for the per-route and severe-weather under-coverage. Both are deliberate non-goals here, documented so the boundary of the claim is explicit
- Recursive forward evaluation: under serving conditions (all delay lags past the origin come from the model's own rolled predictions, `src/forecasting/recursive_eval.py`), pooled MAE decays from 10.7 at depth 1 to about 12.1 by depth 14 and stays below the persistence baseline (the last published actual carried forward, which decays from 15.1 to 22) through depth 90, covering the deepest recursion live serving can reach across the monthly publication cycle. Raw interval coverage degrades from 74% to a floor near 62% over the same depths. Per-horizon conformal offsets (fit on folds 1-3, validated on fold 4) hold coverage at an 82.2% average across depths, per-depth range 78.5-85.5. The full-depth feature-parity test in `tests/test_recursive.py` is the correctness guard. Curves in `outputs/recursive_eval.json` and `images/recursive_degradation.png`
- Drift and adversarial checks: month-matched PSI drift gates on the training data (`src/data_checks.py`, `src/monitoring/psi.py`); upstream-source pinning (`era5`, `gfs_seamless`); leakage tests with post-cutoff spike injection (`tests/test_leakage.py`)

## Limitations and risks

- Failure modes: error concentrates on severe-weather days (predictability-ceiling analysis: about 63% of squared error on about 32% of rows), intervals under-cover on a handful of routes, and delay-lag features are recursively rolled forward (not actuals) at live scoring time
- Assumptions: a frozen route set, a 1-2 month BTS publication lag, and pinned weather sources
- Known weak cohorts: severe-weather days, worst-covered routes listed above
- Human oversight requirements: the data-quality and drift gates halt the build for human review. Nothing retrains or republishes automatically

## Deployment

- Serving: the daily GitHub Actions job (`.github/workflows/daily-forecast.yml`) assembles live weather (`src/pipelines/live_weather.py`: era5 archive for settled days, the live forecast API with `gfs_seamless` pinned for the seam and the future), rolls the recursive engine from the last published BTS actual through today+7, and publishes `live_forecasts.json` to the dashboard
- Failure behavior: any uncovered airport-day aborts the run with no output. The dashboard keeps the previous file and shows a staleness banner
- Verification overlay: the payload also carries a replay of the last published month from the prior month's end with the same engine and offsets, drawn against the published actuals on the dashboard. The replay rolls its delay features from the model's own predictions exactly as live serving does, but its weather comes from the settled archive, information a true month-end run would not have had for most of the month, so it isolates the cost of recursive feedback and reads as a favorable bound rather than a reproduction of serving conditions. It is retrospective by construction and never enters the append-only prediction log, which remains the only record graded as the live track record
- Serving generation: trained through 2025-12-31 on the extended table (2019-2026), early-stopped and conformal-calibrated on disjoint halves of 2026-01..05 (base offset 1.203, shipped as `conformal.json`). The applied per-horizon widening comes from the recursion backtest's fold-fit offsets, a protocol-level estimate from different model instances. This generation's blind test is the live public record: the append-only prediction log can be graded as BTS actuals land
- Startup checks: the daily job verifies that the released artifacts and the repo feature state belong to the same generation, that the per-horizon offsets dominate the serving generation's own base offset at every depth (1.47-4.57 vs 1.20, a necessary consistency floor against its one-step calibration), and that the run's recursion depth stays within the validated 90
- Prediction vintages: each run appends its published horizon to `data/live/predictions/` before any outcome is known, and past entries are never revised. The same as-of discipline is what the St. Louis Fed's ALFRED archive and CMU Delphi's epidata apply to revised ground truth. This project applies it to its own forecasts so they can be scored against actuals exactly as issued
- Retraining policy and refresh cadence: a manual monthly procedure, never automatic. Each BTS refresh runs the data-quality gates, recalibrates the conformal offset, and performs a periodic expanding-window refit only if the gates pass. The month-matched PSI drift gate (`src/monitoring/psi.py` computes the metric, `src/data_checks.py` halts the build past 0.25) alerts and halts rather than triggering retraining, and the 2024-12 wind seam (`docs/incidents/`) is the standing example of input drift that retraining would have baked in. The offset recalibrates more often than the models because probabilistic components go stale faster than point models, roughly half the point model's optimal retraining window in Zanotti's 2025 retail demand-forecasting study of global models, gradient-boosted and deep ([arXiv:2505.00356](https://arxiv.org/abs/2505.00356), directional evidence from a different domain)
- Artifacts: native LightGBM boosters (`lightgbm_q{10,50,90}.txt`, `lightgbm_point.txt`) plus `conformal.json`, shipped together as the `models-latest` GitHub Release the daily job downloads. The boosters and the offset ship together or not at all. Release tags act as the model registry: each generation is an immutable, downloadable artifact set with its training metadata

## Design decisions and rejected tools

- No LLM features: there is no free text in this problem for a language model to read, and nondeterministic components sitting next to drift gates would make the gates harder to trust
- No request-time inference API: the product is a daily batch forecast, scenario exploration is precomputed at publish time, and a per-request service would add an always-on surface with no consumer
- MLflow stays local: runs are tracked in a local store with git SHA and data fingerprints. A hosted tracking server adds operational surface without adding evidence for a single-model project
- No DVC or feature store: the feature table rebuilds deterministically from scripted sources with train-window-gated state, and the data checks gate every rebuild
- No Airflow, Dagster, or Kubernetes: one scheduled job and one manual monthly procedure do not need an orchestrator. GitHub Actions cron plus `make refresh` cover the actual cadence
- Deep models not pursued: LSTM and TCN comparators ran under the earlier protocol (12.69 / 12.79 MAE, documented for scale), and gradient boosting won as expected at this dataset size (Grinsztajn et al. 2022, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815))
