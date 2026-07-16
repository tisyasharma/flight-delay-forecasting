# Dataset card: route-day flight delay table

## Overview

- Dataset name: `data/processed/features.csv` (with `data/processed/feature_state.json` as its serialized train-time state). The table is not tracked in git, `make features` rebuilds it deterministically from the tracked inputs (`daily_route_demand.csv` and the weather tables)
- Version: rebuilt 2026-07 on era5-pinned archive weather plus GFS aviation variables
- Owner / steward: Tisya Sharma
- Purpose: training and evaluation table for daily route-level US flight arrival-delay forecasting (point and quantile models)

## Provenance

- Sources:
  - BTS On-Time Performance (Reporting Carrier On-Time Performance table, TranStats): monthly CSV extracts, one file per month under `data/raw/YYYY-MM.csv` (gitignored), columns restricted to `FL_DATE`, `REPORTING_AIRLINE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DEP_DELAY`, `ARR_DELAY`, `CANCELLED`, `DISTANCE` (`REQUIRED_COLS` in `src/process.py`). US government data, public domain.
  - Open-Meteo historical weather archive, `models=era5` pinned (`src/training/fetch_weather_data.py`): hourly temperature, precipitation, snowfall, wind speed, wind gusts, and weather code for the 23 airports, aggregated to daily and operating-hour metrics (`data/weather/weather_daily.csv`, `data/weather/weather_hourly_agg.csv`). CC BY 4.0.
  - Open-Meteo historical-forecast API, `models=gfs_seamless` pinned (`src/weather/gfs_history.py`): hourly visibility, CAPE, lifted index, low cloud cover, weather code, gusts, wind speed, and freezing-level height, aggregated per airport-day (`data/weather/aviation_daily.csv`). Pinning matches the live forecast API that serves the same variables, so training happens on archived forecasts rather than reanalysis and a live consumer would see a consistent distribution. CC BY 4.0.
- Derived state (not an external source): the hub inbound-delay features (`hub_inbound_lag_1`, `hub_inbound_roll_7`) are computed inside the feature pipeline (`add_hub_features` in `src/build_features.py`) as the prior-day and trailing-7-day average arrival delay over the *modeled routes* into each destination airport. Modeled-routes-only by design, so a live pipeline can rebuild it from rolled-forward predictions with no extra data source. A separate all-inbound aggregate (`data/processed/hub_inbound_daily.csv`, every BTS arrival per destination) appears only as a backtest-only upper-bound variant in the ablation, never in the served features.
- Collection period: 2019-01-01 through 2026-05-31 (extended as BTS publishes, by a manual monthly refresh)
- Rights / permissions: BTS data is public domain. Open-Meteo data is CC BY 4.0 (attribution in the README). Both permit redistribution of the derived tables tracked in this repo.
- Included / excluded populations: the 50 busiest directional US routes by total flights (frozen set, encoded in `feature_state.json`). All other routes and airports are out of scope. Cancelled flights contribute to cancellation counts but not to delay averages.

## Structure

- Rows / entities: 135,400 rows = 50 routes x 2,708 days, one row per route per calendar day
- Features: 119 columns. The 80 model features are defined in `src/features/registry.py` (calendar, COVID period flags, route statistics, delay lags/rolling stats, daily and hourly weather, aviation weather, hub inbound-delay state)
- Target label: `avg_arr_delay` (mean arrival delay in minutes over the route's flights that day, from BTS `ARR_DELAY`)
- Weather severity scale: WMO condition codes map to a 0-5 score (clear 0 to thunderstorm 5, `src/weather/common.py`). A severe-weather day in the model card means the route's worse airport scored 3 or higher
- Missingness / fill semantics (all train-window gated, serialized in `feature_state.json`):
  - calendar gaps are completed to one row per route-day, target forward-filled up to 7 days then zero (`fill_missing_dates` in `src/process.py`), the convention the models were trained on. 40 route-days (0.03% of the table) carry such a filled label, and they stay in training and evaluation, a share too small to move any reported number
  - route statistics and lag-fill medians come from the training window only
  - temperature gaps take train-window medians. Visibility takes the train-window median (zero would mean fog, the opposite of missing). Precipitation, rain, and snowfall fill to zero at processing time, a null weather code reads as clear, and the convective, gust, cloud, and freezing-rain columns default to the same benign zero in the feature builder
- Known duplicates or joins: no duplicate route-day keys can survive, the calendar-shape gate in `src/data_checks.py` enforces exactly one row per route per day. Weather joins on airport-date (origin as `apt1_`, destination as `apt2_`), and hub state joins on destination airport-date, lagged one day

## Labeling

Not applicable in the annotation sense: targets are BTS-reported actual arrival delays, not human labels. The label pipeline is `src/process.py` (clean, aggregate to route-day, calendar-fill, weather merge).

## Risk and sensitivity

- Personal or sensitive data: none. All inputs are aggregate operational and meteorological data
- Protected attributes: none
- Known representation gaps and biases:
  - GFS aviation variables have a coverage gap for the Alaska/Hawaii airports in the early years of the window. Affected rows are handled by the aviation fill rules above
  - Lifted index is fetched but excluded from the model features: its month-matched PSI averages 0.175 but breaches 0.42-0.44 in February/October with a monotone winter drift across GFS eras, so it does not represent one consistent physical series (CAPE covers convection instead)
  - COVID regime: 2020-2022 traffic collapse and recovery are flagged with period indicator features rather than excluded
  - Wind seam (resolved): rows from 2024-12 through 2025-06 originally ingested an upstream model change in Open-Meteo's `best_match` composite. Fixed by pinning `models=era5` and refetching. See `docs/incidents/2024-12-openmeteo-wind-seam.md`
- Foreseeable misuse: treating route-day average delays as per-flight predictions, or using this table for routes/periods outside its coverage

## Split manifest

Every end date is an exclusive bound quoted verbatim from `src/config.py`
(`date < end`), so the last included day is the day before the date shown. The
fold rows end on month-end dates, which means each fold's last scored day is
the 29th or 30th, an artifact of the original fold constants kept because the
published numbers were computed on them. The serving rows end on the first of
the next period and tile exactly.

| Slice | Window | Use |
|---|---|---|
| HPO train | 2019-01-01 to 2022-06-30 | hyperparameter search fitting |
| HPO validation | 2022-07-01 to 2023-01-01 | hyperparameter selection (fold 0's validation slice, no fold's test window) |
| Fold 1 test | 2023-01-01 to 2023-06-30 | walk-forward evaluation (train_end 2022-07-01) |
| Fold 2 test | 2023-07-01 to 2023-12-31 | walk-forward evaluation (train_end 2023-01-01) |
| Fold 3 test | 2024-01-01 to 2024-06-30 | walk-forward evaluation (train_end 2023-07-01) |
| Fold 4 test | 2024-07-01 to 2024-12-31 | walk-forward evaluation (train_end 2024-01-01) |
| Dev test | 2024-07-01 to 2025-01-01 | production-model spot checks (fold 4's span plus its final day) |
| Locked final test | 2025-01-01 to 2025-07-01 | consumed: evaluated once at release, 2026-07-10 (`--final-test`, results in the model card) |
| Serving train | 2019-01-01 to 2026-01-01 | serving-generation training (`feature_state.json` train_end 2026-01-01). Includes the consumed locked-test span, so the trainers skip the historical holdouts for this generation |
| Serving early stopping | 2026-01-01 to 2026-03-17 | serving-generation early stopping (early half of its validation window) |
| Serving calibration | 2026-03-17 to 2026-06-01 | serving-generation conformal offset (`conformal.json`), untouched by early stopping |

Walk-forward feature statistics are rebuilt at each fold's own `train_end`
(`src/training/fold_features.py`). Every rebuild runs the data-quality gates in
`src/data_checks.py` (route set, calendar shape, completeness, holiday
coverage, month-matched wind-seam PSI, weather-table continuity).

## Historical extraction recipe (BTS)

Download the Reporting Carrier On-Time Performance table from BTS TranStats month by month, selecting the `REQUIRED_COLS` fields above, and save each month as `data/raw/YYYY-MM.csv`. `src/process.py` filters to the frozen 50-route set and rebuilds `daily_route_demand.csv`. `make features` then rebuilds `features.csv` plus `feature_state.json` and runs the data checks. The monthly ritual is the same recipe condensed to one command: save the new month's extract to `data/raw/`, then run `make refresh`, which extends the weather tables to the latest settled archive day, rebuilds the processed and feature tables, and runs the data-quality gates.

## Recommended use

- Suitable tasks: daily route-level delay point and quantile forecasting, plus drift and data-quality research on the included window
- Unsuitable tasks: per-flight prediction, operational decision-making, routes or dates outside the frozen set/window
- Maintenance / update cadence: extended by a manual monthly BTS refresh (`make refresh`) as new months publish. Weather sources stay pinned (`era5`, `gfs_seamless`) so refetches are reproducible
