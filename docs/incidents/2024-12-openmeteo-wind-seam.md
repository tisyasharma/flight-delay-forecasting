# Data-quality incident: wind speed discontinuity from an unpinned weather model

**Window affected:** training rows dated December 2024 through June 2025
**Detected:** July 2026, during a distribution audit of the training data
**Status:** resolved (model pinned, data refetched, models retrained)

## Symptom

A month-matched population stability index (PSI) sweep over the weather
features flagged `max_wind` at 0.78 against its 2019-2024 reference, far past
the 0.25 major-drift threshold, while every other weather feature scored
below 0.05. Wind gusts stayed flat over the same window, which ruled out an
actual change in the weather.

The 19-month gap between the seam and its detection is itself part of the
record: no drift monitoring existed before this audit, which was being built
as the project's data-quality gates. This incident is why the month-matched
PSI check now runs on every rebuild instead of on demand.

## Diagnosis

Monthly cross-airport median wind speeds broke abruptly at December 2024:
each month from 2024-12 through 2025-06 ran 13-23% below the median of the
same calendar month in 2019-2023, with no corresponding shift in gusts.
Re-fetching individual days from the Open-Meteo archive reproduced the stored
values exactly on both sides of the break, so the discontinuity was in the
upstream series itself, not in this repo's processing.

The fetch script requested the archive without a `models=` parameter, which
serves Open-Meteo's `best_match` composite. That composite changed its
underlying reanalysis source around December 2024, and the replacement model
reports systematically lower 10 m wind speeds. Requesting `models=era5`
explicitly returned a temporally consistent series across the same dates.

## Root cause

The archive fetch relied on the API's default model selection, which is not
guaranteed to be stable over time. Seven months of training rows silently
mixed two different wind climatologies.

## Fix

- Pinned `models=era5` in the archive fetch and refetched the full
  2019-2025 range for all 23 airports, so the entire series comes from one
  model rather than splicing the old composite with era5 at the seam.
- Rebuilt the feature table and retrained every model on the consistent
  series.
- After the rebuild, the worst monthly deviation from same-month history in
  the affected window is 10% with no systematic direction, and the
  month-matched PSI on `max_wind` is 0.01.
- Audited every other Open-Meteo call site for the same anti-pattern. The
  aviation fetch and the live forecast fetch pin `models=` explicitly
  (`gfs_seamless`), so the class of bug is closed, not just this instance.
- No serving model was live during the affected window, so the impact was
  confined to offline experiment results, which were regenerated after the
  rebuild. Rows added after the pin (2025-07 onward) were fetched as era5 from
  the start, which is why the affected window ends at June 2025.

## Lessons

- Pin every upstream model or dataset version explicitly. Defaults that
  resolve to "best available" can change underneath a pipeline without any
  error or header hinting at it.
- Month-matched references are essential for drift checks on seasonal
  features: against a naive full-history reference, temperature alone scores
  PSI 0.30 from seasonality, which would have buried this real signal in
  false alarms.
- Input drift is not a retraining trigger. Retraining on the contaminated
  window would have baked the seam into the models; the correct response was
  to fix the source. This incident is the reference case for why the
  monitoring design alerts on input drift but never auto-retrains.
