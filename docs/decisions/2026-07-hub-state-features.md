# Decision: keep the hub_state feature group

**Date:** 2026-07-10
**Status:** accepted. Revisit outcome recorded 2026-07-16, next trigger at
the first monthly refresh retrain

## Context

The hub inbound-delay features (`hub_inbound_lag_1`, `hub_inbound_roll_7`)
add little backtest accuracy. The serving feature contract freezes when the
recursive engine lands, so keep them or cut them now.

## Evidence

Feature-set ablation over the four walk-forward folds, per-fold feature
state, shared tuned params (`outputs/ablation.json`, 2026-07-10):

| set | features | mae | severe_mae | coverage_80 |
|---|---|---|---|---|
| v1_plus_aviation | 78 | 11.11 | 15.57 | 74.94 |
| full (with hub_state) | 80 | 11.05 | 15.48 | 74.97 |
| full_allinbound_hub | 80 | 11.02 | 15.45 | 74.77 |

The gain is 0.06 MAE overall and 0.09 on severe days, inside fold noise
(std ~0.8), but the sign is consistent across all four folds, and the
all-inbound variant (backtest-only upper bound: it uses every BTS arrival,
which a live pipeline cannot observe) shows a little more headroom in the
same direction. The ablation harness aggregates slightly differently from the
headline walk-forward run, which is why coverage reads 74.97 here against the
74.99 in the model card.

## Decision

Keep hub_state. The expected payoff is at live horizons, not in this
backtest: under recursion the route's own delay lags degrade fastest, while
the hub term aggregates 50 routes' worth of signal per airport and is
rolled forward the same way. The cost is two features that ride an existing
join. An earlier ablation on the global-cutoff protocol showed a smaller
gain (0.04 MAE, flat severe), so this keep is a judgment call on live-horizon
value, not a claim of proven backtest lift.

## Revisit trigger

At the recursive engine's backtest (`recursive_eval`), compare MAE(k) and
coverage(k) with and without the hub features. Cut them before the contract
freezes if the recursion-fed gain stays inside noise at depth k, or if
serve-time PSI on the hub features ever breaches 0.25.

## Revisit outcome (2026-07-16)

The serving contract froze with the hub features in the 80-feature set before
the with/without-hub recursion comparison ran, so that comparison is recorded
here as open debt rather than completed analysis. The engine's full-depth
parity test covers the hub features' correctness under recursion, not their
marginal value. Serve-time PSI on hub features was not implemented, the drift
gates cover the input weather series instead. The sequencing broke because the
launch retrain and the contract freeze shipped together, ahead of the
comparison this trigger assumed would come first. The new trigger is tied to a
concrete step rather than a phase: the first monthly refresh retrain runs the
ablation on the extended table before a new generation ships.
