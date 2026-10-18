# Fleet forecast v2: preregistration

This file must be committed before executing the v2 simulation or computing its
count results. Starting repository commit:
`f17199a156f7a0d8ec0f642cfb023ff432a05715`. Input-schema, record-extent and existing
E6 calibration reports were inspected to design this study; no v2 count results
were computed. Do not tune this plan after seeing results. Preserve the frozen
baseline, all E0–E9 artifacts, and `reports/fleet_forecast/` byte for byte.

## Question and scope

Does summing held-out per-cell forecasts improve 30/60/90-scenario-day incident
replacement counts in larger cohorts with staggered commissioning? Evaluate
within-dataset leave-one-cell-out (LOCO) and leave-one-dataset-out (LODO)
separately. Replacement is the first interpolated **70% independently measured
capacity** crossing, not a recorded field replacement. No Baaz data are present.

These are resampled fixed installed cohorts, not continuously replenished active
fleets: known-already-below copies are excluded from future incident counts at
each today and reported. Nominal cohort size must never be substituted for the
smaller at-risk or forecast-covered population. This design changes age mix and
aggregation size; it does not simulate maintenance policy, pack interactions,
calendar ageing, variable duty, new degradation noise or replacement stock.

## Frozen resampling and clock

- Fleet kinds in RNG order: `calce`, `nasa`, `oxford`, `mixed` (indices 0–3).
  Sizes: **50, 200, 1,000**; 20 layout seeds: integers **2026101800–2026101819**.
- For each seed/kind, use NumPy `Generator(PCG64(SeedSequence([seed, kind_index])))`.
  Generate one 1,000-copy master layout; sizes 50/200 are its first 50/200 rows.
  Reuse precisely the same copies/offsets for all methods, protocols, dates and
  horizons. No seed selection, rejection sampling or dropping poor histories.
- Source pools are the complete inventory: CALCE 7, NASA 34, Oxford 8. Sort cell
  IDs lexically within dataset. A single-dataset fleet samples cells uniformly
  with replacement. For mixed fleets first draw 1,000 dataset indices using
  `rng.integers(0, 3, size=1000)` in order CALCE/NASA/Oxford (equal dataset
  probability, not proportional to inventory). Next, for every fleet kind draw
  1,000 `rng.random` cell uniforms and select `floor(u * pool_size)`; then draw
  1,000 independent `rng.random` age uniforms. Single-dataset kinds do not draw
  dataset indices. Never select cells for eligible forecasts or known outcomes.
- One scenario day = **2 source laboratory cycles** (2 swaps/day, 1 lab cycle
  per swap), exactly as v1. Dataset maximum valid reference-cycle extent / 2
  fixes its commissioning-age span: CALCE **1,008 days**, NASA **98.5 days**,
  Oxford **4,100 days**. Commissioning day = minus age uniform times this span;
  source cycle at fleet today `t` = `2 * (t - commissioning_day)`. Do not reset
  each cell to its first eligible charge, stretch cell lifetime, or extrapolate
  a missing observed history. Different spans represent an explicit synthetic
  age mix for each battery type, not measured operator demographics.
- Common fleet today points: **0, 15, 30, 45, 60 scenario days**, horizons
  **30, 60, 90 days**. All copies have commissioned by day 0. Copies known below
  threshold before day 0 remain in the installed-cohort audit, not at risk.
  No retiring copy is replenished. The mixed fleet shares this same calendar;
  only its commissioning-age distribution differs by source dataset.
- This yields 240 nominal fleets, 3,600 fleet/date/horizon cases per protocol,
  and **7,200 scenario rows**. Seeds, nested sizes, overlapping windows and
  duplicate source histories are dependent, not independent statistical trials.

## Prediction provenance and fixed methods

Use only `reports/improvements/baseline/predictions.parquet`, filtered to
`model=training_selected`, `variant=no_temperature`, and saved E1 cycle-only
predictions from `reports/improvements/E1_cycle/predictions.parquet`. No fit,
refit, target calibration or model selection is allowed. Verify every available
source's held-out membership against the frozen baseline and E1 fold audits;
the source must be a test cell and absent from training/inner-selection cells.
For LODO, its entire source dataset must be absent from training/selection.
Copies always inherit that source's own outer-fold predictions. Missing sources
remain missing. Record fold provenance and input/code SHA-256 fingerprints.

LOCO models were trained within their source dataset on other whole cells.
Mixed LOCO is their aggregation. Mixed LODO aggregates three routed models,
each excluding the particular copy's dataset; it is **not** a single model for
which all three datasets are simultaneously unseen. Other cells' full labelled
trajectories can enter training: this is retrospective cell/dataset holdout,
not a prospective training-calendar cutoff.

1. **Charge count:** v1's deterministic raw linear RUL rule, not probabilities.
   Existing E6 results explicitly find badly miscalibrated risks/intervals, and
   E6 did not validate the 60-source-cycle (30-day) horizon. No calibrated
   crossing probabilities suitable for this test are available. Use the frozen
   RUL function with last 12 distinct-cycle predictions, minimum 6 observations,
   minimum fade 0.000001 SoH/cycle, maximum extrapolation 20,000 cycles and 70%
   endpoint. Retain v1's immediate-below-threshold branch even with a short
   prefix. Only predictions at or before the copy's current age may be read.
   Reject no-prefix and ended-source histories. Translate the predicted crossing
   from the last observed cycle to current age; count 1 only if strictly after
   today and at/before today + horizon, else 0. Predicted overdue is a usable
   zero **incident** forecast, separately counted, not an unreported abstention.
2. **Cycle-count model:** exactly the same RUL/count rule applied to E1's frozen
   held-out quadratic-cycle Ridge predictions, with its own abstention reasons.
3. **Historical replacement rate:** a fixed crude incidence-rate extension of
   v1 for staggered entry. At each today, sum observed first-crossing events
   (first below-70% check already available) and observed at-risk battery-days
   across all copies in that installed cohort. Exposure starts at first valid
   reference and ends at the last reference available by today, capped at first
   observed below-threshold check; initial-left-censored/no-prefix copies add
   no event/exposure. No future check/endpoints enter the rate. Rate = events /
   exposure; if exposure=0 set rate=0 and flag it. Per-copy horizon probability
   is `min(1, rate * horizon_days)`; multiply by the at-risk population (or matched
   population for matched comparisons). No smoothing, fitted parameter or tuning.
   This exposure denominator adapts v1's elapsed-fleet-time count rate to the
   staggered cohorts and is explicitly a methodological difference from v1.
   Both protocols use the same local fleet past: LODO means unseen battery type
   to the learned models, with local past replacement information available to
   the historical comparator, not a zero-history cold start.

Forecast functions receive prediction-only columns, never reference SoH or future
truth. The inventory includes cells without model predictions. An absent, short,
flat, ended or unsupported history is not silently a healthy zero.

## Truth, coverage and scoring

Use frozen baseline endpoints and valid reference histories. Known-already-below
means any <=70% reference observed by current age; remove these copies from
incident risk. Report initially left-censored and retrospectively pre-existing
interpolated crossings (endpoint <= today but below-threshold check not yet
available). These retrospective pre-existing events have zero future incidents.
For observed endpoints, actual = 1 exactly if today < endpoint <= today+horizon,
otherwise 0. A right-censored copy has actual=0 only if its valid reference
follow-up reaches the horizon; otherwise outcome is unknown. Left-censored and
no-valid-reference outcomes are unknown until excluded as known-already-below.

**Full at-risk cohort is the operational population.** Save each scenario's
nominal, already-below, at-risk, predicted-overdue, usable, missing and
unknown-outcome counts; usable/matched/at-risk unique source-cell counts; and
method-specific abstention reasons. Actual range = [known incidents,
known incidents + unknown outcomes]. Model count range = [usable subtotal,
usable subtotal + missing forecasts]. Historical prediction is a point count.
Missing forecasts can contribute anywhere from zero to one incident each.
Report subtotal-vs-truth signed error range and full prediction-vs-truth signed
error range, and absolute-error lower/upper bounds. Do not describe a subtotal
as a complete fleet forecast or report a fabricated exact full-fleet MAE.

**Paired complete-case diagnostic:** require both charge and cycle forecasts
usable AND actual known on precisely the same at-risk copies. Sum actual and
each prediction there; historical = the same rate probability times matched
size. Save predicted/actual counts, signed error (positive = too many), absolute
error and squared error for every date/horizon. Report scenario-macro MAE, RMSE,
signed bias, error per 100 installed copies and error per 100 matched copies,
separately by protocol, fleet kind, size and horizon. Empty matched cases remain
unscored, not zero error. Also save method-specific known-outcome diagnostics
to reveal dependence on the common-support restriction.

Primary interpretation must combine full-cohort uncertainty/coverage with the
paired diagnostic. A conditional charge win means smaller paired MAE than
**both** historical and cycle baselines; report dataset/horizon exceptions and
whether historical still wins LODO. Aggregate comparisons equally over fleet
kinds, sizes, dates, horizons and seeds (no sample-size weighting). Mixed and
single-dataset findings remain separately visible. No significance test,
independent-battery confidence interval or result-driven alternate age mixes.

## Artifacts and interpretation

Write only new v2 source/tests and `reports/fleet_forecast_v2/` artifacts. Save
copy-to-source/commissioning layouts, source/fold audit, copy/date audit, full
scenario results, coverage/reason counts, comparison metrics and a run manifest
with the preregistration commit/hash and protected input hashes before/after.
One plot: 1,000-copy, 60-day-horizon paired actual and all three predicted counts
against the five today points, averaged over all seeds, facets by fleet kind
and protocol. Label it conditional on matched coverage. No selected best seed.

Start `results.md` with <=3 plain-English sentences answering the larger-fleet
comparison, unseen-type transfer, and what Baaz must provide. Then comparison
tables, coverage/bounds and unique-cell support, the plot, exact reproduction
command and limitations. Explicitly state that a 1,000-copy CALCE cohort still
has only 7 real degradation histories; random offsets/copies never create new
independent batteries, and observed reference crossings are not field swaps.
List a Baaz pilot's pack IDs/types and commissioning dates, timestamped swap logs
and throughput, dated replacement records/reasons and pack linkage, and periodic
independent capacity checks under documented conditions, with 90-day follow-up
and training/held-out packs, types and calendar periods fixed before scoring.
