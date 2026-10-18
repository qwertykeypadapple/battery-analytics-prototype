# Fleet replacement-count forecast: preregistration v1

Committed before computing fleet-count results. No model fitting or tuning in
this experiment. Inputs are frozen baseline-v1 training-selected common-feature
out-of-fold predictions, and preregistered E1 cycle-only out-of-fold predictions.
Primary protocol: leave-one-cell-out; repeat leave-one-dataset-out separately.
Probabilistic RUL is not yet available at registration, so this experiment uses
deterministic counts of predicted RUL crossings, NOT sums of calibrated risks.
Do not replace it with a later probabilistic winner after inspecting results.

## Fleets, dates and horizons

Each dataset is one retrospective fleet (NASA34, CALCE7, Oxford8 inventory cells,
including cells without eligible model histories). Source-cycle age is treated
as a common simulated fleet clock; these are not contemporaneous real fleets.
Today points are 20%,40%,60%,80% of the interval from the earliest valid reference
cycle to the latest valid reference cycle IN EACH DATASET, rounded to nearest
integer cycle. Full recorded extent locates evaluation anchors only. No truth
after today enters a forecast. These choices cannot be changed after results.

Horizons=30,60,90 scenario days =60,120,180 source cycles at 2 swaps/day and
1 laboratory cycle/swap. Partial swaps need a measured throughput conversion in
a real deployment. Endpoint=baseline interpolated first 70% reference crossing;
also retain the first observed below-70% check date for information availability.

## As-of populations, prediction and comparators

- Known-already-below: at least one valid reference <=70% observed by today;
  remove from incident replacement forecasting and report separately. Also
  report retrospectively pre-existing threshold crossings not yet observed.
- No usable model history: no eligible prefix, fewer than six distinct cycle
  observations, source record ended before today, flat/increasing trend, or
  unsupported >20,000-cycle extrapolation. Keep distinct reason counts. No
  usable history is never treated as a healthy battery or a predicted zero.
- Main charge-model and E1 predictions: raw held-out SoH prefixes only; baseline
  linear RUL, last12 distinct observations, min6. Project the as-of crossing
  date to today. Predicted already-below/overdue is reported separately from a
  predicted incident crossing strictly AFTER today and within the horizon.
- Historical-rate baseline: number of incident crossings whose below-threshold
  check has been observed by today / elapsed fleet source cycles since the
  first valid reference. Multiply by horizon and cap at the at-risk population.
  Initial left-censored cells are not incident historical events. No smoothing,
  pseudo-count, fitted threshold, or future endpoint is used in this baseline.
- The primary table shows BOTH model counts, historical-rate expected count,
  known observed actual count and its censoring upper bound, plus already-below,
  no-history, predicted-overdue and unknown-outcome counts.

## Truth, missing follow-up and metrics

Actual incident endpoints satisfy today < interpolated endpoint <= today+H.
For a cell above threshold at its final reference before today+H, absence of a
recorded crossing does not establish a zero. Its outcome is unknown unless its
recorded above-threshold follow-up reaches the horizon, or an observed first
crossing settles the outcome. Already-left-censored endpoints are not invented.
If the first interpolated crossing is before today but not yet observed, its
incident outcome for this period is zero, with that mismatch disclosed.

Full-fleet actual count is [observed incidents, observed incidents+unknowns].
Report signed/absolute error bounds against this interval, rather than inventing
an exact actual count when follow-up is missing. Unknown predictions have count
bounds too; point count is a subtotal over usable cells, clearly labelled.

Supplementary matched evaluation: restrict each fleet/date/horizon to cells with
usable forecasts from BOTH charge and E1 methods and a known outcome. Compare
both counts and historical-rate baseline on precisely these cells. Report its
population size and exclusions for every row. This is a coverage sensitivity,
not a replacement full-fleet headline. Historical-rate comparator is scaled to
the matched at-risk share. Macro MAE, RMSE and signed bias are averaged over
predeclared fleet/date/horizon scenarios; overlapping dates/horizons are NOT
independent replications. Also report each fleet and horizon separately.

## Interpretation and plot

One plot: observed actual-count bounds and charge/cycle/historical counts vs
today, faceted by dataset and horizon, LOCO only. Report signed bias (positive
means overprediction) and whether errors cancel. Few batteries, few endpoint
events, censoring and artificial synchronised age preclude an operational
replacement-budgeting claim without a prospective larger-fleet test. Even a
descriptive count improvement does not establish energy savings or life extension.
