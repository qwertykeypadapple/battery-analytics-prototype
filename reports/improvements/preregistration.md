# Improvement experiments: preregistration v1

Written and committed before E0–E9 are run. Baseline tag: `baseline-v1`.
The baseline code and its saved evaluation artifacts will not be overwritten.
`baseline/manifest.json` freezes their hashes and the starting commit.
Reference: 3.719682 pp LOCO cell-macro MAE, 13.887386 pp LODO,
and zero timely first alarms among 19 observed 70% capacity endpoints at 28 cycles.

## Common design, estimands, and selection

- Reuse exactly the 41 common-feature LOCO and three LODO outer folds in the
  baseline fold audit. Reuse their exact inner train/validation cell lists.
  No random row split, test-based early stopping, or test-based feature selection.
- Eligible rows, labels, 70% endpoint, and original source cycle indices are
  frozen. Do not remove a bad cell after inspecting errors. An unavailable
  prediction remains an explicit coverage loss. Label QC remains the baseline QC.
- Primary SoH outcome: mean of per-cell MAEs, in percentage points. Also save
  per-cell/dataset MAE and RMSE, pooled-row and equal-dataset aggregates,
  below-70% and 70–80% bands, and row/cell coverage, for BOTH protocols.
  Restricted-history variants get paired baseline results on precisely the same
  rows, alongside the unrestricted frozen headline. No claiming a cohort change
  as an accuracy gain. Smoothing is supplementary, never the headline by default.
- All new regressors are StandardScaler + Ridge(alpha=1), with the baseline
  inverse-cell-count sample weights. A degree-two cycle baseline includes
  `cycle_index` and its square, scaled on training rows. Parameters are fixed;
  this series does not repeat the original four-family search.
- Model/representation choices inside E4/E8 use only the original inner cell
  split and cell-macro MAE. Prefer the simpler candidate when within 0.20 pp of
  the minimum, using the candidate order stated below. Refit the selected
  configuration on all outer training cells. Record every candidate score.
- The entire series is exploratory research on previously inspected public
  datasets, NOT an independent confirmatory trial. Reusing locked outer splits
  preserves cell separation but cannot erase prior knowledge of these data.
- Seeds: 42 for fitting/splits; calibration repetitions 42–46. No test-label
  values enter any input except the single explicitly permitted entry anchor.
  IDs are grouping keys only; dataset IDs do not enter regressors.
- New charge features receive only current charge arrays and rated Ah. Reference
  transformations receive ordered charge features, not capacity labels. Capacity
  anchoring, filter-prior training, endpoint evaluation, and calibration are
  separate functions with explicit interfaces and timing assumptions.
- Every stage saves its full predictions/audit and commits results before the
  next experiment. Implementation fixes are logged in `deviations.md`; any
  scientific configuration change requires original AND revised outputs.

## E0 — diagnostics only

1. Post-hoc minimum-history sensitivity: retain cells with >=20 baseline eligible
   predictions. Recompute metrics from saved predictions; report excluded cells
   and rows. This does not replace the headline.
2. NASA below-70% errors by charge/test ambient: cold <10°C, room >=10 and <35°C,
   warm >=35°C. Show per-cell and group cell-macro/pooled errors and counts for
   both protocols. Hypothesis: cold groups have worse low-capacity errors. This is
   observational and cannot distinguish reversible cold loss from ageing.

## E1 — known-history cycle baseline

Fit the fixed degree-two Ridge curve on cycle index alone, using the same outer
and inner cell groups. LOCO is dataset-specific by construction; LODO uses the
two source datasets and common source-cycle units without a target identifier.
Primary: LOCO cell-macro MAE and paired per-cell comparison with frozen charge
features. Report LODO even if inappropriate across different cycle conventions.
Label this setting **known-history**; frozen charge-only is **unknown-age**.

## E2 — normalised upper-window charge throughput

For 3.9→4.0, 4.0→4.1, 4.1→4.18, and 3.9→4.18 V, integrate positive charging
current using trapezoids with interpolated threshold endpoints; divide by
3600 × rated Ah. Same voltage smoothing, crossings and eligibility as baseline.
If finite-current integration is impossible but baseline duration and median
current exist, use duration × median current / (3600 × rated Ah), and log the
fallback for each interval/row. Do not use any discharge or whole-charge counter.

- E2_replace: four throughputs + charging C-rate.
- E2_alongside: baseline five predictors + four throughputs.
- Primary interest: LODO cell-macro MAE; all common metrics still reported.

## E3 — per-cell entry references

K=5, fixed (no K sweep on outer tests). Order by source cycle and charge end time.
E3_delta subtracts the median of each cell's first five eligible charge-feature
vectors from the current baseline vector. The first five observations are warmup,
excluded from training/scoring, and cells with <=5 observations have no forecast.
This is **entry-history / no labels**, not evidence that entry is actually new.

E3_anchor fits the unmodified baseline-feature Ridge on training cells. For each
cell, add `first measured SoH - first model prediction` to subsequent predictions.
Exactly ONE first-entry capacity measurement per cell is allowed. Exclude that
observation from scoring. It is assumed available after its paired discharge and
before the next eligible charge; periodic Oxford's same-checkpoint discharge
already precedes its charge. Source records lacking absolute NASA timestamps use
documented operation order, not invented absolute times. No subsequent true SoH
is read by the anchoring function. This variant is **commissioning-labelled**.

Report separately cells whose first eligible SoH <70%, 70–90%, or >=90%, plus
baseline left-censoring status. A delta does not force entry SoH=100%.

## E4 — combined known-history model

All candidates add cycle and cycle-squared to one of: raw features,
E2_replace, E2_alongside, E3_delta (simplicity/tie order as written). No capacity
anchor is allowed in E4. All candidates train/validate/score after the same K=5
warmup, so selection has a common cohort. Choose on the original inner cells,
then refit on outer training cells. Compare with E1, paired baseline, and the
unrestricted frozen baseline. Primary: LOCO cell-macro MAE.

## E5 — causal SoH/rate state-space filter

Apply to E4 held-out observation estimates, adding a state [SoH, rate] with rate
in SoH per training-median cell-history span. Gaussian constant-velocity model,
F=[[1,dt],[0,1]], white-acceleration Q=q²[[dt³/3,dt²/2],[dt²/2,dt]]. Initial
SoH=first model observation; initial rate=median training-cell OLS reference rate,
with training-cell rate variance and observation residual variance as covariance.
No test-cell reference capacity is used to initialise or update the filter.

Noise/prior fitting uses only the INNER TRAINING cells: three-fold GroupKFold
whole-cell cross-fits produce residual observation MSE (cell-balanced; floor
1e-6 SoH²). q in {0.0001,0.001,0.01,0.05} is selected by cross-fitted training
cell-macro filtered MAE, preferring the smallest q within 0.20 pp of the minimum.
Rate-variance floor=1e-6 in normalised-time units. Store all noise/prior provenance.
Outer refit changes the observation regressor only; noise and prior stay frozen.
Duplicate source cycles are processed chronologically; dt floor=1e-6 normalised
units. No centred smoother, reverse pass, test residual fitting, or monotonic
projection. Save Gaussian nominal 50/80/95% SoH intervals, empirical coverage,
cell-macro coverage, width and MAE. Compare with E4 raw and trailing-five median.

## E6 — probabilistic RUL and first alarms

Use the E5 fleet-trained rate prior and causal posterior. Draw 512 fixed Sobol
normal pairs (seed42) from each [SoH,rate] Gaussian posterior. Hold each sampled
rate constant over the forecast; project positive sampled rates to zero (no
crossing), and treat already-below-threshold draws as immediate crossings. This
defines a simple empirical-Bayes linear-degradation approximation, NOT a full
electrochemical/first-passage stochastic process. Future acceleration is omitted;
report calibration honestly rather than describing intervals as guaranteed.

P(cross within H) is the draw fraction crossing by H. H={28,120,180} source cycles,
corresponding to 14,60,90 scenario days at 2 swaps/day and 1 lab cycle/swap. RUL
posterior 50/80/95% intervals may have infinite upper bounds; retain these and
report finite-bound coverage/width separately. Evaluate at the existing E0 review
20/50/85%-of-observed-life anchors; truth locates evaluation points only.

For EACH outer fold and horizon, choose p* on its original inner validation cells
from {0.50,0.65,0.80,0.90,0.95,0.99,1.01}. Cost per observed endpoint: timely=0,
premature=2, late/no-alert=3; right-censored known-premature first alarm=2.
Average per cell; left-censored cells excluded from incident-event loss. If no
observed inner endpoint exists, p*=1.01 (explicit abstention). Ties choose larger
p*. Noise/prior never fit these validation cells. Report no-eligible-history
separately and keep all 19 observed endpoints in outer event denominators.

Report first timely/premature/late/none, lead-time quartiles in cycles/days,
pointwise Brier/reliability for known outcomes, and RUL interval calibration.
Censoring: right-censored negatives are known only when observation+horizon <=
last above-threshold reference; no exact errors for right/left-censored endpoints.
Compare frozen raw-linear alarms at ALL three horizons using the same histories.
Report median training/reference degradation rate per dataset and diagnostic
`SoH MAE (fraction)/median positive degradation rate` in source cycles. It is an
error-propagation scale, not a mathematical lower bound or proof of feasibility.

## E7 — relative fast-degradation flag

X=25% (upper quartile). Training true rates are negative OLS slopes of each
cell's valid reference SoH vs source cycles; nonpositive degradation rates remain
in the population. Set cutoff to the training 75th percentile (dataset-specific
in LOCO, pooled sources in LODO). At the 20th eligible E4 observation, flag if the
causal E5 posterior mean degradation rate exceeds this cutoff. Cells with <20
available observations abstain, counted explicitly. True fast labels compare
the held-out cell's full-reference OLS rate with the TRAINING cutoff. Report
precision/recall, coverage, counts, and observed endpoint ages for flag/no-flag;
also show descriptive within-dataset full-reference quartile membership.
No endpoint is invented for censored cells; full-life truth is evaluation only.

## E8 — incremental-capacity addition

Build charge-only cumulative integral Q/rated Ah, interpolate on 3.900–4.180 V
at 0.002 V (141 points), enforcing a monotone voltage envelope and using only
the first time each voltage is reached. Smooth Q with Savitzky–Golay window11,
polynomial2, then differentiate. `find_peaks`: prominence >=0.01 fractional-Ah/V,
distance10 bins; exclude peaks within five bins of either boundary. Select the
largest-prominence eligible peak: voltage, height, and integrated area within
±0.020 V. Explicit peak-present flag; missing values get TRAINING medians
(all-missing training column→0 with explicit peak-present flag). Report peak
coverage per cell/dataset, boundary limitations, and throughput fallback count.

Add these four predictors to EACH E4 candidate, with its identical K=5 cohort.
Inner selection compares E4's four candidates then their four dQ/dV additions,
same 0.20-pp simpler-first rule. Also report a forced-dQ/dV candidate winner
selected among the four additions, so a rejected addition is still visible.
Primary: LOCO cell-macro MAE. The optional-addition E8 pipeline is the dashboard
"inner-selected improvement", not a winner chosen by inspecting outer scores.
E5/E6/E7 remain explicitly evaluated on E4; dashboard uncertainty/alerts use the
E4/E5 evaluated tracker rather than silently grafting its covariance onto E8.

## E9 — target calibration learning curve

Freeze an E2_replace source-only Ridge for each LODO target. No source fine-tuning.
Per target, choose one E9 locked cell by seeded42 permutation of sorted eligible
IDs (last element); exclude it from ALL calibration/development curves. This
cell has appeared in prior public-data analyses, so "locked" means locked for
this calibration experiment, not never previously inspected.

For k={0,1,2,4} target calibration cells, choose five seeded42–46 permutations
of the remaining pool; first k calibrate, the other cells score. Calibrate only
an additive SoH offset = mean of calibration-cell mean(target SoH-source estimate).
k=0 offset0. No target labels used to choose a slope, features or hyperparameters.
Save every choice, remaining cell prediction, and spread across repeats. Shared
cell choices make repeats correlated; report ranges/SD, not independent CIs.

Only after all choices/rules above are fixed, score the locked cell for the same
predeclared k/repetitions. Keep its table separate from development curves.
Plot mean cell-macro MAE ± repeat SD and frozen LODO reference. Report changing
cohorts, per-cell results and matched k0 comparisons. Cells with only one or two
measurements remain included. Primary: target cell-macro MAE vs k0 on matched
evaluation cells; other metrics and low-health bands accompany every k.

## Reporting and recommendation

Show every experiment, all fixed variants and rejected candidate inner scores.
Do not claim "best" from the minimum outer-test score as an unbiased estimate.
Recommendation considers the predeclared inner-selected E4/E8 pipelines and the
simple E1/E2 controls; <0.20 pp improvements do not justify extra complexity.
If uncertainty, alarms, or transfer get worse, say so. None of these experiments
establishes Baaz pack performance, field-failure prevention, energy savings,
life extension, federation, or >45°C performance.
