# Survival analysis for replacement timing: preregistration

Commit this file before any survival fitting, threshold selection, evaluation
or fleet/hybrid survival scoring. Starting repository commit: `1fadd8f`.
Only frozen definitions, schemas, dependencies, source/fold counts and methods
documentation were inspected. No survival results were calculated. Do not tune
on outer-test outcomes. Keep every pre-existing tracked file byte-identical;
write only new survival source/tests and `reports/survival/` artifacts.

## Endpoint, populations and clocks

Use frozen `reports/improvements/baseline/end_of_life.csv` exactly: the first
**observed reference 70% crossing interpolated between adjacent valid checks**,
as documented by the frozen RUL study. Do not replace it with the later first
below-threshold check. Save that check separately for as-of fleet eligibility.
Time is absolute source lab cycles; scenario days = cycles / 2 (two swaps/day,
one lab cycle/swap). No field replacement date, physical failure or Baaz outcome
is implied. Right censor at last valid capacity check; initially <=70% cells
are left-censored and excluded from incident fitting and individual validation.

The full inventory is 49 cells, with 19 observed crossings, 14 right-censored
and 16 initially below. Frozen held-out charge predictions cover 41 cells:
17 observed crossings, 14 right-censored and 10 initially below, hence **31
incident-modelling cells with 17 events**. Two of the 19 full-inventory events
have no eligible frozen charge predictions. Report both denominators, all
missing coverage and the all-19-event first-alert denominator. Repeated charge
records, landmarks, person-periods, fleets and copies are not new batteries.

## Frozen upstream model and nested feature provenance

Use the same 44 no-temperature outer folds (41 LOCO within dataset, three LODO)
from the frozen audit. Outer test covariates use the frozen training-selected
held-out raw charge SoH predictions, never the all-cell fitted reference model.
No outer-test cell or its dataset (LODO) may enter fitting, scaling, selection,
feature generation, censoring curves or threshold selection for its prediction.

Global out-of-fold predictions of **training** cells can have indirect access
to the outer test cell. Therefore construct survival-training charge features
by **two-fold cross-fitting within the current outer training pool**. Within
each dataset sort cells by SHA-256 of `survival-charge-crossfit|42|group_id` and
assign alternating indices 0/1; fit on one half and predict the other, twice.
Use the frozen five charge predictors, candidate specifications, cell-balanced
weights, seed 42 and frozen inner-selection algorithm, applied only inside
that feature-training half. If the half has fewer than three cells, choose
the fixed frozen linear Ridge alpha=1 candidate without selection. Log every
upstream fit/selection/test pool and assert source and outer holdouts absent.
This refits the frozen **algorithm** for nested training covariates; it does not
change frozen artifacts. Pool-level caching is allowed, never test-label reuse.

Threshold selection uses the frozen outer fold's inner-training/validation
cell lists. Generate cross-fitted charge features inside inner-training only;
fit a charge model on all inner-training (same frozen selection rule) to predict
inner-validation cells. Fit survival models/censoring curves on inner-training
only and select thresholds from inner-validation outcomes. Final survival models
fit all outer-training cross-fitted features; outer test charge features remain
the original frozen outer-fold predictions. No survival hyperparameter search.

## Inputs, history and person-periods

- Primary inputs are exactly **current raw predicted SoH** and **recent fade
  rate** = minus OLS slope of the last 12 distinct source-cycle predictions,
  requiring at least six finite points and a nonzero cycle span. Do not clip
  rising slopes or SoH, smooth, use true capacity, IDs or discharge features.
- Sort by cycle then charge-end time, keeping the final point within each cycle.
  At a requested today only points <=today can enter either input. Use the last
  available eligible prefix (last-observation-carried-forward, including after
  the last charge record); retain its age/staleness in the audit, not as a third
  primary predictor. This is an explicit difference from v2's ended-record
  abstention. No future trajectory is supplied to a forecast.
- The separately labelled **known-history hazard** adds current source-cycle
  count as its third and final input. No other known-history variant or Weibull
  AFT is run; four fixed survival models are sufficient for these few events.
- Start a feature-based training cell when six past predictions first permit
  both inputs, no earlier than its first valid reference. End at the frozen
  event or censor time. Cells/events before feature entry have no feature-model
  exposure; report this loss rather than inventing labels. Carry the causal
  inputs between charge updates. Split into at most **10-cycle periods** from
  entry, with a shorter terminal interval. Cox uses exact start/stop exposure.
  Logistic uses full non-event periods plus the terminal event period (event
  within that 10-cycle bucket); omit a partially observed terminal non-event
  interval. A cell has at most one event row. No class reweighting or oversampling.

## Fixed models and forecast rules

1. **Kaplan–Meier (KM):** delayed entry at first valid reference, one event/censor
   record per incident training cell, no charge inputs. Risk over H =
   `1 - S(today+H)/S(today)`. Never fit using test survival times. If S(today)=0,
   or today+H exceeds maximum training event/censor time, forecast unavailable.
2. **Discrete-time logistic hazard (primary):** standardise the two training
   inputs using training person-period rows only; scikit-learn L2 logistic
   regression C=1, intercept enabled, lbfgs, max_iter=2000, tol=1e-9, no class
   weights. Ten-cycle hazard p produces H-cycle risk `1-(1-p)^(H/10)`.
   Hold current covariates fixed over the forecast horizon. Partial-horizon
   conversion is a declared constant-hazard approximation within a bucket.
3. **Known-history logistic hazard:** identical settings with the third cycle
   input; freeze all three current inputs over a forecast horizon. Label separately.
4. **Time-varying Cox PH:** same two primary standardised inputs on start/stop
   rows; risk sets satisfy start < event_time <= stop, Breslow ties. Minimise
   negative partial log likelihood / number of events plus `0.5*||beta||^2`
   (fixed L2 alpha=1). SciPy L-BFGS-B from zero, maxiter=1000, ftol=1e-12,
   gtol=1e-8. Breslow baseline cumulative hazard; risk =
   `1-exp(-[H0(today+H)-H0(today)]*exp(beta*x_today))`. Future inputs are held
   at current values. No baseline-hazard extrapolation beyond maximum training
   event/censor time. Save convergence, coefficients and available event counts.

If eligible training periods have zero events, feature models give zero risk
with a conspicuous no-training-events flag; no fit/search is invented. Other
fit failures remain unavailable and reported. KM zero-event risk is zero within
its support. No alternative penalty, feature window or rescue model after results.

## Probability accuracy, calibration and ranking

At each distinct native frozen charge checkpoint before the true event/censor,
evaluate risks at **30/60/90 scenario days = 60/120/180 cycles**. Initial-below
cells never enter. Report forecast coverage and real cell/event support, not
record counts alone. Use source-cell-macro means; dataset-macro and pooled rows
are supplementary. Explicitly retain right-censored observations.

Estimate censoring survival G from the **outer-training** incident cells, with
delayed entry and event/censor indicators reversed. At landmark t use conditional
G(u|t)=G(u)/G(t): an event by t+H contributes `(1-p)^2/G(T-)`; a known survivor
through t+H contributes `p^2/G(t+H)`; censoring before H contributes zero but
remains in the IPCW Brier denominator. These denominators mean conditional G.
Require relevant conditional G >=0.05 and times within training support;
otherwise mark unsupported, never silently clip weights or score a fabricated
zero. Report this limitation and the independent-censoring assumption. Primary
Brier is the mean IPCW contribution per cell, then across cells. Also save raw
known-outcome Brier, clearly conditional on observed follow-up.

Comparators: frozen raw linear RUL converted to a **deterministic 0/1 horizon
indicator**, not a calibrated probability; and frozen E6 posterior risks.
Recompute E6's same fixed 512 posterior draws for the additional 60-cycle
probability horizon; no E6 model/threshold refit. Existing 120/180 risks must
match saved values. At native checkpoints E6 needs a state at that same cycle;
absent states remain missing. Report pairwise Brier comparisons on exactly
common model/landmark/IPCW support, so availability cannot masquerade as a gain.

Calibration uses fixed bins [0,.2,.4,.6,.8,1]. Within a bin, weight each cell's
records to total one, then use IPCW known outcomes for an observed event-frequency
ratio; predicted mean uses those same weights. Show row/cell/event-cell counts.
A bin is sufficiently populated only with **>=5 distinct cells and >=2 distinct
event-source cells**; all other bins are sparse, visibly marked, never hidden.
No probability recalibration is fitted. The single calibration plot facets both
protocols and all three horizons, showing KM/primary hazard/Cox/E6, sparse bins
as hollow/faint markers; known-history and linear bins remain in tables/CSV.

Time-dependent ranking: **event-landmark concordance C(H)**. For each real
observed event i at T_i, let today=T_i-H; require today >= both cells' first valid
reference. Comparator j must have a known event/censor time strictly >T_i.
Compare each cell's as-of H-cycle risk, using its own held-out outer model:
1 if i's risk is higher, .5 for ties within 1e-8, else 0. For E6 use the latest
past saved state (record staleness), not a future state. Weight each pair by
`1/G_i(T_i-|today)^2`, requiring the same >=0.05 support; also save unweighted
concordance. This is the explicitly defined horizon-specific dynamic concordance,
not interchangeable with a static Harrell statistic. Report within-dataset
pair/cell/earlier-event counts; equal-dataset aggregation excludes unscorable
groups. No pair, landmark or resampled copy is independent evidence. Save
pairwise common-method ranking support as well as method-specific coverage.

## First-alert decision backtest

Horizons **14/60/90 days = 28/120/180 cycles**. For each survival model/fold/H,
choose threshold from [.50,.65,.80,.90,.95,.99,1.01] using inner-validation only.
Same first-event cost as E6: timely 0; premature 2; late/no alarm/no eligible
history 3. Right-censored cells add cost 2 only if first_alarm+H <=last check,
otherwise 0; explicitly report unresolved censoring. Average costs over cells;
ties choose largest threshold. No observed inner endpoint =>1.01 (abstain).

Evaluate the **first** native eligible charge-checkpoint alert, including a late
first alert after the endpoint, using the same blind-history convention as E6.
For the all-19-event denominator classify timely (0<lead<=H), premature (lead>H),
late (lead<=0), or none; separately show no history and any eligible timely
opportunity. Compare directly with the frozen raw-linear and E6 saved first-alert
events at these exact horizons, without retuning them. Right-censored false and
unresolved first alerts are a separate table; initial-below cells are excluded.
Also show the 17-event eligible-frozen-source cohort and exact common histories.

## Fleet probabilities and survival hybrid

Use every saved v2 layout/seed/size/today/horizon/protocol, with v2's known-below
eligibility and outcome bounds. A copy always uses its source's own held-out
survival model/feature history; sources with no frozen outer-test fold remain
missing. Sum each survival method's risks. Missing probabilities yield count
bounds [subtotal, subtotal+missing]; missing outcomes yield v2 actual bounds.
Compare each method with the same historical-rate probability on precisely that
method's known-outcome/usable-probability subset; report matched count MAE/bias,
full count/error bounds, missing counts, and distinct real/event support.
Do not pool fleets as independent or infer accuracy at 1,000 real packs.

Hybrid exists: re-run its saved v2 and unique-cell cases, same historical/oracle
K, ties, censoring/metric rules and missing-score age fallback. Rank each of the
four survival probabilities descending, separately per horizon. Do not discard
unavailable probabilities or unknown outcomes. Oracle remains unavailable if
any outcome is unknown. Compare with saved charge SoH/RUL, cycle SoH, age/random
rankings, retaining all conditional/full-bound and unique/event counts. Unique
cohorts remain exactly the prior hybrid's fixed 49-source union and its dataset
subsets; copies add no independent evidence. Primary survival hybrid is the
two-input hazard, without selecting a winning model after results.

## Reporting, evidence and integrity

Save nested charge split audits/predictions, person-period training exposure
counts, fold model parameters/convergence, fitted model bundles, threshold audit,
outer probability snapshots, Brier/calibration/concordance and paired-support
tables, all first-alert events including failures, fleet counts and hybrid
results, input/code hashes and protected tracked-file hashes before/after.
Record this preregistration's commit/hash and execution timestamps. Failed or
worse methods are retained. No significance/independent-cell CI from repeated
records, no claim that IPCW solves informative censoring or chemistry transfer.

`results.md` begins with <=3 plain-English sentences answering alert timing,
probability calibration, and whether this supports a Baaz demo. Then comparison
tables, calibration plot, alert tables, fleet/hybrid findings and coverage. State
plainly that 19 full-inventory events (17 in the 41 prediction-source cells,
31 after excluding initial-below) are very little evidence: indicative, not proof.
List Baaz stable pack IDs/types, commissioning dates, replacement dates/reasons,
periodic independent capacity checks with test conditions, linked swap/throughput
and charge logs, enough follow-up through 90 days after final today, and fixed
pack/type/calendar holdouts with calibration packs separate from evaluation.

Method references: [IPCW Brier](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html),
[training-only censoring support](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html),
[time-varying Cox start/stop design](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html).
Implement with the available NumPy/SciPy/scikit-learn environment; no dependency
installation or modification of earlier requirements is needed.
