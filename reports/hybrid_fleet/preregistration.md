# Hybrid fleet forecast: preregistration

Commit this file before running hybrid rankings or calculating hybrid metrics.
Starting repository state: `c110d7b` (completed fleet forecast v2). Only input
schemas, existing study rules and source endpoint inventory were inspected.
No hybrid outcomes/rank comparisons have been computed. No result-driven tuning,
model fitting, calibration, alternate seeds, score transforms or fallback changes.
Keep every pre-existing tracked file, including baseline and earlier experiments,
byte-identical. Write only new hybrid source/tests and `reports/hybrid_fleet/`.

## Question and fixed populations

Does historical-rate budgeting plus charge-based prioritization identify the
right first 70% capacity crossings better than cycle-model SoH, age and random?
Charge **current SoH** is the primary model ranking. Charge **remaining life**
is a separately reported secondary ranking, not a post-result winner selection.
Events are laboratory reference-capacity crossings, not field replacements.

1. **V2 layouts:** read saved `reports/fleet_forecast_v2/layouts.csv` and
   `copy_date_audit.parquet` without regenerating or changing them. Use every
   registered kind (CALCE, NASA, Oxford, equal-dataset mixed), installed cohort
   size (50/200/1,000 nested copies), seed (2026101800–2026101819), today
   (0/15/30/45/60 scenario days), horizon (30/60/90 days), and LOCO/LODO protocol.
   One scenario day remains two source cycles; copy ages/eligibility/RUL/truth
   and historical exposure/events are exactly v2's saved as-of audit values.
   No already-observed-below copy is at risk; no retiring copy is replenished.
2. **Unique real cells:** one fixed cohort per dataset containing every inventory
   cell exactly once (CALCE 7, NASA 34, Oxford 8); mixed = their 49-cell union.
   Membership is not sampled. To retain mixed ages on v2's common today clock,
   each cell retains the commissioning offset of its **first copy_index** in
   its single-dataset master layout at the first registered seed, 2026101800.
   Use that same cell/offset in mixed; do not draw new offsets or optimize them.
   Evaluate at all five v2 today points/horizons and both protocols. There are
   no repeated source cells inside a unique cohort. Offsets are still simulated,
   not actual lab/pack commissioning dates. These four overlapping cohorts and
   repeated windows do not create independent replication.

Unique-cell evaluation is the main independent-support check. Resampled fleets
are aggregation/age-mixture sensitivities, not new batteries. Report each result
with the number of unique real source cells and distinct source event histories,
not a claim of 1,000 independent batteries. Across all sources there are only
19 observed crossings (CALCE 7, NASA 10, Oxford 2), with fewer incident sources
represented after conditioning on today/eligibility/follow-up. Count distinct
positive sources across windows, never add repeated endpoints as new events.

## Prediction provenance and scores

Use frozen training-selected, no-temperature baseline held-out raw predictions
and frozen E1 cycle-only held-out raw predictions, exactly as v2. Verify every
available source against baseline/E1 outer-fold audits. LOCO excludes that cell
from fitting and selection; LODO excludes its whole dataset. Mixed LODO routes
each source to its own excluded-dataset model, not one model excluding all three
datasets. Other source cells' full trajectories can enter fitting; this is
retrospective cell/type holdout, not a prospective training-calendar cutoff.

At each snapshot, before reading any future outcome for ranking:

- **Charge SoH:** lowest last raw predicted SoH at/before current source age.
  Sort prediction history by cycle then charge-end time, keeping the last
  within-cycle row. At least one finite past prediction is needed; no-prefix,
  nonfinite last prediction, or age beyond the last eligible prediction makes
  the score missing. Never read the reference SoH to rank. No smoothing/clipping.
- **Charge RUL:** lowest remaining source cycles from v2's frozen linear raw-RUL
  lookup. Retain finite negative/zero overdue RUL as highest priority, rather
  than the zero-incident convention used to count future crossings in v2.
  Short/flat/unsupported/ended/no-prefix forecasts are missing as in v2.
- **Cycle SoH:** lowest last raw E1 predicted SoH, with the same current-score
  eligibility as charge SoH. This is not the E1 RUL/incident forecast.
- **Age:** longest time since commissioning first (equivalently source cycle
  age under the fixed common two-cycles/day scenario).
- **Random:** uniform random ranking; average **2,000 draws** per scenario/K
  choice. Draw the numbers of selected known positives, known negatives and
  unknown outcomes from the equivalent multivariate hypergeometric distribution,
  rather than sorting thousands of explicit random permutations. RNG =
  `Generator(PCG64(SeedSequence([20261018, 77, representation_index, kind_index,
  seed, today, horizon_days, k_mode_index])))`, with representation indices
  unique=0/v2=1, kind order calce/nasa/oxford/mixed, K mode historical=0/oracle=1.
  Use the same random draws across LOCO and LODO. Also save exact random expected
  true positives `K * known_positive_count / at_risk_count` as an MC check.

**Missing-score fallback:** put all finite model scores first, ascending. Append
all missing-score batteries, ordered oldest first. No battery disappears.
Record missing-score and fallback-selected copies AND distinct real sources.
All ties (including equal finite model scores) use the ascending SHA-256 hex
digest of `hybrid-fleet-v1|representation|kind|seed|unit_id`; a v2 unit_id is the
decimal copy_index, a unique-cell unit_id is dataset:cell_id. This tie-break is
fixed across methods, dates, horizons and protocols, never based on an outcome.

## K and ranking selection

- **Primary, historical K:** use v2's crude local historical incidence rate:
  sum prior observed first-crossing events / sum observed at-risk battery-days
  in the full installed cohort (including already-retired copies' past exposure).
  If exposure is zero, rate=0 and flag it. Count =
  `at_risk_count * min(1, rate * horizon_days)`; recompute for the unique cohort
  using its own histories. K = `floor(count + 0.5)`, clipped to [0, at_risk_count]
  (half-up rounding, not Python bankers' rounding). Every method flags exactly
  the same K-budget on the same at-risk population. No future labels enter K.
- **Oracle K, diagnostic only:** K = actual incident count in that same full
  at-risk population, only if **every** at-risk outcome is known. If any outcome
  is unknown, mark oracle unavailable; do not substitute the observed lower
  bound or drop unknown-outcome batteries from the oracle's candidate pool.
  This future-informed K is not usable in practice and cannot be a pilot claim.
- Rankings/fallback are identical for both K modes. Only the cutoff changes.
  Selection is performed before truth is attached for evaluation. Do not filter
  candidates by future follow-up, actual crossing or paired model availability.

## Truth, precision/recall and censoring

Use exactly v2's known-already-below exclusion, frozen interpolated first 70%
reference endpoint, right/left censoring and retrospective pre-existing flags.
For every method/window save M at risk, K, known actual positives A, unknown
outcomes U, selected known positives T, selected unknowns Uf, selected known
negatives, distinct real sources and distinct source positives/catches.

- If all outcomes are known: precision = T/K (undefined when K=0), recall = T/A
  (undefined when A=0). K=0 with A>0 gives recall=0, not a perfect result.
- With incomplete truth, report **conditional** precision among flagged copies
  with known outcomes: `T/(K-Uf)` (undefined if none); conditional recall of known
  incidents: `T/A` (undefined if none). These are not exact full-fleet metrics.
  Save fully observed-only summaries separately. Do not label unknowns negative.
- Always retain full-population precision bounds `[T/K, (T+Uf)/K]` for K>0.
  Let Uo=U-Uf. For recall conditional on at least one actual incident existing,
  lower=`T/(A+Uo)` and upper=`(T+Uf)/(A+Uf)`. If A+U=0 both are undefined; if
  A+Uo=0 but an incident is possible, lower=1 (all possible incidents flagged);
  if A+Uf=0 but an incident is possible, upper=0 (all possible incidents missed).
  These are censoring bounds, not calibrated probability intervals.
- For random, compute metrics on each of the 2,000 equivalent draws and average
  defined ratios/bounds; save draw-availability counts and expected selected
  positives. Never claim random draws are additional real batteries.

Report scenario-macro conditional precision/recall, full bounds, exact-only
summaries and numbers of defined metric cases, zero-K cases, zero-incident cases,
oracle-unavailable cases and fallback exposure. Groups: representation, protocol,
fleet kind, unique support, installed copies (for v2 only), horizon, K choice,
method. No pooling/weighting by copy size or event count for a headline. Any
across-size summary weights the three v2 sizes equally; all seeds/dates equally.
Unique-cell tables retain all three horizons and each dataset/mixed separately.

For the main ranking comparison also save paired charge-SoH-minus-baseline
precision and recall differences, using only the windows on which each metric
is defined for both methods. Aggregate differences equally over dataset/mixed
and horizon groups. Report how many real positive source histories support
them. A descriptive advantage over all alternatives requires both paired
precision and recall improvements over cycle SoH, age and random; report LOCO
and LODO separately. Report RUL independently, without selecting a winner.
No significance test or confidence interval based on copies/repeated windows.
Nineteen total events, coarse capacity checks and incomplete follow-up prevent
an operational accuracy claim even if a descriptive ranking advantage appears.

## Outputs and interpretation

Save layouts for the unique-cell cohorts, held-out source audit, score/rank audit
with per-method order and fallback flags, all scenario/K/method metrics, grouped
comparison/coverage/source-support tables, paired comparisons and a run manifest
with preregistration commit/hash, input/code hashes and all pre-existing tracked
file hashes before/after. There are 7,200 v2 and 120 unique cohort/date/horizon/
protocol cases, each with five methods and both planned K modes, retaining
unavailable oracle cases rather than dropping them.

`results.md` starts with <=3 plain-English sentences: does primary charge SoH
beat cycle SoH, age and random; does the RUL alternative change the picture;
is it enough for a Baaz accuracy demo (versus a clearly labelled workflow demo)?
Then precision/recall tables for both K choices and both representations,
coverage/bounds, unique real/event support and limitations. One simple chart:
unique-cell historical-K conditional precision and recall by ranking method,
LOCO/LODO separately, equally macro-averaged over available kind/horizon groups;
label unknown-outcome conditioning, total real support and undefined cases.
Do not use oracle or resampled copies as the chart's primary evidence.
End with the exact reproduction command and a short Baaz data note: stable pack
IDs/types, commissioning dates, timestamped swap/throughput/charge logs,
replacement dates/reasons linked to IDs, independent periodic capacity checks,
complete 90-day follow-up and preregistered pack/type/calendar holdouts.
