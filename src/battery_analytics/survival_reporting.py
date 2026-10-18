"""Presentation of every fixed survival method, without fitting or selection."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .fleet_v2 import PROTOCOLS
from .survival import SURVIVAL_METHODS

LABELS = dict(km='Kaplan–Meier', hazard='Hazard: 2 inputs', hazard_known_history='Hazard + cycles',
              cox='Time-varying Cox', linear='Frozen linear', e6='Frozen E6')


def number(x, digits=3):
    return f'{x:.{digits}f}' if pd.notna(x) and np.isfinite(x) else 'Unscored'


def calibration_plot(calibration, out):
    colors = dict(km='#7b6d58', hazard='#00877c', cox='#bf6f24', e6='#626cba')
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
    g = calibration[calibration.dataset.eq('all')]
    for i, protocol in enumerate(PROTOCOLS):
        for j, days in enumerate([30,60,90]):
            ax=axes[i,j]
            ax.plot([0,1],[0,1],ls='--',color='#aaaaaa',lw=1)
            for method,color in colors.items():
                a=g[g.validation.eq(protocol)&g.days.eq(days)&g.method.eq(method)]
                a=a[a.predicted_mean.notna()&a.observed_frequency.notna()]
                for enough, marker in [(True,'o'),(False,'o')]:
                    b=a[a.sufficiently_populated.eq(enough)]
                    ax.scatter(b.predicted_mean,b.observed_frequency,marker=marker,s=45,
                               facecolors=color if enough else 'none',edgecolors=color,
                               alpha=1 if enough else .45,
                               label=LABELS[method] if enough else None)
            ax.set_title(f'{"LOCO" if i==0 else "LODO"} / {days} scenario days')
            ax.set_xlim(-.03,1.03);ax.set_ylim(-.03,1.03);ax.grid(alpha=.15)
            ax.set_xlabel('Predicted crossing probability')
            ax.set_ylabel('IPCW observed frequency')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor=c,label=LABELS[m]) for m,c in colors.items()]
    handles.append(Line2D([0],[0],marker='o',color='none',markerfacecolor='none',markeredgecolor='#888888',label='Hollow = sparse bin'))
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('Held-out survival calibration; repeated checkpoints are not new batteries\n'
                 'Filled bins: at least 5 real cells and 2 incident-source cells; all others sparse',fontsize=12)
    fig.tight_layout(rect=[0,.055,1,.92])
    fig.savefig(out/'calibration.png',dpi=170)
    plt.close(fig)


def report(root,out):
    probability=pd.read_csv(out/'probability_metrics.csv')
    paired=pd.read_csv(out/'paired_brier.csv')
    calibration=pd.read_csv(out/'calibration.csv')
    alerts=pd.read_csv(out/'alert_metrics.csv')
    censored=pd.read_csv(out/'censored_alert_metrics.csv')
    concordance=pd.read_csv(out/'concordance_metrics.csv')
    fleet=pd.read_csv(out/'fleet_metrics.csv')
    fleet_cases=pd.read_csv(out/'fleet_scenarios.csv')
    hybrid=pd.read_csv(out/'hybrid_metrics.csv')
    hybrid_pairs=pd.read_csv(out/'hybrid_paired.csv')
    models=pd.read_csv(out/'model_audit.csv')
    sources=pd.read_csv(out/'source_inventory.csv')
    execution=json.loads((out/'execution_start.json').read_text())
    calibration_plot(calibration,out)
    all_alerts=alerts[alerts.cohort.eq('all_19_events')&alerts.dataset.eq('all')]
    at14=all_alerts[all_alerts.validation.eq(PROTOCOLS[0])&all_alerts.days.eq(14)].set_index('method')
    timing_better=at14.loc['hazard','timely']>max(at14.loc['linear','timely'],at14.loc['e6','timely'])
    delta=paired[paired.baseline.eq('e6')].groupby('validation').hazard_minus_baseline.mean()
    probability_message=('Its matched probability errors are worse than E6 within datasets and better on unseen datasets'
                         if delta.get(PROTOCOLS[0],np.nan)>0 and delta.get(PROTOCOLS[1],np.nan)<0 else
                         'Its matched probability errors do not show a consistent improvement over E6')
    opening=(f'The primary survival model {"improves" if timing_better else "does not improve"} the within-dataset 14-day first-warning count: '
             f'{int(at14.loc["hazard","timely"])}/19 timely warnings versus {int(at14.loc["linear","timely"])}/19 for linear extrapolation '
             f'and {int(at14.loc["e6","timely"])}/19 for E6. '
             + probability_message+'; the calibration checks remain limited by sparse real-cell/event support and censoring. '
             'This is an indicative laboratory workflow demo for Baaz, not accuracy proof: only 19 source events exist, 17 among the 41 cells with frozen predictions.')
    prob_rows=[]
    for key,g in probability.groupby(['validation','dataset','method']):
        r=dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Dataset=key[1],Method=LABELS[key[2]])
        for days in [30,60,90]:
            a=g[g.days.eq(days)].iloc[0]
            r[f'{days}d IPCW Brier']=number(a.ipcw_cell_macro_brier)
            r[f'{days}d cells / event sources']=f'{int(a.real_cells_scored)} / {int(a.real_event_sources)}'
        prob_rows.append(r)
    pair_rows=[]
    for key,g in paired.groupby(['validation','days','baseline']):
        pair_rows.append(dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Days=key[1],Baseline=LABELS[key[2]],
            **{'Common real cells':int(g.common_real_cells.sum()),'Hazard Brier':number(g.hazard_brier.mean()),
               'Baseline Brier':number(g.baseline_brier.mean()),'Hazard minus baseline':number(g.hazard_minus_baseline.mean())}))
    concord_rows=[]
    for key,g in concordance.groupby(['validation','dataset','method']):
        r=dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Dataset=key[1],Method=LABELS[key[2]])
        for days in [30,60,90]:
            a=g[g.days.eq(days)].iloc[0]
            r[f'{days}d C']=number(a.c_index)
            r[f'{days}d pairs / event sources']=f'{int(a.comparable_pairs)} / {int(a.earlier_event_sources)}'
        concord_rows.append(r)
    alert_display=all_alerts[['validation','method','days','real_events','timely','premature','late','none','no_history','timely_opportunities']].copy()
    alert_display['method']=alert_display.method.map(LABELS)
    censor_rows=[]
    for key,g in censored.groupby(['validation','method']):
        r=dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Method=LABELS[key[1]])
        for days in [14,60,90]:
            a=g[g.days.eq(days)]
            counts=[int(a[c].sum()) if c in a else 0 for c in ['verified_false_alert','censor_unresolved','none']]
            r[f'{days}d false / unresolved / none']=' / '.join(map(str,counts))
        censor_rows.append(r)
    fleet_rows=[]
    for key,g in fleet.groupby(['validation','kind','method']):
        fleet_rows.append(dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Fleet=key[1],Method=LABELS[key[2]],
            **{'Real source range':f'{int(g.real_sources_min.min())}–{int(g.real_sources_max.max())}',
               'Mean matched real sources':g.mean_matched_unique.mean(),
               'Missing probabilities %':100*(g.mean_missing/g.mean_at_risk).mean(),
               'Model count MAE':g.model_mae.mean(),'Historical count MAE':g.historical_mae.mean(),
               'Model signed bias':g.model_signed_bias.mean()}))
    bounds_rows=[]
    for key,g in fleet_cases[fleet_cases.kind.eq('mixed')&fleet_cases.nominal.eq(1000)&fleet_cases.days.eq(60)].groupby(['validation','method']):
        bounds_rows.append(dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',Method=LABELS[key[1]],
            **{'Actual bounds':f'{g.actual_lower.mean():.1f}–{g.actual_upper.mean():.1f}',
               'Predicted bounds':f'{g.probability_subtotal.mean():.1f}–{g.probability_upper.mean():.1f}',
               'Missing / at risk':f'{g.probability_missing.mean():.1f} / {g.at_risk.mean():.1f}',
               'Full signed-error bounds':f'[{g.full_signed_error_lower.mean():+.1f}, {g.full_signed_error_upper.mean():+.1f}]'}))
    hybrid_rows=[]
    for key,g in hybrid[hybrid.representation.eq('unique')].groupby(['validation','k_mode','method']):
        hybrid_rows.append(dict(Protocol='LOCO' if key[0]==PROTOCOLS[0] else 'LODO',K=key[1],
            Method=LABELS.get(key[2],key[2]),**{'Conditional precision %':100*g.precision.mean(),
            'Known-incident recall %':100*g.recall.mean(),'Defined precision windows':int(g.precision_cases.sum()),
            'Defined recall windows':int(g.recall_cases.sum())}))
    hp=hybrid_pairs[hybrid_pairs.representation.eq('unique')&hybrid_pairs.k_mode.eq('historical')&hybrid_pairs.method.eq('hazard')]
    hp=hp.groupby(['validation','baseline','metric'],as_index=False).agg(difference=('difference','mean'),common_window_records=('common_cases','sum'))
    hp['difference_pp']=hp.difference*100
    hp=hp.drop(columns='difference')
    model_table=models[models.context.eq('outer')].groupby(['validation','method','model_status']).agg(
        folds=('fold','count'),min_training_events=('training_observed_events','min'),max_training_events=('training_observed_events','max'),
        min_period_events=('period_events','min'),max_period_events=('period_events','max')).reset_index()
    source_table=sources.groupby(['dataset','censoring']).agg(real_cells=('group_id','nunique'),
        with_frozen_predictions=('has_frozen_predictions','sum')).reset_index()
    periods=pd.read_parquet(out/'training_person_periods.parquet')
    censor_ids=set(sources.loc[sources.censoring.eq('right_censored'),'group_id'])
    outer_periods=periods[periods.context.eq('outer')]
    right_cox=len(set(outer_periods.group_id)&censor_ids)
    right_hazard=len(set(outer_periods.loc[outer_periods.logistic_eligible,'group_id'])&censor_ids)
    fleet_macro=fleet.groupby(['validation','method'])[['model_mae','historical_mae']].mean()
    enough=calibration[calibration.dataset.eq('all')].groupby(['validation','method','days']).agg(
        sufficiently_populated_bins=('sufficiently_populated','sum'),bins=('bin','count')).reset_index()
    enough.to_csv(out/'calibration_support.csv',index=False)
    text=f'''{opening}

# Survival analysis for replacement timing

## Frozen design, data and leakage controls

Preregistration `{execution['preregistration_commit']}` was committed before
fitting. No outer-test results selected inputs, penalties, windows, models,
threshold grids or age mixtures. All pre-existing tracked files remain unchanged.
The endpoint is exactly the frozen RUL study's first observed reference 70%
crossing **interpolated** between valid capacity checks, not a field replacement.
One scenario day is two source cycles; these are not measured operator days.

{source_table.to_markdown(index=False)}

The complete inventory is **49 real cells / 19 observed crossings**. Frozen
predictions exist for 41 cells; removing their 10 initially below-threshold
cells leaves **31 incident-modelling cells / 17 observed events / 14 right
censors**. The other two crossing cells have no eligible frozen prediction
history. Initial-below cells are excluded from incident fitting and probability
validation, and are separately identified in [source_inventory.csv](source_inventory.csv).
The alert table retains all 19 events, including missing-history failures.

The same 44 outer folds are used: LOCO within dataset and LODO with an entire
dataset excluded. Outer test features are the original frozen held-out raw
charge estimates. Survival-training charge estimates are generated by two-fold
cross-fitting **inside** each training pool, so their upstream charge models,
scalers and model-family selection cannot see the outer test cell/dataset or
the predicted training cell. Inner alert-threshold validation is separately
cross-fitted inside its inner-training pool. Frozen charge model specifications
are reused; no frozen model/artifact is overwritten. Audits record every pool.

## What was fitted

- **Kaplan–Meier:** one event/censor record per training cell, delayed entry at
  first valid reference, population risk conditional on survival through today.
- **Primary discrete hazard:** L2 logistic regression, C=1, on 10-cycle
  person-periods, exactly two inputs: latest raw predicted SoH and minus the
  OLS slope of the last 12 distinct predicted points (minimum six).
- **Known-history hazard:** the same model plus source cycle count, clearly
  separate from the two-input primary model.
- **Time-varying Cox:** start/stop risk sets, Breslow baseline and ties, fixed
  L2 alpha=1 on event-normalised partial likelihood, the same two inputs.

Each cell contributes at most one event; right-censored cells contribute
observed at-risk exposure. Logistic drops a partly observed terminal non-event
bucket, while Cox retains its exact exposure. Events before feature entry are
lost to feature-based fitting and reported in the period audits. Future
covariates are held at current values for a forecast. Last observed inputs are
carried forward even after the last charge record, unlike v2's ended-record
abstention; age/staleness is recorded but is not an extra primary predictor.
KM/Cox probabilities are unavailable beyond their training time support, and
no alternative fit is substituted for failed models. No Weibull AFT was run.

All {right_hazard} right-censored source histories contribute to at least one
outer logistic training fit, and all {right_cox} to Cox exposure across folds;
they are used as censored histories, not discarded or labelled as eventual
non-events. This counts real histories across fits, not that many in each fold.

Outer-fold model/convergence and event-exposure audit (inner folds also saved):

{model_table.to_markdown(index=False)}

The period-event columns describe availability of the common feature design.
KM uses all its training endpoint records and the training-event columns, without
the six-charge entry restriction. One inner fold had no feature-period events;
its feature models used the preregistered flagged zero-risk rule. Every outer
fold fit converged; no alternative hyperparameters were tried.

## Probability accuracy at 30/60/90 days

IPCW Brier is lower when probability predictions are better. It uses only the
corresponding outer-training cells' censoring distribution, conditional on
survival to the native charge checkpoint, and averages first within real cell,
then across cells. A test cell censored before a horizon has zero weighted
contribution and remains in the denominator; it is **not labelled healthy**.
Relevant conditional censor survival below .05 or beyond training support is
unscorable, not clipped. Independent censoring is an assumption, especially
questionable across lab protocols. Coverage is part of the result.

“Cells / event sources” counts unique scored cells and sources with an observed
endpoint, not checkpoints. Methods may have different supported populations,
so use the matched table for comparisons. Frozen linear is a deterministic 0/1
RUL-within-horizon indicator, not a calibrated probability. E6's unchanged fixed
posterior algorithm supplies a new 60-cycle horizon; its stored 28/120/180 risks
were checked exactly and no E6 model or threshold was refitted.

{pd.DataFrame(prob_rows).to_markdown(index=False)}

Matched two-input hazard versus each comparator: same cell/checkpoint/censoring
support, cell-macro Brier, then equal-dataset average within each horizon.
Negative difference favours the primary hazard. Counts sum disjoint datasets
within a horizon; repeated horizons are not new cells/events.

{pd.DataFrame(pair_rows).to_markdown(index=False)}

Against E6, the primary hazard's equal-dataset/horizon matched Brier difference
is {delta.get(PROTOCOLS[0],np.nan):+.3f} within datasets (worse) and
{delta.get(PROTOCOLS[1],np.nan):+.3f} on unseen datasets (better). This does not
establish calibrated probabilities: the calibration plot shows substantial
departures from the diagonal, and the supported populations differ by protocol.

All raw known-outcome conditional Brier, row-weighted scores, availability,
right-censor contribution counts and per-dataset matched comparisons are in
[probability_metrics.csv](probability_metrics.csv), [paired_brier.csv](paired_brier.csv)
and [probability_scores.parquet](probability_scores.parquet). A score reduction
does not by itself establish calibration or reliable decisions.

## Calibration and sparse-bin check

Fixed probability bins are 0–20–40–60–80–100%. Cell-balanced IPCW observed
frequencies and predicted means use the same weights within each bin. Filled
points require at least five real cells and two distinct horizon-incident
sources; hollow points are sparse, including zero-event bins. No recalibration
or confidence interval treating repeated checkpoints as independent is fitted.
Even the populated bins have very little independent support and cannot
validate a “30% means 30%” claim on another battery type.

![Held-out calibration with sparse bins marked](calibration.png)

All six methods, both protocols, datasets and horizons, including bins with no
usable observations, are in [calibration.csv](calibration.csv). The plot displays
KM, primary hazard, Cox and E6; the cycle-input variant and deterministic linear
bins are retained in that table. [calibration_support.csv](calibration_support.csv)
counts populated versus sparse bins using real cells/events.

## Time-dependent ranking

C(H) is the preregistered **event-landmark concordance**: today=T_i−H for an
earlier observed event i; comparator j must be known to survive beyond T_i.
Compare their held-out as-of H risks, with half credit for ties and training-only
inverse-censoring pair weights. It is a specifically defined dynamic statistic,
not a static Harrell C-index. Early events lacking an H-long prefix, missing
features and unsupported censoring/time tails are excluded and counted. E6 uses
the latest past state here, with no future state. Larger C is better; .5 is the
tie/random reference. Cell pairs and repeated horizons are correlated.

{pd.DataFrame(concord_rows).to_markdown(index=False)}

[concordance_pairs.csv](concordance_pairs.csv) records all candidate/usable pairs,
weights and source IDs; [paired_concordance.csv](paired_concordance.csv) restricts
comparisons to the exact same pairs. Some dataset/horizon combinations have no
comparable earlier events: an unscored C is not a successful ranking result.

## First-alert decision backtest: 14/60/90 days

Survival thresholds are selected only on the frozen inner-validation cells from
the preregistered grid/cost. The **first** alarm determines timely/premature/
late/none, with late alarms retained after the endpoint as in the original
blind-history backtest. A later timely alarm does not erase an earlier bad one.
All-19-event results follow. `no_history` is included in `none`; timely
opportunities are method-specific available checks, not a filtered denominator.
Frozen linear and E6 first-alert timestamps/categories match the earlier saved
event records; neither comparator was retuned.

{alert_display.to_markdown(index=False)}

The primary hazard has fewer timely first alerts than E6 at all three within-
dataset horizons, and fewer than linear at 60/90 days. On unseen datasets it
catches two events at 60/90 days versus linear's four and E6's zero. The separate
known-history variant catches seven/six unseen-dataset events at 60/90 days,
versus linear's four/four, but has no timely unseen 14-day alerts; this small
positive result does not rescue the preregistered two-input primary model.

Right-censored cells cannot be assumed to have no future event. A first alarm is
a verified false alert only if its full horizon is observed without a crossing;
otherwise it is unresolved. The fixed threshold-selection cost assigns zero
to unresolved censored alarms, as E6 did; that permissive cost is a limitation,
not proof such alarms are harmless.

{pd.DataFrame(censor_rows).to_markdown(index=False)}

Per-dataset counts, the separate 17-event eligible-source cohort and every
cell's lead/threshold/history are in [alert_metrics.csv](alert_metrics.csv),
[alert_events.csv](alert_events.csv) and [censored_alerts.csv](censored_alerts.csv).
[threshold_audit.json](threshold_audit.json) retains the inner candidates/costs
and training/selection/outer-test IDs, including abstention thresholds.

## Fleet expected counts on the unchanged v2 layouts

Each held-out source risk is copied to its v2 commissioning age and summed over
the at-risk pool for 30/60/90 days, across all fixed seeds/todays/sizes. These are
expected-count **subtotals** when probabilities are missing. For each method,
matched count MAE/bias compares it with the historical rate on precisely its own
known-outcome/usable-probability subset. Historical errors differ between rows
because those subsets differ; they are not an unqualified contest across models.
The following averages weight sizes/horizons equally; detailed rows retain each.

{pd.DataFrame(fleet_rows).to_markdown(index=False,floatfmt='.3f')}

The two-input hazard's matched count MAE is
{fleet_macro.loc[(PROTOCOLS[0],'hazard'),'model_mae']:.2f} versus historical
{fleet_macro.loc[(PROTOCOLS[0],'hazard'),'historical_mae']:.2f} within datasets,
but {fleet_macro.loc[(PROTOCOLS[1],'hazard'),'model_mae']:.2f} versus historical
{fleet_macro.loc[(PROTOCOLS[1],'hazard'),'historical_mae']:.2f} when a dataset is
unseen. Historical rate wins the aggregate unseen-dataset comparison for every
survival method. The cycle-input hazard is particularly poor there: count MAE
{fleet_macro.loc[(PROTOCOLS[1],'hazard_known_history'),'model_mae']:.2f}; its alert
timing gain therefore does not imply useful replacement budgeting.

For the pre-existing v2 1,000-copy mixed fleet / 60-day scenario, means of full
prediction/truth bounds show what the matched subset omits:

{pd.DataFrame(bounds_rows).to_markdown(index=False)}

These are missing-data bounds, not probability confidence intervals. A 1,000-copy
CALCE fleet still has only **seven real degradation histories**, and mixed fleets
have at most 49 inventory sources, only 41 with frozen predictions. Probability
summation does not require independent events for its arithmetic, but copies do
not validate probabilities or accuracy at 1,000-pack scale. See
[fleet_scenarios.csv](fleet_scenarios.csv), [fleet_metrics.csv](fleet_metrics.csv)
and [fleet_probabilities.parquet](fleet_probabilities.parquet) for all count,
coverage, bias, unique-support and full error-bound records.

## Hybrid ranking with survival risks

The four survival probabilities rank candidates highest first at each horizon,
using the original hybrid's historical K, half-up rounding, tie keys and
finite-first/missing-oldest fallback. Candidate pools, truth, unique-cell offsets
and oracle availability are identical. Missing risks never silently disappear.
Below are equal kind/horizon macro averages on **unique-cell** cohorts; conditional
precision/recall omit undefined ratios, with counts shown. Oracle K is future
information and unavailable when any candidate outcome is unknown.

{pd.DataFrame(hybrid_rows).to_markdown(index=False,floatfmt='.2f')}

Primary hazard minus the saved hybrid rankers on mutually defined unique-cell
historical-K windows (percentage points):

{hp.to_markdown(index=False,floatfmt='.2f')}

Survival probabilities do not establish a better top-K policy: within datasets,
the primary hazard loses known-incident recall to the prior charge-SoH ranking;
on unseen datasets it has no recall gain over age and loses recall to random.

Only **four distinct incident histories** occur in the prior hybrid's fixed
unique-cell windows; all 19 appear somewhere among its resampled scenarios.
CALCE historical K rounds to zero throughout the unique version, NASA's longer
horizons flag all at-risk members, and Oxford has no unique incident windows.
Survival ranking cannot repair a zero or indiscriminate count budget. Oracle
results cannot establish an operational gain. All four methods, both K choices,
both protocols, each dataset/mixed fleet, every v2 size and their conditional/
full-bound metrics are in [hybrid_metrics.csv](hybrid_metrics.csv),
[hybrid_scenarios.csv](hybrid_scenarios.csv) and [hybrid_paired.csv](hybrid_paired.csv).

## Limits and data needed from Baaz

There are only **19 full-inventory events, 17 among the 41 prediction cells**,
and just 31 eligible incident-modelling cells after initial-below exclusions.
This is very little evidence: every result is **indicative, not proof**.
LOCO/LODO models share training histories; records, pairs, windows and copies
do not increase independent support. Sparse calibration, charge-feature entry,
held covariates, fixed 10-cycle buckets, unsupported Cox/KM tails, censoring
assumptions and chemistry/protocol transfer all limit interpretation. A labelled
workflow demo can explain these outputs; it cannot promise accurate Baaz
replacement warnings, budgeting, targeting or savings.

Baaz would need stable **pack IDs/types**, original **commissioning dates** and
refurbishment/exit history; dated permanent **replacement records with reasons**
and successor IDs; periodic independent **capacity checks** with rated reference,
Ah/Wh, temperature, current, cutoffs and QC; and linked timestamped **swap/charge
logs** measuring SoC, throughput and temperature. Obtain complete follow-up
through 90 days after the final today, record censoring and competing removals,
and enough unique packs and independently observed replacement events per type.
Freeze pack/type/calendar holdouts and thresholds before outcomes, with any
target-calibration packs separate from evaluation.

## Reproduction and integrity

Run `.venv/bin/python scripts/survival.py` from the repository root, or use
`--stage fit`, `--stage evaluate`, and `--stage report` in order. The fit stage
uses only its registered training pools; evaluate/report reuse those saved fits
without tuning. [preregistration.md](preregistration.md),
[execution_start.json](execution_start.json) and [run_manifest.json](run_manifest.json)
record the pre-fit commit, timestamps, frozen inputs, code and before/after
protected-file hashes. [nested_charge_audit.json](nested_charge_audit.json),
[nested_charge_predictions.parquet](nested_charge_predictions.parquet),
[training_person_periods.parquet](training_person_periods.parquet),
[model_audit.json](model_audit.json), [inner_probabilities.parquet](inner_probabilities.parquet)
and [outer_probabilities.parquet](outer_probabilities.parquet) expose the full
cross-fitting, exposure, coefficient, threshold and prediction trail.
`models.joblib` contains held-out fold models, not a new production all-cell refit.

The implementation follows the declared [training-only IPCW Brier framework](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html)
and its [censoring-support restrictions](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html);
the Cox design uses [time-varying start/stop risk sets](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html).
The event-landmark concordance definition here is explicitly study-specific.
'''
    (out/'results.md').write_text(text)
    print('Saved',out/'results.md',flush=True)
