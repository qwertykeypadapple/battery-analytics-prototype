"""Assemble every preregistered result, including negative findings."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import OUT,BASE,ROOT,baseline,data,metric_rows,verify_baseline


def table(frame,columns=None):
    return (frame[columns] if columns else frame).to_markdown(index=False,floatfmt='.3f')


def run():
    verify_baseline(); figs=OUT/'figures'; figs.mkdir(exist_ok=True)
    b=baseline(); b['experiment']='Frozen_baseline'
    allmetrics=[metric_rows(b)]; paired=[]
    variants=['E1_cycle','E2_replace','E2_alongside','E3_delta','E3_anchor','E4_combined','E5_filter',
              'E6_probabilistic','E7_fast','E8_forced_ica','E8_optional_ica']
    for name in variants:
        allmetrics.append(pd.read_csv(OUT/name/'metrics.csv'))
        p=pd.read_csv(OUT/name/'matched_baseline_metrics.csv'); p['for_experiment']=name; paired.append(p)
    m=pd.concat(allmetrics,ignore_index=True)
    m.to_csv(OUT/'all_metrics.csv',index=False)
    matched=pd.concat(paired,ignore_index=True); matched.to_csv(OUT/'all_matched_baselines.csv',index=False)
    a=m[m.level.eq('overall')&m.signal.eq('raw')].copy()
    match=matched[matched.level.eq('overall')&matched.signal.eq('raw')][['for_experiment','validation','cell_macro_mae_pp']].rename(
        columns={'for_experiment':'experiment','cell_macro_mae_pp':'matched_baseline_mae_pp'})
    a=a.merge(match,on=['experiment','validation'],how='left')
    a['delta_vs_matched_baseline_pp']=a.cell_macro_mae_pp-a.matched_baseline_mae_pp
    a=a.drop(columns=['dataset','cell_id','band'],errors='ignore')
    # E9 has repeated, changing target-cell calibration cohorts: aggregate each
    # repetition first, rather than treating repeated predictions as new cells.
    e9=pd.read_parquet(OUT/'E9_calibration/predictions.parquet')
    e9_overall=[]; e9_frozen=[]
    for (k,seed,phase),g in e9.groupby(['k','seed','phase']):
        from battery_analytics.evaluation import causal_median
        g=causal_median(g.copy()); g['experiment']=f'E9_k{k}_{phase}'
        z=metric_rows(g); z['k']=k; z['seed']=seed; z['phase']=phase; e9_overall.append(z)
        ref=b.merge(g[['validation','cycle_uid']],on=['validation','cycle_uid'],validate='1:1')
        ref['experiment']=f'E9_k{k}_{phase}'
        zref=metric_rows(ref); zref['k']=k; zref['seed']=seed; zref['phase']=phase
        e9_frozen.append(zref)
    er=pd.concat(e9_overall,ignore_index=True); er.to_csv(OUT/'E9_calibration/aggregate_metrics.csv',index=False)
    ea=er[er.level.eq('overall')&er.signal.eq('raw')].groupby(['experiment','validation']).agg(
        n_rows=('n_rows','mean'),n_cells=('n_cells','mean'),cell_macro_mae_pp=('cell_macro_mae_pp','mean'),
        cell_macro_rmse_pp=('cell_macro_rmse_pp','mean'),pooled_mae_pp=('pooled_mae_pp','mean'),
        pooled_rmse_pp=('pooled_rmse_pp','mean'),equal_dataset_mae_pp=('equal_dataset_mae_pp','mean'),
        repeat_sd_cell_macro_mae_pp=('cell_macro_mae_pp','std')).reset_index()
    ef=pd.concat(e9_frozen,ignore_index=True)
    ef.to_csv(OUT/'E9_calibration/matched_frozen_metrics.csv',index=False)
    ef_summary=ef[ef.level.eq('overall')&ef.signal.eq('raw')].groupby(['experiment','validation']).cell_macro_mae_pp.mean().rename('matched_baseline_mae_pp').reset_index()
    ea=ea.merge(ef_summary,on=['experiment','validation'])
    ea['delta_vs_matched_baseline_pp']=ea.cell_macro_mae_pp-ea.matched_baseline_mae_pp
    a=pd.concat([a,ea],ignore_index=True); a.to_csv(OUT/'ablation.csv',index=False)
    dataset=m[m.level.eq('dataset')&m.signal.eq('raw')]
    bands=m[m.level.eq('band')&m.signal.eq('raw')]
    bands.to_csv(OUT/'low_soh_bands.csv',index=False)
    e0=pd.read_csv(OUT/'E0/minimum_history_metrics.csv'); temp=pd.read_csv(OUT/'E0/nasa_low_soh_temperature.csv')
    intervals=pd.read_csv(OUT/'E5_filter/interval_calibration.csv')
    alerts=pd.read_csv(OUT/'E6_probabilistic/alert_metrics.csv')
    rc=pd.read_csv(OUT/'E6_probabilistic/rul_interval_calibration.csv')
    scales=pd.read_csv(OUT/'E6_probabilistic/rul_error_scale.csv')
    fast=pd.read_csv(OUT/'E7_fast/classification_metrics.csv')
    peak=pd.read_csv(OUT/'E8_peak_coverage.csv')
    peaks=peak.groupby('dataset').agg(n_rows=('n_rows','sum'),peaks=('peaks','sum')).reset_index()
    peaks['peak_coverage']=peaks.peaks/peaks.n_rows
    learning=pd.read_csv(OUT/'E9_calibration/learning_curve.csv')
    em=pd.read_csv(OUT/'E9_calibration/metrics.csv')
    zero=em[em.level.eq('dataset')&em.signal.eq('raw')&em.experiment.str.endswith('_matched_k0')]
    zero=zero.groupby(['phase','dataset','k']).cell_macro_mae_pp.mean().rename('matched_k0_mae_pp').reset_index()
    learning=learning.merge(zero,on=['phase','dataset','k'])
    frozen_learning=ef[ef.level.eq('dataset')&ef.signal.eq('raw')].groupby(['phase','dataset','k']).cell_macro_mae_pp.mean().rename('matched_frozen_mae_pp').reset_index()
    learning=learning.merge(frozen_learning,on=['phase','dataset','k'])
    learning['change_from_matched_k0_pp']=learning.mean_mae_pp-learning.matched_k0_mae_pp
    learning.to_csv(OUT/'E9_calibration/paired_learning_curve.csv',index=False)
    # Entry condition and left-censoring are evaluation strata, never predictors.
    d=data(); entries=d.groupby('group_id').head(1)[['group_id','soh']].rename(columns={'soh':'entry_soh'})
    eols=pd.read_csv(BASE/'end_of_life.csv'); eols['group_id']=eols.dataset+':'+eols.cell_id
    strata=[]
    for name in ['E3_delta','E3_anchor']:
        p=pd.read_parquet(OUT/name/'predictions.parquet').merge(entries,on='group_id').merge(eols[['group_id','censoring']],on='group_id')
        for (protocol,stratum_dataset,censor),h in p.groupby(['validation','dataset','censoring']):
            errors=pd.DataFrame({'cell':h.cell_id,'e':(h.prediction-h.soh).abs()*100})
            strata.append(dict(experiment=name,validation=protocol,dataset=stratum_dataset,censoring=censor,
                n_rows=len(h),n_cells=h.cell_id.nunique(),cell_macro_mae_pp=float(errors.groupby('cell').e.mean().mean()),
                median_entry_soh=float(h.groupby('cell_id').entry_soh.first().median())))
    strata=pd.DataFrame(strata); strata.to_csv(OUT/'E3_censoring_strata.csv',index=False)

    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,ds in zip(axes,['calce','nasa','oxford']):
        for phase,color,label in [('development','#00847e','Other unseen target cells'),('locked','#3976af','E9 locked cell')]:
            g=learning[learning.dataset.eq(ds)&learning.phase.eq(phase)]
            ax.errorbar(g.k,g.mean_mae_pp,yerr=g.sd_mae_pp,marker='o',color=color,label=label,capsize=4)
        ref=learning[learning.dataset.eq(ds)&learning.phase.eq('development')]
        ax.plot(ref.k,ref.matched_frozen_mae_pp,color='#555555',ls='--',label='Frozen LODO on same development cells')
        ax.set_title(ds.upper()); ax.set_xlabel('Target calibration cells (full trajectories)'); ax.set_ylabel('Cell-macro SoH MAE (pp)'); ax.set_xticks([0,1,2,4]); ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle('Source model frozen; offset-only calibration; bars = SD across 5 choices')
    fig.tight_layout(rect=[0,.10,1,.94]); fig.savefig(figs/'calibration_learning_curve.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); z=a[a.validation.eq('leave_one_cell_out')&~a.experiment.isin(['E6_probabilistic','E7_fast'])]
    xpos=np.arange(len(z)); base_values=z.matched_baseline_mae_pp.fillna(z.cell_macro_mae_pp)
    ax.bar(xpos-.19,base_values,width=.38,color='#777777',label='Frozen baseline on matched rows')
    ax.bar(xpos+.19,z.cell_macro_mae_pp,width=.38,color='#00847e',label='Variant on the same rows')
    ax.set_xticks(xpos,z.experiment,rotation=35,ha='right'); ax.legend()
    ax.set_ylabel('Cell-macro MAE (SoH pp)'); ax.set_title('LOCO ablation: compare each variant with its matched baseline'); fig.tight_layout(); fig.savefig(figs/'ablation.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,ds in zip(axes,['calce','nasa','oxford']):
        g=intervals[intervals.dataset.eq(ds)&intervals.validation.eq('leave_one_cell_out')]
        ax.plot(g.nominal,g.cell_macro_coverage,marker='o',color='#00847e'); ax.plot([0,1],[0,1],ls='--',color='#555555')
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title(ds.upper()); ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Actual cell-macro coverage'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figs/'interval_calibration.png',dpi=170); plt.close(fig)

    def loco(frame): return frame[frame.validation.eq('leave_one_cell_out')]
    def summary_table(frame):
        return table(frame,['experiment','validation','n_rows','n_cells','cell_macro_mae_pp','cell_macro_rmse_pp','pooled_mae_pp','pooled_rmse_pp','equal_dataset_mae_pp','matched_baseline_mae_pp','delta_vs_matched_baseline_pp'])
    text=f'''# Battery analytics: preregistered improvement results

**E1 answer: yes—the frozen charge-feature workflow beats the preregistered cycle-count-only curve on unseen cells, 3.72 versus 8.98 SoH percentage points cell-macro MAE.**

The simple charge-only baseline remains the recommendation. None of the added
tracking/feature pipelines establishes a robust improvement on a matched cohort;
probabilities are badly miscalibrated and short-horizon alerts remain weak.
Target calibration provides limited evidence for a paid validation pilot, not
out-of-the-box Baaz performance or validated replacement budgeting.

## Frozen reference and discipline

Tag `baseline-v1`; [hash manifest](baseline/manifest.json); [preregistration](preregistration.md).
Original source modules, prediction artifacts and metrics were verified unchanged.
Exact original 41 LOCO +3 LODO fold cell lists and inner splits were reused.
All new regressor fits are cell-balanced StandardScaler+Ridge(alpha1); E4/E8
representation selection occurs only inside the original inner cell split with
the preregistered 0.20-pp simpler-first rule. Noise and fleet priors use only
inner training cells; alert thresholds use only inner validation cells.

This is exploratory work on previously inspected public data. Cell separation
does not make the series an independent confirmatory trial. See all candidate
scores and provenance in each experiment's `audit.json`; no failed experiment
or difficult cell was removed after scoring. No hyperparameter changed after
outer-test results. [Deviations and implementation log](deviations.md).

## Ablation: primary and secondary metrics

Lower error is better. `delta_vs_matched_baseline_pp` >0 means WORSE on identical
rows. The frozen headline is 41 cells. K5 warmup variants score36; E3_anchor37.
Their lower unpaired averages cannot be advertised as gains over3.72. E6/E7
reuse E5 SoH, adding decisions rather than independent new SoH estimators.
E9 rows average five repeated choices (not independent cells); locked results
are separate. Missing matching fields are not a claim of full-cohort equivalence.

{summary_table(a)}

![Ablation](figures/ablation.png)

Full [metrics](all_metrics.csv), [matched baselines](all_matched_baselines.csv),
and per-experiment `coverage.csv` retain per-cell errors, raw/trailing-median,
row counts and unavailable cells. E9's complete metrics are in
[aggregate_metrics.csv](E9_calibration/aggregate_metrics.csv) and
[per-target/repetition metrics](E9_calibration/metrics.csv).

### Every variant by dataset

{table(dataset,['experiment','validation','dataset','n_cells','n_rows','cell_macro_mae_pp','cell_macro_rmse_pp','pooled_mae_pp'])}

### Low-health regions

Every variant and both protocols retain <70 and70–80% results in
[low_soh_bands.csv](low_soh_bands.csv); E9 bands are in its two complete metric
files above. Below are primary LOCO bands (sparse Oxford below70 is two points).

{table(loco(bands),['experiment','dataset','band','n_cells','n_rows','cell_macro_mae_pp','pooled_mae_pp'])}

## E0: post-hoc sensitivities, not replacement headlines

At least20 eligible observations:36cells,14,104rows; excludes five cells and six
observations from the41-cell baseline. LOCO MAE2.499pp vs3.720 unrestricted;
LODO12.307 vs13.887. [Excluded cells](E0/excluded_cells.csv).

{table(e0[e0.level.eq('overall')&e0.signal.eq('raw')],['validation','n_cells','n_rows','cell_macro_mae_pp','pooled_mae_pp','equal_dataset_mae_pp'])}

NASA below70 by fixed ambient groups:

{table(temp)}

Cold is worse than room temperature, supporting part of the hypothesis, but
warm tests are worse still in LOCO and contain only THREE points from two cells.
This contradicts an exclusive cold explanation. Available capacity depends on
test conditions; these observations cannot identify reversible temperature loss
versus permanent ageing. Never validate a Delhi pack from these strata alone.

## E1: the cycle-count comparison

The fixed quadratic cycle-only Ridge gives8.984pp LOCO and48.996pp LODO. The
original charge-only workflow is3.720/13.887. Per dataset LOCO: CALCE4.225 vs1.592,
NASA12.660 vs5.110, Oxford1.198 vs1.063. Thus the charge advantage is substantial
overall but marginal on Oxford. Some individual cycle-only cells are easy;
that is not a fleet-wide accuracy claim. Known cycle count is legitimate for a
pack tracked since commissioning; it remains excluded from the unknown-age
baseline. Polynomial extrapolation and incompatible source ageing timescales
make cycle-only transfer especially poor. No post-result clipping was added.

The original fixed Ridge itself achieved3.635pp LOCO and10.461pp LODO. E2's tiny
within-dataset changes should be assessed against that simple fixed model too,
not credited entirely to new physics because family selection also differs.

## E2: normalised throughput

Four local positive-current integrals /ratedAh replace or supplement duration.
No discharge quantity or full-charge capacity is used. Every eligible interval
was integrable; [fallback log](E2_fallbacks.csv) records the count explicitly.
Replacing times gives3.639pp LOCO /30.350pp LODO; alongside gives3.578/62.611.
**The primary transfer hypothesis fails.** Normalising units does not normalise
voltage-curve shape, chemistry, test current or protocol. Do not recommend this
as a universal transfer fix; the within-dataset gains are small.

## E3: entry-relative features and one capacity anchor

K5 entry medians use only charge features; they do NOT assume entry SoH=100%.
The near-new interpretation is unjustified for already degraded cells. The
first five charges are warmup, with explicit loss of five cells. Delta-only
LOCO4.344 vs matched baseline2.502; LODO7.578 vs12.244. This is a transfer gain
under a changed entry-history assumption with worse local accuracy.

Single commissioning-labelled offset: exactly the first eligible capacity is
allowed, then excluded from scoring. LOCO4.805 vs paired2.779; LODO16.973 vs11.967.
A one-time label cannot correct different ageing slopes or curve nonlinearity;
it can also transmit a noisy/temperature-dependent entry check to every later
estimate. Do not present it as an improvement. Full entry-health bands are in
[E3_entry_health_metrics.csv](E3_entry_health_metrics.csv).

Initially left-censored strata (full table includes all censoring types):

{table(strata[strata.censoring.str.startswith('left')])}

## E4: known-history combination

The original inner split chooses raw, throughput, alongside or delta features
plus cycle and squared-cycle; no test score chooses the representation. Same K5
cohort for every candidate. LOCO3.017 vs paired baseline2.502; LODO59.676 vs12.244.
It beats the simple cycle-count comparator locally, but adding age can damage
transfer severely. Original dataset cycle numbers are not comparable physical
throughput across these protocols. The baseline remains the default.

## E5: state-space tracking and interval calibration

State=SoH and rate, constant-velocity Gaussian dynamics, causal updates only.
Three-fold whole-cell cross-fits inside INNER TRAINING estimate observation
noise and select process noise; test references never initialise the tracker.
MAE3.190pp vs E4raw3.017, so smoothing is not an improvement. Saved metrics also
retain E4's trailing-five median comparison on identical rows.

{table(loco(intervals))}

![Coverage](figures/interval_calibration.png)

The95% cell-macro coverages are16.2% CALCE,81.7% NASA,54.0% Oxford. These are
**uncalibrated bands**, not95% reliability. Repeated model errors are biased and
serially correlated; a filter treating observations as independent can become
confident without becoming accurate. A future correlated-bias/conformal model
would require a NEW preregistration, not retuning these test cells.

## E6: probabilistic RUL and first alerts

512 fixed Sobol Gaussian posterior draws, nonpositive projected rate, linear
future paths. Positive-rate draws imply no crossing. No future acceleration
uncertainty is added, so this approximation must earn calibration empirically.
Infinite RUL upper bounds remain infinite, never silently clipped.

Horizon28/120/180 labcycles =14/60/90 scenario days. Probability thresholds are
selected per fold/horizon by predeclared inner-validation event costs, with
explicit abstention if the inner validation set has no observed endpoint.
These are laboratory capacity crossings, not stranding, safety or field failures.

{table(loco(alerts))}

At14days: probabilistic2/19 timely vs frozen0/19. At60days:6/19 vs6/19; at90days:
8/19 vs6/19. Fewer premature first alarms do not by themselves establish value:
late, none and no-history outcomes remain substantial. Denominators include all
19 observed endpoints. Oxford has no observation in either28-cycle event window.
See [event-level leads](E6_probabilistic/alert_events.csv),
[Brier scores](E6_probabilistic/brier.csv) and [risk reliability](E6_probabilistic/reliability.csv).

RUL interval calibration at the late-life anchors (all stages in linked CSV):

{table(loco(rc)[loco(rc).stage.eq('late')])}

### Arithmetic accuracy scale and feasible horizons

`RUL timing error ≈ SoH error / degradation rate` is a local error-propagation
diagnostic, **not a hard minimum achievable error**. Rates below use median
positive per-cell full-reference OLS degradation rates, with source-condition
confounding. They are descriptive evaluation quantities, not test-fitted priors.

{table(scales)}

LOCO filter timing scales are roughly85cycles/42days CALCE,20cycles/10days NASA,
329cycles/164days Oxford. A14-day CALCE or any <=90-day Oxford promise is not
supported by this accuracy/cadence;60–90-day CALCE budgeting is more plausible
to investigate but not validated. NASA's faster apparent decline makes short
timing errors look smaller, but temperature effects, missing histories and
bad calibration prevent calling even that operationally feasible today.

## E7: fast-degradation detection

Upper25% training-rate cutoff; decision at20th E4 observation (after K5 warmup).
True labels use the held-out full-reference slope only for evaluation. Cells
with fewer than20 E4 observations abstain; recall including abstentions is shown.

{table(fast)}

No CALCE true-fast cells are caught locally. NASA has one false flag and no true
catches; its flagged observed endpoint occurs later, not earlier, than the
unflagged median. Oxford catches1/3, too few cells for a product claim. Across
datasets, different cycle-age scales further undermine pooled rate comparisons.
Full descriptive fleet quartile membership and censoring remain in
[cell_flags.csv](E7_fast/cell_flags.csv).

## E8: incremental capacity

Fixed2mV grid and Savitzky–Golay11/poly2, prominence0.01, ten-bin peak separation;
peak voltage/height/±20mV area and explicit coverage flag. Missing peaks get
training-only median imputation, with all-missing columns explicit zero plus
the availability flag. A peak is a local numerical feature, not a validated
electrochemical attribution; the narrow upper window may truncate real peaks.

{table(peaks)}

Forced ICA3.173pp LOCO; optional inner-selected ICA3.042, vs E4without3.017.
LODO35.162/38.885 is still far worse than the frozen13.887. Optional inclusion
does not establish a gain; no post-test smoothing/window change was made.

## E9: target calibration learning curve

Source-only E2_replace Ridge stays frozen. Calibration is an additive offset
from k target cells' equal-cell mean residual. Five choices, k0/1/2/4. Critically,
**calibration uses complete labelled laboratory trajectories, not one entry
measurement or k independent capacity checks**. This is a substantial data
requirement for a pilot. The zero-cell E2 source model is worse than the original
baseline; reducing its bias does not automatically beat the original workflow.

{table(learning)}

![Calibration curve](figures/calibration_learning_curve.png)

Bars show SD across correlated choices, not independent-sample confidence
intervals. Test cohorts shrink with k: CALCE6→2, Oxford7→3, NASA25→21 development
cells. `matched_k0_mae_pp` explicitly compares the same evaluated cells.
One locked cell per target (CALCE CX2-16, NASA B0029, Oxford Cell1) is excluded
from every calibration/development choice. It was already present in earlier
public-data analyses: locked means locked for E9, not a fresh external trial.

CALCE and Oxford improve substantially over their uncalibrated E2 source model,
but after one cell the gains mostly plateau; NASA is unstable and k4 worsens.
Offset correction cannot fix shape/slope differences. Do not claim that four
packs suffice for Baaz or that error must fall monotonically with more packs.

## Separate fleet replacement-count experiment

[Fleet forecasting report](../fleet_forecast/results.md), preregistered before
running, uses original held-out raw-linear forecasts and E1 at30/60/90days.
Matched-subset charge count MAE0.367 batteries vs0.800 cycle and0.428 historical
rate; signed bias -0.033. This superficially encouraging mean rests on about
2–3 matched CALCE/NASA cells and zero actual Oxford incidents in the selected
matched windows. Full-fleet count bounds and missing forecasts remain visible.
Near-zero aggregate bias is not proof that individual errors cancel. This does
not validate replacement budgeting.

## What changed in the pitch claims

**Improved evidence:** charging features outperform the specific fixed
cycle-only quadratic baseline overall; model choices were preregistered and
cell-separated. Entry-relative features can reduce cross-dataset error under
a five-charge history assumption, and target-labelled offset calibration can
reduce source-model bias on CALCE/Oxford. These support a carefully scoped
calibration/validation pilot, with assumptions and failures stated.

**Stayed the same:** use the conservative frozen headline, about4SoH percentage
points on41unseen lab cells, alongside about14points for unseen datasets.
No evidence for Baaz pack accuracy, >45°C deployment, field-failure prevention,
energy savings, life extension or implemented federation. No positive ROI input
is justified by this series; operational intervention trials remain necessary.

**Got worse / must be qualified:** normalised throughput did not solve transfer;
adding cycle age hurt transfer; one-time anchoring hurt SoH accuracy; filters
added false confidence; E8 complexity did not earn its cost; fast-ageing flags
and short-horizon alerts remain weak. Do not put nominal95% uncertainty,
replacement dates or fleet budgets on a headline slide as validated accuracy.

**Recommendation:** keep the original explainable charge-only baseline and
held-out validation display as the default. Treat the E4/E5 tracker and E8
inner-selected candidate as labelled research views. Prefer a target-specific
pilot with independent capacity measurements and an unchanged BMS comparator
over more public-data model complexity. A future study should preregister
correlated error/bias calibration, heat/protocol holds and partial-window/noise
robustness, then test on new packs with complete operational outcomes.

## Dashboard and reproduction

`python scripts/run_improvements.py E0` through `E9`, one stage at a time;
`python scripts/run_improvements.py report`; `python scripts/fleet_forecast.py`;
`python -m pytest -q`; `python -m streamlit run app.py`.

The dashboard defaults to baseline, with a known-history E8 view and a separate
E5/E6 tracker view for its own uncalibrated bands/probabilities/fast-rate flag.
Covariance from one model is never grafted onto another. All views use held-out
predictions. Interactive controls do not change saved validation experiments.
'''
    (OUT/'results.md').write_text(text)
    from .summary import sync_main_results
    sync_main_results()
    print('Improvement report, ablation and scientific plots saved:',OUT/'results.md')
