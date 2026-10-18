"""Run only the preregistered v2 study; leave all pre-existing artifacts intact."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT / 'src'))

from datetime import datetime, timezone
import hashlib
import json
import subprocess
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from battery_analytics.improvements.core import baseline, verify_baseline, BASE
from battery_analytics.rul import reference_histories
from battery_analytics.fleet_v2 import (
    KINDS, DATASETS, SIZES, SEEDS, TODAYS, HORIZONS, PROTOCOLS, AGE_SPANS, METHODS,
    make_layout, audit_predictions, prefix_crossings, lookup_forecasts,
    reference_asof, scenario, summarize,
)

OUT = ROOT / 'reports/fleet_forecast_v2'
PREREG = 'reports/fleet_forecast_v2/preregistration.md'
INPUTS = [
    'reports/improvements/baseline/predictions.parquet',
    'reports/improvements/baseline/end_of_life.csv',
    'reports/improvements/baseline/fold_audit.json',
    'reports/improvements/baseline/manifest.json',
    'reports/improvements/E1_cycle/predictions.parquet',
    'reports/improvements/E1_cycle/audit.json',
    'reports/improvements/E6_probabilistic/reliability.csv',
    'reports/improvements/E6_probabilistic/rul_interval_calibration.csv',
    'reports/inventory/cells.csv', 'data/processed/capacity_references.parquet',
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def preregistration_commit():
    commit = git('log', '--diff-filter=A', '--format=%H', '--', PREREG).decode().strip()
    assert len(commit.splitlines()) == 1 and len(commit) == 40, 'Commit preregistration before running'
    assert git('show', f'{commit}:{PREREG}') == (ROOT / PREREG).read_bytes(), 'Preregistration changed after commit'
    return commit


def build_copies(layout, references, endpoints):
    parts = []
    for today in TODAYS:
        g = layout.copy()
        g['today'] = today
        g['source_age_cycles'] = 2 * (today - g.commissioning_day)
        truth_parts = []
        for gid, h in g.groupby('group_id', sort=False):
            truth = reference_asof(references[gid], endpoints.loc[gid], h.source_age_cycles.to_numpy())
            truth.index = h.index
            truth_parts.append(truth)
        parts.append(pd.concat([g, pd.concat(truth_parts).sort_index()], axis=1))
    return pd.concat(parts, ignore_index=True)


def make_plot(s):
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    colors = {'actual': '#202329', 'charge': '#00877c', 'historical': '#3378bb', 'cycle': '#ca692f'}
    for i, kind in enumerate(KINDS):
        for j, protocol in enumerate(PROTOCOLS):
            ax = axes[i, j]
            g = s[s.kind.eq(kind) & s.validation.eq(protocol) & s.nominal.eq(1000) & s.days.eq(60)]
            means = g.groupby('today')[['matched_actual', 'matched_charge', 'matched_historical', 'matched_cycle']].mean()
            for name in ('actual', *METHODS):
                field = f'matched_{name}'
                ax.plot(means.index, means[field], marker='o', markersize=4,
                        color=colors[name], label='Actual' if name == 'actual' else name.capitalize())
            ax.set_title(f'{kind.upper()} / {"within dataset" if j == 0 else "unseen dataset"}\n'
                         f'mean matched copies {g.matched_n.mean():.0f}; unique matched sources {g.matched_unique.mean():.1f}')
            if not g.matched_n.gt(0).any():
                ax.text(.5, .5, 'Unscored: no paired usable forecasts', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=11)
                ax.set_ylim(0, 1)
            ax.set_ylabel('60-day replacement count')
            ax.set_xlabel('Today: scenario day')
            ax.set_xticks(TODAYS)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('Paired known-outcome subset of 1,000-copy cohorts; all 20 seeds retained\n'
                 'Empty cases unscored; copies do not add independent cells', fontsize=13)
    fig.tight_layout(rect=[0, .035, 1, .94])
    fig.savefig(OUT / 'replacement_counts.png', dpi=160)
    plt.close(fig)


def write_report(s, metrics, coverage, reasons, support, source_audit, commit):
    # Report-only aggregation: no changes to fleet design, models or scoring.
    def comparison_table(protocol):
        g = metrics[metrics.validation.eq(protocol)]
        rows = []
        for key, h in g.groupby(['kind', 'nominal', 'days']):
            m = h.set_index('method')
            cases = s[s.validation.eq(protocol) & s.kind.eq(key[0]) & s.nominal.eq(key[1]) & s.days.eq(key[2])]
            scored = cases[cases.matched_n.gt(0)]
            r = dict(Fleet=key[0], N=key[1], Days=key[2],
                     **{'Unique sources': f'{h.min_unique_sources.iloc[0]:.0f}–{h.max_unique_sources.iloc[0]:.0f}',
                        'Scored / 100': len(scored),
                        'Matched copies / sources': f'{scored.matched_n.mean():.1f} / {scored.matched_unique.mean():.1f}' if len(scored) else 'Unscored',
                        'Actual': f'{scored.matched_actual.mean():.2f}' if len(scored) else 'Unscored'})
            for method in METHODS:
                r[method.capitalize() + ' predicted / MAE / bias'] = (
                    f'{scored["matched_"+method].mean():.2f} / {m.loc[method, "mae_count"]:.2f} / {m.loc[method, "signed_bias_count"]:+.2f}'
                    if len(scored) else 'Unscored')
            rows.append(r)
        return pd.DataFrame(rows).to_markdown(index=False, floatfmt='.2f')

    overall = metrics.groupby(['validation', 'method'])[['mae_count', 'signed_bias_count', 'mae_per100_installed', 'mae_per100_matched']].mean().reset_index()
    norm = metrics.groupby(['validation', 'kind', 'method'])[['mae_per100_matched', 'bias_per100_matched']].mean().reset_index()
    norm_display = norm.copy()
    for field in ('mae_per100_matched', 'bias_per100_matched'):
        norm_display[field] = norm[field].map(lambda x: f'{x:.3f}' if pd.notna(x) else 'Unscored')
    headline = overall.pivot(index='validation', columns='method', values='mae_count')
    within, unseen = headline.loc[PROTOCOLS[0]], headline.loc[PROTOCOLS[1]]
    within_win = within.charge < min(within.historical, within.cycle)
    unseen_win = unseen.charge < min(unseen.historical, unseen.cycle)
    opening = (
        ('On the subset with usable forecasts and known outcomes, charge-based counts beat both simple baselines on average in the larger, mixed-age cohorts'
         if within_win else 'Charge-based counts do not beat both simple baselines on average in the larger, mixed-age cohorts')
        + f' (within-dataset count MAE {within.charge:.2f}, historical {within.historical:.2f}, cycle {within.cycle:.2f}); missing coverage prevents a full-fleet accuracy claim. '
        + ('They also lead on the unseen-dataset matched subset' if unseen_win else 'Historical rate still wins overall on unseen battery types')
        + f' (MAE: charge {unseen.charge:.2f}, historical {unseen.historical:.2f}, cycle {unseen.cycle:.2f}), although charge narrowly wins in mixed fleets on a very small covered subset. '
        + 'Baaz would need real pack commissioning dates, swap logs, replacement records and periodic independent capacity checks on many held-out packs, with at least 90 days of follow-up.'
    )
    covrows = []
    for key, g in coverage.groupby(['validation', 'kind', 'nominal']):
        covrows.append(dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1], N=key[2],
                            **{'At risk': g.at_risk.mean(), 'Already below': g.already_below.mean(),
                               'Charge missing': g.charge_unknown.mean(), 'Cycle missing': g.cycle_unknown.mean(),
                               'Charge overdue': g.charge_overdue.mean(),
                               'At-risk unique': g.at_risk_unique.mean(), 'Charge usable unique': g.charge_usable_unique.mean()}))
    covtable = pd.DataFrame(covrows)
    covtable.to_csv(OUT / 'coverage_summary.csv', index=False)
    boundsrows = []
    for key, g in s[s.nominal.eq(1000)].groupby(['validation', 'kind', 'days']):
        r = dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1], Days=key[2],
                 **{'Actual bounds': f'{g.actual_count_lower.mean():.1f}–{g.actual_count_upper.mean():.1f}',
                    'Charge bounds': f'{g.charge_subtotal.mean():.1f}–{g.charge_count_upper.mean():.1f}',
                    'Historical': f'{g.historical_subtotal.mean():.1f}'})
        for method in METHODS:
            r[method.capitalize() + ' bias bounds'] = f'[{g[method+"_full_error_low"].mean():+.1f}, {g[method+"_full_error_high"].mean():+.1f}]'
        boundsrows.append(r)
    boundstable = pd.DataFrame(boundsrows)
    boundstable.to_csv(OUT / 'full_cohort_bounds_summary.csv', index=False)
    sourcetable = []
    for key, g in support.groupby(['kind', 'nominal']):
        r = dict(Fleet=key[0], N=key[1])
        for field, title in [('unique_sources', 'All unique'), ('unique_sources_calce', 'CALCE'),
                             ('unique_sources_nasa', 'NASA'), ('unique_sources_oxford', 'Oxford')]:
            r[title] = f'{g[field].min()}–{g[field].max()}'
        sourcetable.append(r)
    winrows = []
    for key, g in metrics.groupby(['validation', 'kind']):
        p = g.pivot(index=['nominal', 'days'], columns='method', values='mae_count')
        p = p.dropna()
        wins = p.charge.lt(p.historical) & p.charge.lt(p.cycle)
        winrows.append(dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1],
                            **{'Scored size/horizon groups (of 9)': len(p), 'Charge beats both': int(wins.sum()),
                               'Historical beats charge': int(p.historical.lt(p.charge).sum())}))
    mixed60 = s[s.validation.eq(PROTOCOLS[1]) & s.kind.eq('mixed') & s.nominal.eq(1000) & s.days.eq(60)]
    mixed_m = metrics[metrics.validation.eq(PROTOCOLS[1]) & metrics.kind.eq('mixed') & metrics.nominal.eq(1000) & metrics.days.eq(60)].set_index('method')
    calce60 = s[s.validation.eq(PROTOCOLS[1]) & s.kind.eq('calce') & s.nominal.eq(1000) & s.days.eq(60)]
    text = f'''{opening}

# Fleet forecast v2

## Fixed design and forecast method

Preregistration was committed as `{commit}` before the first simulation run.
All 20 preregistered seeds, 50/200/1,000 installed copies, four fleet kinds,
five today points (0/15/30/45/60), both holdout protocols and 30/60/90-day horizons
were used: **{len(s):,} scenario rows**. No parameters were tuned on these results.
Days are scenarios at 2 source lab cycles/day, not measured field calendar days.
Commissioning dates are independently uniform over each source dataset's age
span before day 0 (CALCE 1,008; NASA 98.5; Oxford 4,100 days). Smaller cohorts
are nested prefixes of the same 1,000-copy layouts, with no eligibility filtering.

**Deterministic v1 RUL crossing counts were used.** E6's probabilities/intervals
failed calibration, and its horizons omit the 60-cycle/30-day risk needed here;
they are not validated calibrated probabilities for this aggregation.
The charge model is the frozen training-selected charge-only baseline; the
cycle comparator is frozen E1 quadratic-cycle Ridge, each fed to v1's raw linear
RUL rule. Every resampled copy uses its source's own held-out predictions.
The fold audit verified test membership and exclusion from training/selection
for all {int(source_audit.source_held_out.sum())} available method/source/protocol
combinations; {int((~source_audit.source_held_out).sum())} combinations have no
predictions and remain explicitly missing, not evaluated with a fitted refit.

The historical comparator uses observed prior crossing events divided by past
observed at-risk battery-days, then `min(1, rate × horizon)` per at-risk copy.
This adapts v1's elapsed-fleet-time rate to staggered entry, as preregistered.
It uses only past checks/events from the same simulated fleet in both protocols.
LODO therefore tests a learned model's unseen-type transfer against an operator
with local historical records, not a cold start with no local history.
Mixed LODO aggregates three routed models, each excluding its copy's dataset;
it is not one model with all three datasets simultaneously unseen.

## Comparison: paired usable forecasts and known outcomes

These are **conditional counts**, requiring both model forecasts and known truth
on the same copies. Each method column is **mean predicted count / MAE / signed
bias**, in batteries per scenario; positive bias means too many replacements
predicted, negative means too few. Values average the five
dates and all 20 seeds, with empty cases unscored. “Scored / 100” gives the
nonempty cases; actual and matched copies/sources average those same scored
cases, so the mean actual count never exceeds that table's matched population.
Unique-source ranges describe the full installed layouts, not independent
replicates. Every count/error at every individual today is in [scenarios.csv](scenarios.csv).

### Within dataset (LOCO)

{comparison_table(PROTOCOLS[0])}

### Unseen dataset (LODO)

{comparison_table(PROTOCOLS[1])}

The v1 unseen-dataset historical-rate finding must be assessed per dataset as
well as overall; group counts below use strict lower paired MAE (ties do not win).

{pd.DataFrame(winrows).to_markdown(index=False)}

Historical rate still wins the overall scored LODO comparison and every NASA
size/horizon group. The mixed LODO exception is small: at N=1,000 and 60 days,
charge MAE is {mixed_m.loc['charge', 'mae_count']:.2f} versus historical
{mixed_m.loc['historical', 'mae_count']:.2f}, on only {mixed60.matched_n.mean():.1f}
matched copies out of {mixed60.at_risk.mean():.1f} at risk
({100 * mixed60.matched_n.mean() / mixed60.at_risk.mean():.1f}%).
It does not establish unseen-type fleet accuracy. CALCE LODO has **no paired
score**, because the cycle model abstains on every at-risk copy; the charge
model still covers {calce60.charge_usable.mean():.1f} of {calce60.at_risk.mean():.1f}
at-risk copies on average, with its own-population scores retained separately.
Oxford's LODO paired population contains **zero actual incidents**; its zero
charge/cycle MAE is not evidence of catching replacement events.

Equal weighting of the available scored fleet-kind/size/horizon groups gives the
following averages; CALCE's nine unscored LODO groups are omitted, never assigned
zero error. Within-dataset and unseen-dataset MAEs use different covered
populations and **cannot be compared as a model-improvement estimate**.
The absolute-count average combines different cohort sizes; per-100 diagnostics
expose the scale and coverage dependence. No statistical significance or
independent-sample uncertainty claim is made.

{overall.to_markdown(index=False, floatfmt='.3f')}

Per 100 **matched** copies, macro-averaged over sizes/horizons:

{norm_display.to_markdown(index=False)}

RMSE, signed bias, and both normalizations for each group are in
[metrics.csv](metrics.csv). [method_specific_metrics.csv](method_specific_metrics.csv)
also retains each method's own known-outcome population; those cohorts differ
and cannot be read as a paired competition.

## Full-cohort coverage and effects on counts

Nominal cohorts contain already-below and unsupported copies. Already-below
copies are excluded from **incident** risk, never replenished; even day 0 may
contain prior retirements. Thus 1,000 installed copies do not mean 1,000 active
batteries or 1,000 usable forecasts. Means below are over dates/seeds, without
triple counting horizons. “Missing” is among at-risk copies; “overdue” is a
usable forecast of a pre-today crossing and contributes zero future incidents.

{covtable.to_markdown(index=False, floatfmt='.1f')}

For each model, a missing forecast can add 0–1 future incidents. Its subtotal is
the sum over usable copies, and the upper bound adds every missing copy. Actual
counts similarly add unknown outcomes to known incidents. The following
1,000-copy examples show means of the per-scenario bounds; signed error bounds
combine both missing predictions and unknown truth. These are **not calibrated
probability intervals**, and there is no exact full-cohort MAE when truth or
forecasts are missing.

{boundstable.to_markdown(index=False)}

All sizes' subtotal error bounds, full prediction/truth signed error bounds and
absolute-error lower/upper bounds are in [scenarios.csv](scenarios.csv).
Positive and negative source-specific biases need not cancel in another age mix;
resampling a common biased degradation curve can scale its bias with fleet size.
The common-support scores above omit these uncertain members and cannot prove
whole-fleet replacement accuracy.

[coverage.csv](coverage.csv) lists every fleet/date/protocol, including source
support and already-below/overdue counts. [abstention_reasons.csv](abstention_reasons.csv)
counts no-prefix, ended, short, flat/increasing, and unsupported histories by
method; these are repeated copy/date records, not independent batteries.
There were {int(s.historical_zero_exposure.sum())} scenario rows with zero historical
exposure (fixed zero-rate fallback). Missing reference prefixes and retrospectively
pre-existing crossings are retained in the audit and scenario columns.

## Predicted vs actual over time

The fixed 60-day/1,000-copy plot retains **all** 20 seeds and averages the
nonempty scored cases at each today. Empty cases remain unscored rather than
zero errors. Each panel's actual and all predictions use that panel's same
paired known-outcome subset; protocol
panels can have different subsets. It is not a plot of full installed-cohort
counts. Cohort membership/coverage can change as today advances.

![Paired replacement counts over scenario time](replacement_counts.png)

## Real independent support and limitations

**Resampling does not create new independent batteries: a 1,000-copy CALCE fleet
still contains only 7 real degradation histories.** The source pools have 7
CALCE, 34 NASA and 8 Oxford inventory cells, but only 41 total cells have eligible
held-out charge predictions and only 19 have observed reference-threshold events.
NASA's inventory includes sources without usable charge histories. Oxford's
coarse 100-cycle capacity checks and few crossings limit short-horizon evidence.
The table gives unique sampled sources across the 20 layouts (min–max);
the comparison tables additionally show unique sources surviving matched scoring.

{pd.DataFrame(sourcetable).to_markdown(index=False)}

Repeated copies, the five dates, overlapping horizons, nested sizes and 20 seeds
do not increase this independent support. Larger simulated cohorts test arithmetic
and sensitivity to this age mixture, **not validated accuracy at 1,000-pack scale**.
The uniform age spans use retrospective record extents only to construct the
scenario, and do not represent measured Baaz demographics. Sampling uses the
whole inventory, with no rejection of failed forecasts or missing outcomes.

Known-below status and historical rates use only checks already observed at
today. Ground truth uses the frozen interpolated first 70% reference crossing;
an endpoint interpolated before today but first detected later is explicitly
flagged as retrospectively pre-existing. Right-censored follow-up is never an
invented zero. Ended charge records are unsupported even if a last fitted trend
could be extrapolated. Other cells' full trajectories can enter training under
cell holdout, so this is not a prospective calendar-cutoff trial. These are lab
cells and surrogate threshold incidents, not Baaz packs or actual replacements.

## Exact data needed from a Baaz pilot

- **Pack registry and commissioning dates:** stable pack ID, cell/pack model,
  chemistry, nominal/rated Ah or Wh, original commissioning timestamp, entry/exit
  dates, refurbishment/transfer history and station/vehicle ID linkage. Preserve
  one ID through every swap so age and prior replacements can be reconstructed.
- **Timestamped swap logs:** pack ID, station/vehicle, swap-in/out timestamps,
  start/end SoC, Ah/Wh throughput, distance/duty and temperature; charge telemetry
  (time, current, voltage, temperature and QC flags) tied to the same pack/cycle.
  Measure partial-swap throughput to replace the artificial 2-lab-cycles/day rule.
- **Replacement records:** pack ID, decision/removal/replacement dates, reason
  (capacity fade, failure, damage, policy, lost/retired), measured trigger and
  successor ID, including no-replacement follow-up and non-capacity removals as
  competing events. Distinguish routine swaps from permanent pack replacement.
- **Periodic independent capacity checks:** pack ID, date, measured discharge
  Ah/Wh, rated reference, SoH, test current, temperature, voltage cutoffs and
  instrument/protocol/QC. Include checks near end of life and the last check for
  packs still operating; do not derive the truth label from charge-model output.
- **Evaluation extract:** a real mixed-age active fleet with many unique packs
  and enough independently observed replacement events, complete records through
  each 30/60/90-day horizon (at least 90 days after the final today), and dated
  censoring/exit reasons. Freeze pack/type/calendar training holdouts, forecasts,
  thresholds, baselines and coverage rules before outcomes; retain abstentions.
  Keep any target-type calibration packs separate from held-out evaluation packs.

## Reproduction and integrity

From the repository root: `.venv/bin/python scripts/fleet_forecast_v2.py`.
To refresh only presentation from saved scores, use the same command with
`--report-only`; the manifest records unchanged raw-result hashes for that step.
The runner checks the committed preregistration and frozen baseline, audits all
available prediction folds, and hashes pre-existing tracked files before/after.
It writes only v2 artifacts; it does not regenerate v1, fit models, update the
dashboard or change main reports. See [run_manifest.json](run_manifest.json) for
input/code hashes, software versions and exact seeds/settings;
[layouts.csv](layouts.csv) for commissioning/source copies;
[source_fold_audit.csv](source_fold_audit.csv) for holdout provenance;
[source_support.csv](source_support.csv) for each sampled fleet's real support;
and [copy_date_audit.parquet](copy_date_audit.parquet) for every master-layout
copy/date/protocol, with smaller sizes selected by `copy_index < N`.
'''
    (OUT / 'results.md').write_text(text)


def run():
    started = datetime.now(timezone.utc).isoformat()
    commit = preregistration_commit()
    verify_baseline()
    protected = [p for p in git('ls-files', '-z').decode().split('\0') if p and
                 not p.startswith(('reports/fleet_forecast_v2/', 'src/battery_analytics/fleet_v2.py',
                                   'scripts/fleet_forecast_v2.py', 'tests/test_fleet_forecast_v2.py'))]
    before = {p: digest(p) for p in protected}
    inputs = {p: digest(p) for p in INPUTS}
    inventory = pd.read_csv(ROOT / 'reports/inventory/cells.csv').sort_values(['dataset', 'cell_id'])
    refs = reference_histories(pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet'))
    assert (refs.groupby('dataset').cycle_index.max() / 2).to_dict() == AGE_SPANS
    references = {f'{d}:{c}': g.sort_values('cycle_index') for (d, c), g in refs.groupby(['dataset', 'cell_id'])}
    endpoints = pd.read_csv(BASE / 'end_of_life.csv')
    endpoints.index = endpoints.dataset + ':' + endpoints.cell_id
    assert set(inventory.dataset + ':' + inventory.cell_id) == set(endpoints.index)
    empty_refs = refs.iloc[:0]
    for gid in endpoints.index:
        references.setdefault(gid, empty_refs)
    predictions = {'charge': baseline(), 'cycle': pd.read_parquet(ROOT / 'reports/improvements/E1_cycle/predictions.parquet')}
    source_audit = audit_predictions(predictions,
                                    json.loads((BASE / 'fold_audit.json').read_text()),
                                    json.loads((ROOT / 'reports/improvements/E1_cycle/audit.json').read_text()), inventory)
    caches = {}
    for method, p in predictions.items():
        for (protocol, gid), h in p.groupby(['validation', 'group_id']):
            caches[method, protocol, gid] = prefix_crossings(h)
        print(f'Cached causal {method} RUL prefixes', flush=True)
    missing_cache = (np.array([]), np.array([]), np.array([], dtype=object))
    layouts, audits, scenarios, coverage, reason_rows, support = [], [], [], [], [], []
    for kind in KINDS:
        for seed in SEEDS:
            layout = make_layout(inventory, kind, seed)
            layouts.append(layout)
            base = build_copies(layout, references, endpoints)
            for size in SIZES:
                l = layout.iloc[:size]
                support.append(dict(kind=kind, seed=seed, nominal=size, unique_sources=l.group_id.nunique(),
                                    **{f'unique_sources_{d}': l[l.dataset.eq(d)].group_id.nunique() for d in DATASETS}))
            for protocol in PROTOCOLS:
                g = base.copy()
                g['validation'] = protocol
                for method in ('charge', 'cycle'):
                    g[f'{method}_rul'] = np.nan
                    g[f'{method}_reason'] = 'no_prefix'
                    for gid, h in g.groupby('group_id', sort=False):
                        remaining, reasons = lookup_forecasts(caches.get((method, protocol, gid), missing_cache), h.source_age_cycles)
                        g.loc[h.index, f'{method}_rul'] = remaining
                        g.loc[h.index, f'{method}_reason'] = reasons
                audits.append(g)
                for today, day in g.groupby('today', sort=True):
                    for size in SIZES:
                        cohort = day[day.copy_index < size]
                        shared = dict(kind=kind, seed=seed, validation=protocol, today=today)
                        for days in HORIZONS:
                            result = scenario(cohort, days, size)
                            scenarios.append(dict(**shared, days=days, horizon_cycles=days * 2, **result))
                            if days == HORIZONS[0]:
                                fields = [k for k in result if k in ('nominal', 'unique_sources', 'at_risk', 'at_risk_unique',
                                    'already_below', 'initially_left_censored', 'retrospective_preexisting', 'no_reference_prefix') or
                                    k.endswith(('_usable', '_usable_unique', '_overdue', '_unknown')) and k != 'historical_unknown']
                                coverage.append(dict(**shared, **{k: result[k] for k in fields}))
                        risk = cohort[~cohort.already_below]
                        for method in ('charge', 'cycle'):
                            for reason, n in risk[f'{method}_reason'].value_counts().items():
                                reason_rows.append(dict(**shared, nominal=size, method=method, reason=reason, copy_date_records=n))
        print(f'Completed all {len(SEEDS)} seeds for {kind}', flush=True)
    s, cov = pd.DataFrame(scenarios), pd.DataFrame(coverage)
    assert len(s) == 7200
    assert not s.duplicated(['kind', 'seed', 'validation', 'today', 'nominal', 'days']).any()
    metrics = summarize(s)
    reasons = pd.DataFrame(reason_rows).groupby(['validation', 'kind', 'nominal', 'method', 'reason'], as_index=False).copy_date_records.sum()
    support = pd.DataFrame(support)
    pd.concat(layouts, ignore_index=True).to_csv(OUT / 'layouts.csv', index=False)
    pd.concat(audits, ignore_index=True).to_parquet(OUT / 'copy_date_audit.parquet', index=False)
    s.to_csv(OUT / 'scenarios.csv', index=False)
    cov.to_csv(OUT / 'coverage.csv', index=False)
    support.to_csv(OUT / 'source_support.csv', index=False)
    source_audit.to_csv(OUT / 'source_fold_audit.csv', index=False)
    reasons.to_csv(OUT / 'abstention_reasons.csv', index=False)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    summarize(s, specific=True).to_csv(OUT / 'method_specific_metrics.csv', index=False)
    make_plot(s)
    write_report(s, metrics, cov, reasons, support, source_audit, commit)
    after = {p: digest(p) for p in protected}
    assert before == after, 'Pre-existing tracked files changed'
    verify_baseline()
    manifest = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration_commit=commit, preregistration_sha256=digest(PREREG),
                    repository_head=git('rev-parse', 'HEAD').decode().strip(), input_sha256=inputs,
                    code_sha256={p: digest(p) for p in ['scripts/fleet_forecast_v2.py', 'src/battery_analytics/fleet_v2.py']},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    settings=dict(kinds=KINDS, sizes=SIZES, seeds=SEEDS, today_days=TODAYS, horizons_days=HORIZONS,
                                  cycles_per_day=2, commissioning_age_spans_days=AGE_SPANS, methods=METHODS,
                                  probabilities_used=False, validation=PROTOCOLS),
                    scenario_rows=len(s), layout_rows=sum(len(l) for l in layouts),
                    copy_date_protocol_rows=sum(len(a) for a in audits), protected_files_unchanged=True,
                    protected_sha256_before=before, protected_sha256_after=after)
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(metrics.groupby(['validation', 'method'])[['mae_count', 'signed_bias_count']].mean().to_string(), flush=True)
    print('Saved', OUT / 'results.md', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Refresh presentation from existing saved scores without rerunning the experiment')
    args = parser.parse_args()
    if not args.report_only:
        run()
    else:
        commit = preregistration_commit()
        verify_baseline()
        manifest = json.loads((OUT / 'run_manifest.json').read_text())
        assert manifest['preregistration_commit'] == commit
        assert manifest['preregistration_sha256'] == digest(PREREG)
        for path, sha in manifest['input_sha256'].items():
            assert digest(path) == sha
        raw = ['scenarios.csv', 'layouts.csv', 'copy_date_audit.parquet', 'metrics.csv',
               'method_specific_metrics.csv', 'coverage.csv', 'source_support.csv',
               'source_fold_audit.csv', 'abstention_reasons.csv']
        raw_hashes = {p: digest(f'reports/fleet_forecast_v2/{p}') for p in raw}
        frames = [pd.read_csv(OUT / p) for p in ['scenarios.csv', 'metrics.csv', 'coverage.csv',
                                                'abstention_reasons.csv', 'source_support.csv', 'source_fold_audit.csv']]
        make_plot(frames[0])
        write_report(*frames, commit)
        assert raw_hashes == {p: digest(f'reports/fleet_forecast_v2/{p}') for p in raw}
        assert manifest['protected_sha256_before'] == {p: digest(p) for p in manifest['protected_sha256_before']}
        manifest.setdefault('execution_code_sha256', manifest['code_sha256'])
        manifest['code_sha256'] = {p: digest(p) for p in manifest['code_sha256']}
        manifest['report_refresh'] = dict(utc=datetime.now(timezone.utc).isoformat(),
                                        reason='Clarify unscored groups, matched denominators and thin-coverage LODO exceptions; no design or score changes',
                                        raw_artifacts_sha256_unchanged=raw_hashes)
        (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Refreshed report/plot; raw experiment artifacts unchanged')
