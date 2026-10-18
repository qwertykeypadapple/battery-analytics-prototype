"""Run the preregistered hybrid study without modifying any earlier experiment."""
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
import platform
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from battery_analytics.improvements.core import baseline, verify_baseline, BASE
from battery_analytics.fleet_v2 import KINDS, SIZES, SEEDS, HORIZONS, PROTOCOLS, audit_predictions
from battery_analytics.hybrid_fleet import (
    METHODS, K_MODES, RANDOM_DRAWS, SCORE_FIELDS, current_soh, tie_keys,
    ranking_order, historical_budget, evaluate_selection, evaluate_random,
    source_mask, summarize, paired_comparisons,
)

OUT = ROOT / 'reports/hybrid_fleet'
V2 = ROOT / 'reports/fleet_forecast_v2'
PREREG = 'reports/hybrid_fleet/preregistration.md'
NEW_PATHS = ('reports/hybrid_fleet/', 'src/battery_analytics/hybrid_fleet.py',
             'scripts/hybrid_fleet.py', 'tests/test_hybrid_fleet.py')
INPUTS = [
    'reports/fleet_forecast_v2/layouts.csv', 'reports/fleet_forecast_v2/copy_date_audit.parquet',
    'reports/fleet_forecast_v2/scenarios.csv', 'reports/fleet_forecast_v2/run_manifest.json',
    'reports/improvements/baseline/predictions.parquet', 'reports/improvements/baseline/fold_audit.json',
    'reports/improvements/baseline/end_of_life.csv', 'reports/improvements/baseline/manifest.json',
    'reports/improvements/E1_cycle/predictions.parquet', 'reports/improvements/E1_cycle/audit.json',
    'reports/inventory/cells.csv',
]
LABELS = dict(charge_soh='Charge SoH', charge_rul='Charge RUL', cycle_soh='Cycle SoH', age='Age', random='Random')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def preregistration_commit():
    commit = git('log', '--diff-filter=A', '--format=%H', '--', PREREG).decode().strip()
    assert len(commit.splitlines()) == 1 and len(commit) == 40, 'Commit preregistration first'
    assert git('show', f'{commit}:{PREREG}') == (ROOT / PREREG).read_bytes(), 'Preregistration changed'
    return commit


def add_point_scores(audit, predictions):
    g = audit.copy()
    for label, p in predictions.items():
        field = label + '_soh'
        g[field] = np.nan
        g[field + '_reason'] = 'no_prefix'
        lookup = {(protocol, gid): h for (protocol, gid), h in p.groupby(['validation', 'group_id'])}
        empty = p.iloc[:0]
        for key, h in g.groupby(['validation', 'group_id'], sort=False):
            values, reasons = current_soh(lookup.get(key, empty), h.source_age_cycles)
            g.loc[h.index, field] = values
            g.loc[h.index, field + '_reason'] = reasons
    return g


def unique_cohorts(layouts, audit, inventory):
    keys = layouts[layouts.seed.eq(SEEDS[0]) & ~layouts.kind.eq('mixed')]
    keys = keys.sort_values(['kind', 'copy_index']).drop_duplicates('group_id', keep='first')
    expected = set(inventory.dataset + ':' + inventory.cell_id)
    assert set(keys.group_id) == expected and len(keys) == 49
    single = audit.merge(keys[['kind', 'seed', 'copy_index', 'group_id']],
                         on=['kind', 'seed', 'copy_index', 'group_id'], how='inner', validate='many_to_one')
    assert len(single) == 49 * 5 * 2
    unique = pd.concat([single, single.assign(kind='mixed')], ignore_index=True)
    unique['representation'] = 'unique'
    unique['unit_id'] = unique.group_id
    unique_layouts = pd.concat([keys, keys.assign(kind='mixed')], ignore_index=True)
    unique_layouts['unit_id'] = unique_layouts.group_id
    assert not unique_layouts.duplicated(['kind', 'group_id']).any()
    return unique, unique_layouts


def support_summary(cases):
    rows = []
    keys = ['representation', 'kind', 'nominal', 'days']
    # Truth, commissioning and historical K are identical across protocols.
    for key, g in cases[cases.validation.eq(PROTOCOLS[0])].groupby(keys):
        mask = 0
        for value in g.actual_positive_source_mask.unique():
            mask |= int(value, 16)
        rows.append(dict(zip(keys, key), scenario_windows=len(g), min_real_sources=g.unique_sources.min(),
                         max_real_sources=g.unique_sources.max(), distinct_positive_sources=mask.bit_count(),
                         positive_source_mask=hex(mask), mean_at_risk_unique=g.at_risk_unique.mean(),
                         mean_unknown=g.unknown_actual.mean(), mean_at_risk=g.at_risk.mean(),
                         zero_historical_k=int(g.historical_k.eq(0).sum()),
                         zero_known_positives=int(g.known_positives.eq(0).sum()),
                         fully_observed_windows=int(g.unknown_actual.eq(0).sum()),
                         oracle_unavailable=int(g.unknown_actual.gt(0).sum())))
    return pd.DataFrame(rows)


def plot_rankings(metrics, cases):
    g = metrics[metrics.representation.eq('unique') & metrics.k_mode.eq('historical')]
    means = g.groupby(['validation', 'method'])[['precision', 'recall']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    positions = np.arange(len(METHODS))
    for ax, field, title in zip(axes, ['precision', 'recall'],
                               ['Precision among known-outcome flags', 'Recall of known incidents']):
        for offset, protocol, label, color in [(-.18, PROTOCOLS[0], 'LOCO: within dataset', '#00877c'),
                                                (.18, PROTOCOLS[1], 'LODO: unseen dataset', '#6279bc')]:
            values = [means.loc[(protocol, method), field] for method in METHODS]
            ax.barh(positions + offset, values, height=.32, color=color, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_yticks(positions, [LABELS[m] for m in METHODS])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Conditional fraction; undefined cases omitted')
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    mask = 0
    for value in cases[cases.representation.eq('unique')].actual_positive_source_mask.unique():
        mask |= int(value, 16)
    fig.suptitle('Unique-cell cohorts, historical K; equal kind/horizon macro averages\n'
                 f'49 real cells; only {mask.bit_count()} incident histories in these windows (19 total source events)', fontsize=12)
    fig.tight_layout(rect=[0, .09, 1, .89])
    fig.savefig(OUT / 'ranking_comparison.png', dpi=170)
    plt.close(fig)


def write_report(results, metrics, cases, pairs, support, commit):
    def pct(x):
        return f'{100 * x:.1f}' if np.isfinite(x) else '—'

    def table(representation, mode):
        g = metrics[metrics.representation.eq(representation) & metrics.k_mode.eq(mode)]
        rows = []
        for key, h in g.groupby(['validation', 'kind', 'days']):
            first = h[h.method.eq('age')]
            mask = 0
            for value in h.actual_positive_source_mask.unique():
                mask |= int(value, 16)
            r = dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1], Days=key[2],
                     **{'Real sources': f'{h.min_real_sources.min()}–{h.max_real_sources.max()}',
                        'Incident sources': mask.bit_count(),
                        'K available / cases': f'{first.k_available_cases.sum()} / {first.cases.sum()}'})
            for method in METHODS:
                m = h[h.method.eq(method)]
                r[LABELS[method] + ' P / R %'] = f'{pct(m.precision.mean())} / {pct(m.recall.mean())}'
            rows.append(r)
        return pd.DataFrame(rows).to_markdown(index=False)

    primary_pairs = pairs[pairs.representation.eq('unique') & pairs.k_mode.eq('historical')]
    paired_macro = primary_pairs.groupby(['validation', 'model', 'baseline', 'metric'], as_index=False).agg(
        difference=('mean_difference', 'mean'), defined_groups=('mean_difference', 'count'),
        paired_window_records=('paired_cases', 'sum'))
    advantages = {}
    for protocol in PROTOCOLS:
        for model in ('charge_soh', 'charge_rul'):
            q = paired_macro[paired_macro.validation.eq(protocol) & paired_macro.model.eq(model)]
            advantages[protocol, model] = len(q) == 6 and q.difference.notna().all() and q.difference.gt(0).all()
    if advantages[PROTOCOLS[0], 'charge_soh'] and not advantages[PROTOCOLS[1], 'charge_soh']:
        soh_message = ('Charge SoH picks the right batteries better than cycle SoH, age and random on the paired known-outcome comparisons within datasets, but that advantage does not survive unseen battery types')
    elif all(advantages[p, 'charge_soh'] for p in PROTOCOLS):
        soh_message = 'Charge SoH ranks better than cycle SoH, age and random on both paired conditional metrics under both holdout protocols'
    else:
        soh_message = 'Charge SoH does not consistently pick the right batteries better than cycle SoH, age and random in the unique-cell test'
    rul_message = ('The RUL alternative also has a descriptive paired advantage under both protocols'
                   if all(advantages[p, 'charge_rul'] for p in PROTOCOLS) else
                   'The RUL alternative does not establish a consistent advantage either')
    bounds_rows = []
    for key, g in metrics[metrics.representation.eq('unique') & metrics.k_mode.eq('historical')].groupby(['validation', 'kind', 'method']):
        bounds_rows.append(dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1], Method=LABELS[key[2]],
                                **{'Precision bounds %': f'[{pct(g.precision_lower.mean())}, {pct(g.precision_upper.mean())}]',
                                   'Recall bounds %': f'[{pct(g.recall_lower.mean())}, {pct(g.recall_upper.mean())}]',
                                   'Defined conditional P / R cases': f'{g.precision_cases.sum()} / {g.recall_cases.sum()} of {g.cases.sum()}'}))
    coverage_rows = []
    historical = metrics[metrics.k_mode.eq('historical') & metrics.method.isin(SCORE_FIELDS)]
    for key, g in historical.groupby(['representation', 'validation', 'kind', 'method']):
        coverage_rows.append(dict(Population=key[0], Protocol='LOCO' if key[1] == PROTOCOLS[0] else 'LODO', Fleet=key[2], Method=LABELS[key[3]],
                                 **{'Mean at-risk real sources': g.at_risk_unique.mean(),
                                    'Mean missing real sources': g.missing_unique.mean(),
                                    'Mean fallback-selected real sources': g.fallback_selected_unique.mean()}))
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / 'coverage_summary.csv', index=False)
    paired_display = paired_macro.copy()
    paired_display['difference_pp'] = paired_display.difference * 100
    paired_display = paired_display.drop(columns='difference')
    unique_support = support[support.representation.eq('unique')].drop(columns=['representation', 'nominal', 'positive_source_mask'])
    masks = {}
    for representation in ['unique', 'v2']:
        mask = 0
        for value in cases[cases.representation.eq(representation)].actual_positive_source_mask.unique():
            mask |= int(value, 16)
        masks[representation] = mask.bit_count()
    opening = (soh_message + '. ' + rul_message + '. '
               f'This supports only a clearly labelled workflow demo for Baaz, not an accuracy claim: just {masks["unique"]} of the 19 source incident histories appear in the unique-cell windows, with synthetic commissioning dates and incomplete outcomes.')
    counts = cases.groupby(['representation', 'kind']).agg(
        windows=('case_id', 'count'), fully_observed=('unknown_actual', lambda v: int(v.eq(0).sum())),
        zero_K=('historical_k', lambda v: int(v.eq(0).sum())),
        zero_known_incidents=('known_positives', lambda v: int(v.eq(0).sum())),
        mean_at_risk=('at_risk', 'mean'), mean_real_at_risk=('at_risk_unique', 'mean'),
        mean_unknown=('unknown_actual', 'mean')).reset_index()
    counts['flag_all_nonempty_windows'] = [int(((cases.representation == row.representation) &
        (cases.kind == row.kind) & cases.at_risk.gt(0) & cases.historical_k.eq(cases.at_risk)).sum())
        for row in counts.itertuples()]
    exact = metrics[metrics.representation.eq('unique') & metrics.k_mode.eq('historical')]
    exact_rows = []
    for key, g in exact.groupby(['validation', 'kind', 'method']):
        exact_rows.append(dict(Protocol='LOCO' if key[0] == PROTOCOLS[0] else 'LODO', Fleet=key[1], Method=LABELS[key[2]],
                               **{'Exact P / R %': f'{pct(g.exact_precision.mean())} / {pct(g.exact_recall.mean())}',
                                  'Exact defined P / R cases': f'{g.exact_precision_cases.sum()} / {g.exact_recall_cases.sum()}'}))
    pd.DataFrame(exact_rows).to_csv(OUT / 'exact_only_summary.csv', index=False)
    text = f'''{opening}

# Hybrid fleet forecast

## Design and leakage controls

Preregistration commit: `{commit}`, made before ranking evaluation. No model
fitting, tuning, score transformation or post-result seed selection occurred.
All earlier tracked artifacts were hashed before/after and remain unchanged.
There are 7,200 v2 cohort/date/horizon/protocol cases and 120 unique-cell cases,
each retaining five methods and two planned K modes, including unavailable
oracle records: **{len(results):,} method/K rows**.

The v2 evaluation reads the saved 50/200/1,000-copy staggered layouts, all 20
seeds, five today points (0/15/30/45/60), three horizons (30/60/90 days), and both
holdout protocols. Unique cohorts contain each original inventory cell once:
CALCE 7, NASA 34, Oxford 8; mixed is their 49-cell union. Their offsets are the
first occurrence of each cell in the first registered v2 single-dataset layout,
reused in mixed. These are **synthetic commissioning dates**, not real pack dates;
the unique version has no duplicate source cell within a cohort. All histories
advance at the same fixed two-source-cycles/day scenario conversion.

Only saved held-out training-selected charge predictions and E1 cycle-only
predictions are used. Every available source was re-audited against its training/
selection and outer-test membership; LODO excludes its entire dataset. Mixed
LODO routes each source to the model excluding that source dataset. Other cells'
full trajectories may be in training; this is not prospective calendar holdout.
Ranking functions receive only model scores, age and fixed outcome-blind tie
keys. Reference capacity labels and future observations never enter a score.

## Fixed rankings, budgets and missing scores

- **Charge SoH (primary):** lowest latest raw held-out predicted SoH first.
- **Charge RUL (secondary):** shortest frozen v2 raw linear remaining life first,
  including finite overdue/negative RUL. It is reported independently of SoH.
- **Cycle SoH:** lowest latest raw E1 cycle-model predicted SoH first.
- **Age:** oldest since commissioning first.
- **Random:** 2,000 equivalent uniform top-K draws per available case, with
  fixed seeds shared between protocols. Hypergeometric category draws evaluate
  uniformly selected sets without sorting 2,000 full permutations. Exact random
  expected positives are saved as a check; random draws add no real evidence.

Current SoH needs a finite last raw past point; absent or ended eligible histories
are missing. RUL retains v2's insufficient/flat/unsupported/ended/no-prefix rules.
For each model, finite scores rank first; **all missing-score members follow,
oldest first**, with SHA-256 tie-breaking fixed before the study. Missing members
are never dropped. This tests each score plus its registered availability/fallback
policy; fallback counts must accompany ranking metrics.

**Historical K is the primary usable budget:** v2's local observed first-crossing
events / observed at-risk battery-days, multiplied by horizon and at-risk size,
capped at that size, then rounded half-up. Unique cohorts recompute this rate
from their own past. All methods use the same K and at-risk pool; no future-label
or paired-availability filtering occurs before selection. Routine swaps are not
permanent replacements. Past events here are reference-threshold detections.

**Oracle K is diagnostic only, unusable in practice:** actual future incidents
in that same pool, available only when every at-risk outcome is known. Unknown
outcomes do not become non-events and are not filtered out to create an oracle.
The ranking order stays fixed; only the K cutoff changes.

## Precision and recall: unique real cells first

Each method entry is **precision / recall (%)**, macro-averaged over defined
today cases. Precision is the fraction of flagged **known-outcome** members
that crossed 70%; recall is the fraction of **known incidents** that were flagged.
These are exact only in fully observed cases; otherwise they are conditional
diagnostics, and methods can have different known-flag denominators. The paired
differences below compare only mutually defined cases. Full-population bounds
are reported later. “—” means undefined, not zero or perfect accuracy.
K=0 leaves precision undefined and gives recall=0 if incidents exist; no incidents
leaves recall undefined. Incident-source counts are distinct real histories with
at least one known incident in that group's windows, not independent events per row.

### Primary: historical K, unique-cell cohorts

{table('unique', 'historical')}

### Diagnostic only: oracle K, unique-cell cohorts

An available oracle with zero incidents has K=0 and undefined P/R. K availability
does not imply an informative positive-event ranking test.

{table('unique', 'oracle')}

### Paired comparisons on unique cells, historical K

Differences are **charge minus baseline, percentage points**, equally averaged
over available fleet-kind/horizon groups. Precision and recall use their own
mutually defined cases; window-record totals are correlated repeated evaluations,
not independent batteries. A consistent advantage requires positive differences
for both metrics against all three alternatives; do not choose SoH or RUL after
seeing which looks better. Detailed per-dataset/horizon paired counts and real
positive-source support are in [paired_comparisons.csv](paired_comparisons.csv).

{paired_display.to_markdown(index=False, floatfmt='.2f')}

The positive within-dataset charge-SoH comparison is descriptive and is driven
by a few mixed-fleet windows. It does not persist in LODO: SoH ties cycle SoH on
the paired conditional metrics, loses precision to age, and loses both metrics
to random. RUL does not dominate all alternatives even within datasets.

The unique-cell **CALCE historical budget rounds to K=0 at every today/horizon**,
so its primary recall is zero when that one source incident is in the horizon;
ranking cannot repair a zero replacement budget. Its 100% oracle results are
one real incident history repeated across windows, also caught by age, not
evidence of model superiority. Unique NASA flags **every at-risk member** at
60 and 90 days, making all ranking methods identical at those cutoffs; its
100% known-incident recall therefore does not demonstrate useful prioritization.
Oxford has no unique-cell incident windows. These cases sharply limit the test.

## One ranking comparison chart

This uses unique-cell historical K only. Bars average available fleet-kind/
horizon groups equally; conditional denominators and omitted undefined cases
are exposed in [metrics.csv](metrics.csv) and the bounds table, not evidence of
complete-fleet accuracy. Mixed and single-dataset cohorts share real histories.

![Unique-cell historical-K ranking comparison](ranking_comparison.png)

## V2 layout sensitivity: copies are not new batteries

The following tables average sizes equally, then all fixed seeds/dates within
each size. They describe the operating scenario of flagging copies; they do not
multiply independent evidence. Real-source ranges are shown instead of labelling
1,000 copies as a sample of 1,000 real batteries. For size-specific precision/
recall, defined cases, K, fallback, and unique/event support, see [metrics.csv](metrics.csv).

### Primary: historical K on saved v2 layouts

{table('v2', 'historical')}

### Diagnostic only: oracle K on saved v2 layouts

{table('v2', 'oracle')}

## Censoring bounds and forecast coverage

If T known incidents and Uf unknown outcomes are flagged, full precision lies
between T/K and (T+Uf)/K. Recall bounds also allow unknown outcomes outside the
flagged set to be incidents; see the preregistration for exact formulas and
zero-denominator rules. Means below are means of per-scenario bounds, not
calibrated uncertainty intervals or precision/recall estimates. Wide bounds
prevent a reliable ranking claim even when conditional metrics look good.

{pd.DataFrame(bounds_rows).to_markdown(index=False)}

Every case's bounds, true positives, known false positives, flagged unknowns,
unique flagged/caught sources, zero-K status and draw availability are saved in
[scenario_metrics.csv](scenario_metrics.csv). Fully observed historical-K unique
metrics are separately saved in [exact_only_summary.csv](exact_only_summary.csv);
undefined exact metrics are not replaced with conditional ones.

Model availability and fallback, averaged over kinds' registered windows/sizes:
the unit below is **unique real sources per scenario**, not independent copies.
RUL and point-SoH have deliberately different eligibility requirements.

{coverage.to_markdown(index=False, floatfmt='.2f')}

Population and zero-denominator audit (windows include both protocols, which
share truth and do not create new endpoint events):

{counts.to_markdown(index=False, floatfmt='.2f')}

## How much independent evidence is present?

Across all source data there are **only 19 observed first-crossing events**:
7 CALCE, 10 NASA, 2 Oxford; the complete inventory is 49 real cells, and only
41 have eligible held-out model predictions. In this fixed unique-cell age mix,
**{masks['unique']} distinct real incident-source histories** appear in any evaluated
incident window; across the resampled v2 layouts the corresponding count is
{masks['v2']}. Sources already known below threshold are outside the future-incident
population. Repeated windows/horizons, both protocols, mixed plus single-dataset
cohorts, random draws and copies never create additional independent events.

Unique-cell source support below counts each real source once across windows.
When a dataset has zero incident sources it cannot test event capture; when it
has only a handful it is too small for a reliable superiority conclusion.
Oracle results may be absent because complete follow-up is unavailable, not
because a method has zero error.

{unique_support.to_markdown(index=False, floatfmt='.2f')}

The uniform age mixture and missing-first/ended histories can change rankings and
fallback exposure. Lower SoH MAE does not guarantee better top-K event selection,
especially across chemistries. The 70% endpoint is a lab capacity surrogate;
Oxford's coarse checks and initially left-censored NASA cells restrict timing
and support. No significance or independent-battery confidence interval is
computed from these correlated evaluations. A qualified demonstration can show
budgeting, prioritization and missing-data handling; these results cannot prove
Baaz replacement targeting accuracy or operating savings.

## Baaz data needed to test this properly

Stable **pack IDs/types** linked across stations/vehicles; original
**commissioning dates** and refurbishment/exit dates; timestamped **swap logs**
with SoC, Ah/Wh throughput and linked charge voltage/current/temperature traces;
dated permanent **replacement records** with pack ID, reason, trigger and successor
ID (distinct from routine swaps); and periodic independent **capacity checks**
with measured Ah/Wh, rated reference, temperature, current, voltage cutoffs and QC.
Follow every evaluation pack through 90 days after the final today, record exits
and competing removals, and freeze pack/type/calendar holdouts, K, rankings and
fallbacks before outcomes. Many more unique packs and independently observed
replacement events are needed; target-calibration packs must be separate.

## Reproduction and audit artifacts

From the repository root: `.venv/bin/python scripts/hybrid_fleet.py`.
Add `--report-only` to refresh prose/chart from saved scores without rerunning
rank selection or Monte Carlo draws; raw-output integrity is checked and logged.
Read [preregistration.md](preregistration.md) and [run_manifest.json](run_manifest.json)
for the pre-run commit, exact inputs, settings, software and integrity hashes.
[unique_layouts.csv](unique_layouts.csv) records each cell's one retained offset;
[source_fold_audit.csv](source_fold_audit.csv) records held-out provenance;
[rank_audit.parquet](rank_audit.parquet) records all at-risk units' raw scores,
per-method order, missing flags, ages and truth for independent replay;
[case_audit.csv](case_audit.csv) records each fleet/date/horizon population and
historical/oracle budget; [source_support.csv](source_support.csv) and
[source_id_map.json](source_id_map.json) decode distinct-source support/masks.
The runner writes only hybrid files and does not rerun earlier experiments.
'''
    (OUT / 'results.md').write_text(text)


def run():
    started = datetime.now(timezone.utc).isoformat()
    commit = preregistration_commit()
    verify_baseline()
    protected = [p for p in git('ls-files', '-z').decode().split('\0') if p and not p.startswith(NEW_PATHS)]
    before = {p: digest(p) for p in protected}
    input_hashes = {p: digest(p) for p in INPUTS}
    # Verify that v2's prediction inputs still match its own execution manifest.
    v2_manifest = json.loads((V2 / 'run_manifest.json').read_text())
    for path, sha in v2_manifest['input_sha256'].items():
        assert digest(path) == sha, path
    inventory = pd.read_csv(ROOT / 'reports/inventory/cells.csv')
    predictions = {'charge': baseline(), 'cycle': pd.read_parquet(ROOT / 'reports/improvements/E1_cycle/predictions.parquet')}
    fold_audit = audit_predictions(predictions, json.loads((BASE / 'fold_audit.json').read_text()),
                                   json.loads((ROOT / 'reports/improvements/E1_cycle/audit.json').read_text()), inventory)
    layouts = pd.read_csv(V2 / 'layouts.csv')
    audit = add_point_scores(pd.read_parquet(V2 / 'copy_date_audit.parquet'), predictions)
    assert len(audit) == 800000
    assert np.allclose(audit.source_age_cycles, 2 * (audit.today - audit.commissioning_day))
    unique, unique_layouts = unique_cohorts(layouts, audit, inventory)
    audit['representation'], audit['unit_id'] = 'v2', audit.copy_index.astype(str)
    source_bits = {gid: i for i, gid in enumerate(sorted(inventory.dataset + ':' + inventory.cell_id))}
    v2_expected = pd.read_csv(V2 / 'scenarios.csv').set_index(['kind', 'seed', 'validation', 'nominal', 'today', 'days']).historical_subtotal.to_dict()
    all_rows, case_rows, rank_frames = [], [], []
    for representation, all_data in [('unique', unique), ('v2', audit)]:
        for (kind, seed, protocol, today), master in all_data.groupby(['kind', 'seed', 'validation', 'today'], sort=True):
            sizes = (master.group_id.nunique(),) if representation == 'unique' else SIZES
            for size in sizes:
                installed = master if representation == 'unique' else master[master.copy_index < size]
                risk = installed[~installed.already_below].reset_index(drop=True).copy()
                m = len(risk)
                predictor_columns = ['source_age_cycles', 'charge_soh', 'charge_rul', 'cycle_soh']
                causal_predictors = risk[predictor_columns].copy()
                ties = tie_keys(representation, kind, seed, risk.unit_id)
                orders, missing = {}, {}
                for method in METHODS[:-1]:
                    orders[method], missing[method] = ranking_order(causal_predictors, method, ties)
                    ranks = np.empty(m, dtype=int)
                    ranks[orders[method]] = np.arange(1, m + 1)
                    risk[method + '_rank'] = ranks
                    risk[method + '_missing'] = missing[method]
                risk['nominal'] = size
                rank_frames.append(risk)
                shared = dict(representation=representation, kind=kind, seed=int(seed), validation=protocol,
                              today=int(today), nominal=int(size), unique_sources=installed.group_id.nunique(),
                              already_below=int(installed.already_below.sum()), at_risk=m,
                              at_risk_unique=risk.group_id.nunique())
                for days in HORIZONS:
                    y = risk[f'actual_{days}'].to_numpy(float)
                    a, u = int(np.sum(y == 1)), int(np.isnan(y).sum())
                    budget = historical_budget(installed, days)
                    if representation == 'v2':
                        assert np.isclose(budget['historical_count'], v2_expected[kind, seed, protocol, size, today, days])
                    case_id = f'{representation}:{kind}:{seed}:{protocol}:{size}:{today}:{days}'
                    known_mask = source_mask(risk.loc[y == 1, 'group_id'], source_bits)
                    case = dict(**shared, case_id=case_id, days=days, known_positives=a, unknown_actual=u,
                                actual_positive_unique=int(known_mask, 16).bit_count(),
                                actual_positive_source_mask=known_mask, oracle_k=a if u == 0 else np.nan,
                                oracle_available=u == 0, **budget)
                    case_rows.append(case)
                    for mode in K_MODES:
                        available = mode == 'historical' or u == 0
                        k = budget['historical_k'] if mode == 'historical' else a
                        for method in METHODS:
                            r = dict(**case, k_mode=mode, method=method, k=k if available else np.nan,
                                     k_available=available)
                            if available:
                                if method == 'random':
                                    values = evaluate_random(y, risk.group_id, k, representation, kind, seed, today, days, mode)
                                else:
                                    values = evaluate_selection(y, risk.group_id, orders[method][:k], missing[method], source_bits)
                            else:
                                values = {f: np.nan for f in [
                                    'selected_known_positive', 'selected_unknown', 'selected_known_negative',
                                    'selected_unique', 'caught_unique', 'fallback_selected', 'fallback_selected_unique',
                                    'precision', 'recall', 'exact_precision', 'exact_recall',
                                    'precision_lower', 'precision_upper', 'recall_lower', 'recall_upper', 'random_expected_tp']}
                                values.update(caught_source_mask='', random_draws=0,
                                              missing_scores=int(missing[method].sum()) if method in missing else 0,
                                              missing_unique=risk.loc[missing[method], 'group_id'].nunique() if method in missing else 0)
                            r.update(values)
                            all_rows.append(r)
        print('Completed all fixed windows:', representation, flush=True)
    results, cases = pd.DataFrame(all_rows), pd.DataFrame(case_rows)
    assert len(results) == 73200 and len(cases) == 7320
    assert not results.duplicated(['case_id', 'k_mode', 'method']).any()
    metrics, pairs = summarize(results), paired_comparisons(results)
    support = support_summary(cases)
    results.to_csv(OUT / 'scenario_metrics.csv', index=False)
    cases.to_csv(OUT / 'case_audit.csv', index=False)
    pd.concat(rank_frames, ignore_index=True).to_parquet(OUT / 'rank_audit.parquet', index=False)
    unique_layouts.to_csv(OUT / 'unique_layouts.csv', index=False)
    fold_audit.to_csv(OUT / 'source_fold_audit.csv', index=False)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    pairs.to_csv(OUT / 'paired_comparisons.csv', index=False)
    support.to_csv(OUT / 'source_support.csv', index=False)
    (OUT / 'source_id_map.json').write_text(json.dumps(source_bits, indent=2) + '\n')
    plot_rankings(metrics, cases)
    write_report(results, metrics, cases, pairs, support, commit)
    after = {p: digest(p) for p in protected}
    assert before == after, 'An earlier tracked artifact changed'
    verify_baseline()
    code_paths = ['src/battery_analytics/hybrid_fleet.py', 'scripts/hybrid_fleet.py']
    manifest = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                    preregistration_commit=commit, preregistration_sha256=digest(PREREG),
                    repository_head=git('rev-parse', 'HEAD').decode().strip(), input_sha256=input_hashes,
                    code_sha256={p: digest(p) for p in code_paths},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    settings=dict(methods=METHODS, k_modes=K_MODES, random_draws=RANDOM_DRAWS,
                                  unique_seed=SEEDS[0], unique_offsets='first per-source copy in single-dataset v2 master',
                                  fallback='finite scores first; missing scores oldest first', rounding='floor(count+0.5)',
                                  unknown_outcomes='conditional metrics plus full bounds; oracle unavailable if any unknown'),
                    case_rows=len(cases), method_k_rows=len(results), source_held_out_combinations=int(fold_audit.source_held_out.sum()),
                    protected_files_unchanged=True, protected_sha256_before=before, protected_sha256_after=after,
                    result_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in OUT.iterdir() if p.is_file() and p.name != 'run_manifest.json'})
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(metrics[metrics.representation.eq('unique') & metrics.k_mode.eq('historical')].groupby(['validation', 'method'])[['precision', 'recall']].mean().to_string(), flush=True)
    print('Saved', OUT / 'results.md', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Refresh presentation while preserving all raw experiment scores')
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
        raw = ['scenario_metrics.csv', 'case_audit.csv', 'rank_audit.parquet', 'unique_layouts.csv',
               'source_fold_audit.csv', 'metrics.csv', 'paired_comparisons.csv', 'source_support.csv', 'source_id_map.json']
        raw_hashes = {p: digest(f'reports/hybrid_fleet/{p}') for p in raw}
        assert raw_hashes == {p: manifest['result_sha256'][p] for p in raw}
        frames = [pd.read_csv(OUT / p) for p in ['scenario_metrics.csv', 'metrics.csv', 'case_audit.csv',
                                                'paired_comparisons.csv', 'source_support.csv']]
        plot_rankings(frames[1], frames[2])
        write_report(*frames, commit)
        assert raw_hashes == {p: digest(f'reports/hybrid_fleet/{p}') for p in raw}
        assert manifest['protected_sha256_before'] == {p: digest(p) for p in manifest['protected_sha256_before']}
        manifest.setdefault('execution_code_sha256', manifest['code_sha256'])
        manifest['code_sha256'] = {p: digest(p) for p in manifest['code_sha256']}
        manifest['report_refresh'] = dict(utc=datetime.now(timezone.utc).isoformat(),
                                        reason='Clarify within/unseen comparisons, four incident histories, zero-K and flag-all cases; no rule or score changes',
                                        raw_artifacts_sha256_unchanged=raw_hashes)
        manifest['result_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in OUT.iterdir() if p.is_file() and p.name != 'run_manifest.json'}
        (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Refreshed presentation; all experiment rankings and metrics unchanged')
