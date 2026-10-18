"""Preregistered fleet-count backtest, using held-out predictions only."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from battery_analytics.common import load_config
from battery_analytics.improvements.core import baseline, verify_baseline, BASE
from battery_analytics.rul import forecast, reference_histories

OUT = ROOT / 'reports/fleet_forecast'


def asof_forecast(h, today, config):
    """No reference capacity is passed into the prediction function."""
    past = h[h.cycle_index <= today]
    if not len(past): return np.nan, 'no_prefix'
    if today > h.cycle_index.max(): return np.nan, 'history_ended'
    f = forecast(past, today, config, 'linear', 'prediction')
    remaining = f['predicted_rul_cycles']
    if not np.isfinite(remaining): return np.nan, f['forecast_reason']
    crossing = f['as_of_cycle'] + remaining
    return float(crossing - today), 'predicted_overdue' if crossing <= today else 'estimated'


def historical_rate(eols, checks, today, start):
    observed = 0
    for _, e in eols[eols.censoring.eq('observed_crossing')].iterrows():
        h = checks[checks.cell_id.eq(e.cell_id)]
        below = h[h.soh <= .7]
        if len(below) and below.cycle_index.min() <= today: observed += 1
    return observed / max(1., today - start), observed


def known_outcome(e, today, horizon):
    if e.censoring == 'observed_crossing':
        return float(today < e.eol_cycle <= today + horizon)
    if e.censoring == 'right_censored' and e.last_reference_cycle >= today + horizon:
        return 0.
    return np.nan


def run():
    verify_baseline()
    config = load_config()['rul']
    p = baseline()
    cycle = pd.read_parquet(ROOT / 'reports/improvements/E1_cycle/predictions.parquet')
    refs = reference_histories(pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet'))
    eols = pd.read_csv(BASE / 'end_of_life.csv')
    inventory = pd.read_csv(ROOT / 'reports/inventory/cells.csv')
    rows, cells = [], []
    for dataset, fleet in inventory.groupby('dataset'):
        r = refs[refs.dataset.eq(dataset)]
        e = eols[eols.dataset.eq(dataset)]
        start, end = r.cycle_index.min(), r.cycle_index.max()
        for protocol in ['leave_one_cell_out', 'leave_one_dataset_out']:
            for fraction in [.2, .4, .6, .8]:
                today = int(round(start + fraction * (end - start)))
                rate, past_events = historical_rate(e, r, today, start)
                for days in [30, 60, 90]:
                    horizon = days * 2
                    records = []
                    for _, cell in fleet.iterrows():
                        h = r[r.cell_id.eq(cell.cell_id)]
                        past = h[h.cycle_index <= today]
                        already = bool(len(past) and (past.soh <= .7).any())
                        truth = e[e.cell_id.eq(cell.cell_id)].iloc[0]
                        cp = p[p.validation.eq(protocol) & p.dataset.eq(dataset) & p.cell_id.eq(cell.cell_id)]
                        ap = cycle[cycle.validation.eq(protocol) & cycle.dataset.eq(dataset) & cycle.cell_id.eq(cell.cell_id)]
                        remaining, reason = asof_forecast(cp, today, config)
                        cremaining, creason = asof_forecast(ap, today, config)
                        actual = known_outcome(truth, today, horizon)
                        record = dict(dataset=dataset, validation=protocol, today=today, days=days,
                            cell_id=cell.cell_id, already_below=already,
                            retrospective_preexisting=bool(truth.censoring == 'observed_crossing' and truth.eol_cycle <= today and not already),
                            actual=actual, charge_rul=remaining, charge_reason=reason, cycle_rul=cremaining,
                            cycle_reason=creason, charge_incident=float(0 < remaining <= horizon) if np.isfinite(remaining) else np.nan,
                            cycle_incident=float(0 < cremaining <= horizon) if np.isfinite(cremaining) else np.nan)
                        records.append(record); cells.append(record)
                    a = pd.DataFrame(records); a = a[~a.already_below]
                    actual_lo = int(a.actual.sum()); unknown = int(a.actual.isna().sum())
                    predicted = int(a.charge_incident.sum()); cycle_count = int(a.cycle_incident.sum())
                    historical = min(len(a), rate * horizon)
                    matched = a[a.actual.notna() & a.charge_incident.notna() & a.cycle_incident.notna()]
                    ma = float(matched.actual.sum())
                    mh = historical * len(matched) / len(a) if len(a) else 0.
                    row = dict(dataset=dataset, validation=protocol, fraction=fraction, today=today, days=days,
                        horizon_cycles=horizon, inventory_cells=len(fleet), at_risk_cells=len(a),
                        already_below=len(fleet)-len(a), retrospective_preexisting=int(a.retrospective_preexisting.sum()),
                        actual_count_lower=actual_lo, actual_count_upper=actual_lo + unknown, unknown_actual=unknown,
                        charge_count_subtotal=predicted, charge_unknown=int(a.charge_incident.isna().sum()),
                        charge_count_upper=predicted+int(a.charge_incident.isna().sum()),
                        cycle_count_subtotal=cycle_count, cycle_unknown=int(a.cycle_incident.isna().sum()),
                        predicted_overdue=int(a.charge_reason.eq('predicted_overdue').sum()),
                        cycle_predicted_overdue=int(a.cycle_reason.eq('predicted_overdue').sum()),
                        historical_rate_count=historical, past_observed_events=past_events,
                        matched_cells=len(matched), matched_actual=ma,
                        matched_charge=float(matched.charge_incident.sum()),
                        matched_cycle=float(matched.cycle_incident.sum()), matched_historical=mh)
                    for name, value in [('charge', predicted), ('cycle', cycle_count), ('historical', historical)]:
                        row[name+'_subtotal_signed_error_low'] = value-(actual_lo+unknown)
                        row[name+'_subtotal_signed_error_high'] = value-actual_lo
                        row[name+'_matched_error'] = row['matched_'+name]-ma if len(matched) else np.nan
                    rows.append(row)
    result = pd.DataFrame(rows); audit = pd.DataFrame(cells)
    result.to_csv(OUT / 'scenarios.csv', index=False); audit.to_csv(OUT / 'cell_audit.csv', index=False)
    summaries = []
    for key, g in result.groupby(['validation', 'dataset', 'days']):
        for name in ['charge', 'cycle', 'historical']:
            v = g[name+'_matched_error'].dropna()
            summaries.append(dict(validation=key[0], dataset=key[1], days=key[2], method=name,
                scored_scenarios=len(v), mean_matched_cells=float(g.matched_cells.mean()),
                mae_count=float(v.abs().mean()), rmse_count=float(np.sqrt((v*v).mean())), signed_bias_count=float(v.mean())))
    summary = pd.DataFrame(summaries); summary.to_csv(OUT / 'metrics.csv', index=False)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex='row')
    for i, dataset in enumerate(['calce','nasa','oxford']):
        for j, days in enumerate([30,60,90]):
            ax=axes[i,j]; g=result[result.dataset.eq(dataset)&result.days.eq(days)&result.validation.eq('leave_one_cell_out')]
            ax.fill_between(g.today, g.actual_count_lower, g.actual_count_upper, color='#777777', alpha=.2, label='Actual censoring range')
            for col, label, color in [('actual_count_lower','Observed incidents','#333333'),('charge_count_subtotal','Charge subtotal','#00847e'),
                    ('cycle_count_subtotal','Cycle subtotal','#b95b20'),('historical_rate_count','Historical rate','#3976af')]:
                ax.plot(g.today,g[col],marker='o',label=label,color=color,markersize=4)
            ax.set_title(f'{dataset.upper()} / {days} scenario days'); ax.set_ylabel('Batteries'); ax.set_xlabel('Today: source cycle')
            ax.grid(alpha=.2)
    handles, labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=5)
    fig.suptitle('Fleet replacement forecasts: held-out cells; subtotals omit unavailable forecasts',fontsize=13)
    fig.tight_layout(rect=[0,.055,1,.96]); fig.savefig(OUT/'replacement_counts.png',dpi=170); plt.close(fig)
    loco=result[result.validation.eq('leave_one_cell_out')]
    totals={name: dict(mae=float(loco[name+'_matched_error'].abs().mean()), bias=float(loco[name+'_matched_error'].mean())) for name in ['charge','cycle','historical']}
    table=loco[['dataset','today','days','inventory_cells','already_below','charge_count_subtotal','charge_unknown','cycle_count_subtotal','cycle_unknown',
                'historical_rate_count','actual_count_lower','actual_count_upper','matched_cells','matched_actual','charge_matched_error','cycle_matched_error','historical_matched_error']]
    text=f'''# Fleet replacement-count forecasting

The current charge-based fleet-count forecast does not establish reliable replacement budgeting: many fleet members lack usable forecasts or complete follow-up. On the matched, fully observed subset its mean absolute count error is {totals['charge']['mae']:.2f} batteries, versus {totals['cycle']['mae']:.2f} for cycle count and {totals['historical']['mae']:.2f} for historical replacement rate; this comparison is conditional on coverage. These small, overlapping retrospective fleets are insufficient for a commercial replacement-budgeting accuracy claim.

## What was tested

Preregistered fixed today fractions 20/40/60/80% of each recorded fleet age range;
30/60/90 days =60/120/180 source cycles. Only held-out SoH histories enter the
raw linear RUL forecasts. Probabilities were unavailable at registration: counts
are deterministic forecast subtotals, not expected counts from calibrated risks.

NASA has 34 inventory cells, CALCE7, Oxford8. Laboratory ages are artificially
synchronised; they do not describe a real fleet's commissioning-date mix. Each
cell's own model excluded that cell from training; LOCO still uses other fleet
members' full labelled histories. This is retrospective cross-cell validation,
not a training-data calendar cutoff or a prospective fleet trial.

## Predicted vs actual counts, primary LOCO protocol

`charge_unknown`/`cycle_unknown` are unavailable incident forecasts, not predicted
healthy cells. Already-below cells are outside incident counts. Actual counts
are bounds when follow-up ends early. Errors are exact only on the explicitly
matched subset (`matched_cells`, both forecasts usable and outcome known).
Full-fleet subtotal error bounds and all LODO rows are in [scenarios.csv](scenarios.csv).

{table.to_markdown(index=False,floatfmt='.2f')}

## Error and systematic bias

Positive signed bias means too many predicted replacements. Matched LOCO mean
bias: charge {totals['charge']['bias']:+.2f}, cycle {totals['cycle']['bias']:+.2f},
historical rate {totals['historical']['bias']:+.2f} batteries. This is NOT proof that
individual errors cancel: conditioning on available forecasts and follow-up can
change bias. Full-fleet prediction and outcome bounds are too incomplete to infer
unconditional bias. Overlapping windows are correlated; no independent-sample CI
or significance claim is made.

{summary.to_markdown(index=False,floatfmt='.3f')}

![Counts over time](replacement_counts.png)

## Missing histories and small samples

Distinct reason counts (repeated across horizons, not independent batteries):

{audit[~audit.already_below].groupby(['validation','dataset','charge_reason']).size().rename('scenario_cell_records').reset_index().to_markdown(index=False)}

No true incident date is assigned to initially below-threshold cells. A prior
observed below-threshold check excludes a cell using only information available
at today. Interpolated crossings already before today but first observed later
are recorded as retrospective pre-existing events, not forecast incidents.
Oxford's 100-cycle checks are coarse relative to these horizons. The entire
inventory contains only 19 observed crossings (7 CALCE,10 NASA,2 Oxford); four
today points and three horizons do not create more independent endpoint events.

## Pitch interpretation

Defensible: the software can aggregate held-out battery forecasts and expose
count uncertainty, missing data and simple baselines. Unsupported: validated
fleet budgeting, unbiased replacement counts, accurate Baaz pack budgets or
operational cost savings. A prospective larger-fleet evaluation with recorded
commissioning ages, independent capacity checks, complete horizon follow-up and
calibrated probabilities is needed. Report both count accuracy and forecast
coverage; improve SoH/rate bias before relying on error cancellation.

## Reproduction

`python scripts/fleet_forecast.py` requires the frozen baseline and E1 artifacts.
No fitting, tuning, threshold selection or future-reference input to the forecast
occurs in this script. See [preregistration.md](preregistration.md),
[cell_audit.csv](cell_audit.csv) and [metrics.csv](metrics.csv).
'''
    (OUT/'results.md').write_text(text)
    print('Matched LOCO mean count errors and signed bias:',totals)
    print('Saved',OUT/'results.md')


if __name__ == '__main__': run()
