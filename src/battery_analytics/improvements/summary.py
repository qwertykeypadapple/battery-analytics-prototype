"""Keep the main results documents current without modifying frozen metrics."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .core import ROOT, OUT, verify_baseline

BASELINE_MARKER = '<!-- frozen-baseline-details:start -->'


def sync_main_results():
    required = [OUT/'ablation.csv', OUT/'E9_calibration/paired_learning_curve.csv',
                ROOT/'reports/fleet_forecast/scenarios.csv']
    if not all(p.exists() for p in required):
        return False
    verify_baseline()
    a = pd.read_csv(required[0])
    names = ['Frozen_baseline','E1_cycle','E2_replace','E2_alongside','E3_delta',
             'E3_anchor','E4_combined','E5_filter','E8_forced_ica','E8_optional_ica']
    rows = []
    for name in names:
        h = a[a.experiment.eq(name)].set_index('validation')
        lo, transfer = h.loc['leave_one_cell_out'], h.loc['leave_one_dataset_out']
        rows.append({'Variant': name, 'Scored cells': int(lo.n_cells),
            'LOCO MAE (pp)': lo.cell_macro_mae_pp, 'LODO MAE (pp)': transfer.cell_macro_mae_pp,
            'Matched baseline LOCO (pp)': lo.matched_baseline_mae_pp if name != 'Frozen_baseline' else lo.cell_macro_mae_pp,
            'Matched baseline LODO (pp)': transfer.matched_baseline_mae_pp if name != 'Frozen_baseline' else transfer.cell_macro_mae_pp})
    ablation = pd.DataFrame(rows)
    intervals = pd.read_csv(OUT/'E5_filter/interval_calibration.csv')
    intervals = intervals[intervals.validation.eq('leave_one_cell_out') & intervals.nominal.eq(.95)].copy()
    intervals['Actual 95% interval coverage (%)'] = intervals.cell_macro_coverage * 100
    alerts = pd.read_csv(OUT/'E6_probabilistic/alert_metrics.csv')
    alerts = alerts[alerts.validation.eq('leave_one_cell_out')]
    alerts = alerts.groupby(['method','horizon_cycles'])[['events','timely','premature','late','none','no_eligible_history']].sum().reset_index()
    alerts['Scenario days'] = alerts.horizon_cycles / 2
    learning = pd.read_csv(required[1])
    cal_rows = []
    for dataset, h in learning[learning.phase.eq('development')].groupby('dataset'):
        h = h.set_index('k')
        row = {'Dataset': dataset}
        for k in [0,1,2,4]:
            row[f'{k} calibration cells: MAE ± SD (pp)'] = f'{h.loc[k,"mean_mae_pp"]:.2f} ± {h.loc[k,"sd_mae_pp"]:.2f}'
        cal_rows.append(row)
    fleet = pd.read_csv(required[2]); fleet = fleet[fleet.validation.eq('leave_one_cell_out')]
    count_rows = []
    for key, label in [('charge','Charge-based RUL'),('cycle','Cycle count'),('historical','Historical replacement rate')]:
        error = fleet[key+'_matched_error'].dropna()
        count_rows.append({'Method':label,'Matched scenarios':len(error),
                          'Count MAE (batteries)':error.abs().mean(),
                          'Count RMSE (batteries)':np.sqrt((error**2).mean()),
                          'Signed bias (batteries)':error.mean()})
    counts = pd.DataFrame(count_rows)
    coverage = fleet.groupby('dataset').agg(mean_matched_cells=('matched_cells','mean'),
        windows_with_actual_replacements=('matched_actual',lambda s:int((s>0).sum())),
        signed_count_bias=('charge_matched_error','mean')).reset_index()
    def table(frame, columns=None):
        return (frame[columns] if columns else frame).to_markdown(index=False,floatfmt='.2f')
    current = f'''## Current findings after E0–E9 and fleet forecasting

The prototype supports a **paid calibration and validation pilot**, with the
frozen charge-only workflow still the recommended default. Its headline remains
**3.72 SoH percentage points cell-macro MAE on 41 unseen lab cells**, rising to
**13.89 pp on an entirely unseen dataset**. Charging features beat the fixed
cycle-count-only curve (8.98 pp LOCO), but the additional pipelines do not show
a robust overall gain on matched cohorts. The tracker intervals are overconfident,
and fleet replacement budgeting is not validated.

The original model code, input tables and metrics remain frozen at `baseline-v1`.
The new experiments and the separate fleet study were preregistered and committed
before execution. Every experiment, including regressions and abstentions, is in
[the improvement report](improvements/results.md); the complete count tables are
in [the fleet forecasting report](fleet_forecast/results.md). This remains an
exploratory public-data study, with no Baaz pack or operational-savings evidence.

### SoH comparison and changed assumptions

All errors below are **cell-macro** percentage points. The matched baseline
columns use exactly each variant's scored rows. Warmup variants have fewer cells;
their lower unpaired average must not be advertised as a gain over the 41-cell
headline. E1/E4/E8 require known cycle history, E3_delta requires five entry
charges, and E3_anchor allows one independently measured entry capacity.
The unknown-age baseline still excludes cycle count and capacity anchors.

{table(ablation)}

The throughput additions' small LOCO gains do not compensate for much worse
transfer. Entry-relative features improve LODO on matched rows (7.58 vs 12.24 pp)
but worsen LOCO (4.34 vs 2.50 pp). The one-entry capacity offset, combined model,
Kalman tracker and ICA additions do not earn a change to the recommended default.
Raw/median errors, RMSE, pooled-row/equal-dataset results, per-cell results and
low-SoH bands for every variant remain in [the complete metrics](improvements/all_metrics.csv)
and [ablation](improvements/ablation.csv), with E9's repeated calibration results
reported separately in its [aggregate metrics](improvements/E9_calibration/aggregate_metrics.csv).

E0's **post-hoc** sensitivity restricted to at least 20 observations gives
2.50 pp across 36 cells, excluding five cells and six observations. It does not
replace 3.72 pp. NASA below-70% errors are 18.37 pp cold, 6.18 room and 38.72 warm; the warm group has
only THREE observations. Cold conditions explain part of the weakness, not all.

### Tracking uncertainty and replacement alerts

E5's nominal 95% bands are uncalibrated; actual cell-macro coverage is:

{table(intervals,['dataset','n_cells','Actual 95% interval coverage (%)','mean_width_pp'])}

E6/E7 retain E5's SoH estimates and add decisions, rather than independent new
SoH improvements. The table below scores the **first** alert against all 19
observed laboratory 70% crossings. Scenario days assume 2 swaps per day and one source
lab cycle per swap; these endpoints are not field failures or stranded riders.

{table(alerts,['method','Scenario days','events','timely','premature','late','none','no_eligible_history'])}

At 14 days, timely alerts improve from 0/19 to 2/19, with substantial misses and
coverage gaps. At 60 days they remain 6/19; at 90 days they rise from 6/19 to 8/19.
These results do not validate replacement decisions or calibrated risk claims.
E7 catches no true-fast cells in CALCE/NASA at its fixed decision point and 1/3
in Oxford. [Interval calibration](improvements/E5_filter/interval_calibration.csv),
[alert results](improvements/E6_probabilistic/alert_metrics.csv), and
[fast-degradation evaluation](improvements/E7_fast/classification_metrics.csv)
retain the denominators and uncertainty limitations.

### Target calibration: evidence and limits

The source-only E2 model stays frozen; the correction is an additive offset.
Each calibration cell supplies its **complete labelled laboratory trajectory**,
not just a commissioning capacity check. Values below average five correlated
calibration-cell choices; SD is not an independent-sample confidence interval.

{table(pd.DataFrame(cal_rows))}

CALCE/Oxford improve over the uncalibrated E2 source model, but that starting
model is worse than the frozen reference. NASA remains poor and adding cells
does not reliably help. Changing test cohorts and a separately locked cell per
dataset are reported explicitly, with [matched frozen-reference comparisons](improvements/E9_calibration/paired_learning_curve.csv).
This does not establish how many Baaz packs or reference checks are sufficient.

![Calibration learning curve](improvements/figures/calibration_learning_curve.png)

### Fleet replacement counts: not yet replacement budgeting

Four preregistered today points per dataset, with 30, 60 and 90 scenario-day horizons;
only held-out raw-linear RUL forecasts are counted. This separate experiment
uses the preregistered deterministic fallback, **not calibrated probabilities**.
Below, errors are conditional on usable forecasts from both methods and known
outcomes. Full-fleet unknown outcomes and missing forecasts remain explicit
count bounds in [the scenario table](fleet_forecast/scenarios.csv).

{table(counts)}

The charge model's 0.37-battery MAE advantage over 0.43 historical-rate MAE is
fragile. Matched CALCE/NASA scenarios average only 2–3 cells, and Oxford's matched
windows have zero actual replacements. Near-zero mean bias does not prove
that individual errors cancel: CALCE slightly overpredicts while NASA
underpredicts, and unconditional full-fleet bias remains uncertain.

{table(coverage)}

![Fleet counts versus observed outcomes](fleet_forecast/replacement_counts.png)

### Updated pitch conclusion and dashboard

Keep the conservative frozen SoH headline and propose a target-specific
calibration/validation pilot against Baaz's existing BMS analytics. Do not claim
accurate replacement dates/budgets, calibrated 95% intervals, pack-level accuracy,
proven Delhi heat performance, prevented failures, life extension or energy
savings. The dashboard defaults to baseline and exposes experimental E8 and
E5/E6 views, uncertainty limitations, and a separate **Fleet budgeting** tab.

The details below are the **frozen baseline report**, retained for audit.
Its feature exclusions, model tables, first-alert policy and proposed next
experiments describe baseline-v1; completed follow-up experiments are above.

'''
    main = ROOT/'reports/results_summary.md'
    text = main.read_text()
    if BASELINE_MARKER in text:
        baseline_body = text.split(BASELINE_MARKER,1)[1].lstrip()
    else:
        baseline_body = text.split('\n',1)[1].lstrip()
        baseline_body = re.sub(r'(?m)^(#{2,}) ',r'#\1 ',baseline_body)
        baseline_body = baseline_body.replace('**No SoH regressor receives','**No frozen-baseline SoH regressor receives')
        baseline_body = baseline_body.replace('No incremental-capacity peak feature is used.',
                                             'No incremental-capacity peak feature is used in the frozen baseline.')
    body = current+'## Frozen baseline details\n\n'+BASELINE_MARKER+'\n\n'+baseline_body
    main.write_text('# Battery analytics prototype: results summary\n\n'+body)
    (ROOT/'reports/skeptical_review.md').write_text('# Skeptical review for the Baaz Bikes pitch\n\n'+body)
    print('Updated main results summary and skeptical review; frozen metrics unchanged.',flush=True)
    return True
