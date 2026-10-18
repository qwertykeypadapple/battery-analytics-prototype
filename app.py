from __future__ import annotations
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from battery_analytics.common import load_config
from battery_analytics.fleet import fleet_snapshot
from battery_analytics.rul import forecast, reference_histories

st.set_page_config(page_title='Battery fleet health', page_icon='🔋', layout='wide')


@st.cache_data
def load_artifacts(stamps):
    return (pd.read_parquet(ROOT / 'reports/evaluation/predictions.parquet'),
            pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet'),
            pd.read_csv(ROOT / 'reports/inventory/cells.csv'),
            pd.read_csv(ROOT / 'reports/evaluation/metrics_per_dataset.csv'),
            pd.read_csv(ROOT / 'reports/evaluation/metrics_overall.csv'),
            pd.read_csv(ROOT / 'reports/rul/rul_metrics.csv'),
            pd.read_csv(ROOT / 'data/processed/feature_coverage.csv'))


required = [ROOT / 'reports/evaluation/predictions.parquet', ROOT / 'reports/rul/rul_metrics.csv']
if not all(p.exists() for p in required):
    st.error('Build the evaluation artifacts first: python scripts/run_all.py')
    st.stop()

config = load_config()
all_predictions, references, cells, metrics, overall, rul_metrics, coverage = load_artifacts(tuple(p.stat().st_mtime_ns for p in required))
st.title('Battery fleet health')
st.caption('Charging-based estimates from laboratory cells. This demonstration contains no Baaz customer data.')

with st.sidebar:
    st.header('Fleet simulation')
    model_view = st.selectbox('Model view', ['Frozen baseline (recommended)',
        'Inner-selected features (E8; known history)', 'Probabilistic tracker (E5/E6; experimental)'])
    improved = not model_view.startswith('Frozen')
    tracker = model_view.startswith('Probabilistic')
    datasets = st.multiselect('Datasets', ['calce', 'nasa', 'oxford'], default=['calce'])
    validation_label = st.radio('Unseen-battery test', ['New cell within dataset', 'Entire dataset held out'])
    protocol = 'leave_one_cell_out' if validation_label == 'New cell within dataset' else 'leave_one_dataset_out'
    signal = st.radio('Prediction history', ['Raw', 'Trailing median'])
    value = 'prediction' if signal == 'Raw' else 'prediction_smoothed'
    signal_name = 'raw' if signal == 'Raw' else 'smoothed'
    method = st.selectbox('RUL trend model', ['linear', 'exponential'])
    swaps = st.number_input('Swaps per battery per day', min_value=.1, max_value=10., value=float(config['rul']['swaps_per_day']), step=.5)
    cycles_per_swap = st.number_input('Assumed lab cycles per swap', min_value=.1, max_value=3., value=float(config['rul']['test_cycles_per_swap']), step=.1)
    lookback = st.slider('Recent observations for RUL', min_value=6, max_value=30, value=int(config['rul']['lookback_points']))
    st.subheader('Alert settings')
    watch_soh_pct = st.slider('Watch below SoH (%)', min_value=70, max_value=100, value=80)
    replace_days, watch_days = st.slider('Replace soon / Watch thresholds (days)', min_value=0, max_value=365, value=(14, 45))
    st.caption('Days are a scenario conversion. Partial swaps do not automatically equal full laboratory cycles.')
    probability_horizon = st.selectbox('Probability alert horizon (lab cycles)', [28, 120, 180]) if tracker else 28

if not datasets:
    st.info('Select at least one dataset.')
    st.stop()

chosen = all_predictions[all_predictions.model.eq('training_selected') & all_predictions.variant.eq('no_temperature') & all_predictions.validation.eq(protocol)]
if improved:
    variant_path = ROOT / ('reports/improvements/E6_probabilistic/predictions.parquet' if tracker else
                           'reports/improvements/E8_optional_ica/predictions.parquet')
    if not variant_path.exists():
        st.info('Run the preregistered improvement stages before selecting this view.')
        st.stop()
    chosen = pd.read_parquet(variant_path)
    chosen = chosen[chosen.validation.eq(protocol)]
chosen = chosen[chosen.dataset.isin(datasets)]
inventory = cells[cells.dataset.isin(datasets)]
if not len(chosen):
    st.warning('No validated predictions exist for this selection.')
    st.stop()
max_cycle = max(1, int(chosen.cycle_index.max()))
last_cycles = chosen.groupby(['dataset', 'cell_id']).cycle_index.max()
default_today = max(1, min(max_cycle, int(last_cycles.min() * .60)))
today = st.slider('Today cycle', min_value=0, max_value=max_cycle, value=default_today, step=1)
rul_config = dict(config['rul'], lookback_points=lookback, swaps_per_day=swaps, test_cycles_per_swap=cycles_per_swap)
snapshot = fleet_snapshot(chosen, inventory, today, rul_config, value=value, method=method,
                          watch_soh=watch_soh_pct / 100, replace_days=replace_days, watch_days=watch_days)
fast_path = ROOT / 'reports/improvements/E7_fast/cell_flags.csv'
fast = pd.read_csv(fast_path) if fast_path.exists() else pd.DataFrame()
if tracker:
    for index, row in snapshot.iterrows():
        h = chosen[chosen.dataset.eq(row.dataset) & chosen.cell_id.eq(row.cell)]
        prefix = h[h.cycle_index <= today].sort_values(['cycle_index', 'charge_end_time_s'])
        if not len(prefix) or today > h.cycle_index.max(): continue
        last = prefix.iloc[-1]
        probability = last[f'probability_{probability_horizon}']
        threshold = last[f'alert_threshold_{probability_horizon}']
        snapshot.loc[index, 'soh_pct'] = last.prediction * 100
        snapshot.loc[index, 'Crossing probability (uncalibrated)'] = probability
        snapshot.loc[index, 'Learned alert threshold'] = threshold
        snapshot.loc[index, 'cycles_left'] = last.rul_median if np.isfinite(last.rul_median) else np.nan
        snapshot.loc[index, 'scenario_days_left'] = last.rul_median / (swaps * cycles_per_swap) if np.isfinite(last.rul_median) else np.nan
        snapshot.loc[index, 'status'] = ('Check estimate' if not 0 < last.prediction <= 1.5 else
            ('Replace soon' if probability >= threshold else ('Watch' if last.prediction < watch_soh_pct / 100 else 'Healthy')))
        snapshot.loc[index, 'evidence'] = 'Experimental probability policy; probabilities are not calibrated'
        if len(fast):
            flag = fast[fast.validation.eq(protocol) & fast.dataset.eq(row.dataset) & fast.cell_id.eq(row.cell)]
            if len(flag):
                f = flag.iloc[0]
                snapshot.loc[index, 'Fast degradation'] = ('Flagged' if f.flag else 'Not flagged') if f.available and today >= f.decision_cycle else 'Insufficient history'

fleet_tab, validation_tab, methods_tab, improvement_tab, budgeting_tab = st.tabs(['Fleet', 'Validation', 'Data and methods', 'Improvement experiments', 'Fleet budgeting'])
with fleet_tab:
    if tracker:
        st.warning('Experimental tracker: nominal 95% SoH bands cover only 16% CALCE, 82% NASA and 54% Oxford on a cell-macro basis. The 14-day policy gives 2 timely first alerts out of 19. These probabilities and replacement labels are not operationally calibrated.')
    elif improved:
        st.warning('This known-history candidate uses cycle count and a five-charge entry reference. It is selected inside each training fold; it does not establish a gain over the frozen baseline. No E5 uncertainty or probability is silently assigned to this E8 model.')
    else:
        st.warning('The saved 14-day/raw-linear laboratory backtest has zero timely first alerts across 19 observed capacity-threshold events. The replacement labels below demonstrate the workflow; their timing is not an operationally validated policy.')
    a, b, c, d = st.columns(4)
    a.metric('Cells in view', len(snapshot))
    b.metric('Replace soon', int(snapshot.status.eq('Replace soon').sum()))
    c.metric('Watch', int(snapshot.status.eq('Watch').sum()))
    d.metric('No current data', int(snapshot.status.isin(['No data', 'History ended', 'Check estimate']).sum()))
    st.caption('Every estimate below comes from a held-out cell or a held-out dataset. The final all-data refit does not supply these demo predictions.')
    display = snapshot.rename(columns={'dataset': 'Dataset', 'cell': 'Cell', 'as_of_cycle': 'Latest eligible cycle',
                    'soh_pct': 'Predicted SoH (%)', 'cycles_left': 'Cycles left', 'scenario_days_left': 'Scenario days left',
                    'status': 'Status', 'evidence': 'Reason', 'forecast_reason': 'RUL support'})
    st.dataframe(display, hide_index=True, use_container_width=True,
                 column_config={'Predicted SoH (%)': st.column_config.NumberColumn(format='%.1f'),
                                'Cycles left': st.column_config.NumberColumn(format='%.0f'),
                                'Scenario days left': st.column_config.NumberColumn(format='%.1f')})
    st.download_button('Download fleet snapshot', snapshot.to_csv(index=False), file_name='fleet_snapshot.csv', mime='text/csv')
    st.caption('A blank RUL means the recent trend does not support a forecast. History ended means the slider exceeds the recorded eligible history, rather than evidence of a healthy battery.')
    choices = [f'{r.dataset} / {r.cell_id}' for _, r in inventory.iterrows()]
    selected = st.selectbox('Battery detail', choices)
    dataset, cell = selected.split(' / ')
    h = chosen[chosen.dataset.eq(dataset) & chosen.cell_id.eq(cell)]
    actual = reference_histories(references)
    actual = actual[actual.dataset.eq(dataset) & actual.cell_id.eq(cell)]
    show_future = st.checkbox('Show later recorded data for retrospective inspection', value=False)
    visible_h = h if show_future else h[h.cycle_index <= today]
    visible_actual = actual if show_future else actual[actual.cycle_index <= today]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=visible_actual.cycle_index, y=visible_actual.soh * 100, name='Discharge reference', mode='lines+markers', line_color='#718096'))
    if tracker and len(visible_h):
        figure.add_trace(go.Scatter(x=visible_h.cycle_index, y=visible_h.upper_95 * 100, mode='lines', line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(x=visible_h.cycle_index, y=visible_h.lower_95 * 100, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(49,130,206,.15)', name='Uncalibrated 95% tracker band'))
    figure.add_trace(go.Scatter(x=visible_h.cycle_index, y=visible_h.prediction * 100, name='Held-out raw estimate', mode='lines', line_color='#3182ce', opacity=.65))
    figure.add_trace(go.Scatter(x=visible_h.cycle_index, y=visible_h.prediction_smoothed * 100, name='Trailing median estimate', mode='lines', line_color='#00847e'))
    estimate = forecast(h, today, rul_config, method, value)
    if not tracker and estimate['forecast_reason'] == 'estimated' and today <= h.cycle_index.max():
        origin = estimate['as_of_cycle']
        horizon = min(estimate['predicted_rul_cycles'] * 1.15, rul_config['maximum_horizon_cycles'])
        x = np.linspace(origin, origin + max(1, horizon), 100)
        y = estimate['fitted_current_soh'] + estimate['fitted_slope'] * (x - origin) if method == 'linear' else estimate['fitted_current_soh'] * np.exp(estimate['fitted_slope'] * (x - origin))
        figure.add_trace(go.Scatter(x=x, y=100 * y, name=f'{method.title()} forecast', line=dict(color='#dd6b20', dash='dash')))
    figure.add_hline(y=config['rul']['end_of_life_soh'] * 100, line_dash='dash', line_color='#c53030', annotation_text='70% end of life')
    figure.add_vline(x=today, line_dash='dot', line_color='#718096')
    figure.update_layout(title=f'{dataset.upper()} {cell}', xaxis_title='Source laboratory cycle index', yaxis_title='SoH (% of rated Ah)', height=470, legend=dict(orientation='h', y=-.2), margin=dict(t=60, b=90))
    st.plotly_chart(figure, use_container_width=True)
    if tracker:
        st.write('RUL uses the current Gaussian SoH/rate state and fixed posterior draws. Infinite lifetime draws remain unavailable in the fleet table.')
    else:
        st.write(f"RUL support: {estimate['forecast_reason'].replace('_', ' ')}. History points used: {estimate['history_points']}.")
    st.caption('The dotted line marks the selected cycle. Later data, when shown, are visual context only and never enter the forecast. Laboratory cell alerts require pack-specific validation before maintenance decisions.')
    if tracker:
        st.caption('Uncertainty and probability alerts refer to the raw E5 state, even if the supplementary trailing-median history is selected. Saved probability thresholds come only from inner validation cells; display controls do not retune them.')
        st.caption('The fast-degradation flag is the saved decision at the 20th E4 observation, after five warmup charges. It is unavailable before that point and is not a newly evaluated late-life detector.')

with validation_tab:
    st.subheader('Accuracy on batteries excluded from training')
    st.write('MAE and RMSE are in SoH percentage points. Cell-macro results give every cell equal weight; pooled rows would let CALCE dominate.')
    comparison = metrics[metrics.variant.eq('no_temperature')].copy()
    st.dataframe(comparison[['validation', 'dataset', 'model', 'n_cells', 'n_rows', 'macro_cell_mae_raw_pp', 'macro_cell_rmse_raw_pp', 'macro_cell_mae_smoothed_pp', 'macro_cell_rmse_smoothed_pp']], hide_index=True, use_container_width=True)
    selected_metrics = comparison[comparison.model.eq('training_selected')]
    st.subheader('Transfer across datasets')
    cross = selected_metrics[selected_metrics.validation.eq('leave_one_dataset_out')]
    chart = go.Figure(go.Bar(x=cross.dataset, y=cross.macro_cell_mae_raw_pp, marker_color=['#00847e', '#c53030', '#3182ce']))
    chart.update_layout(yaxis_title='Cell-macro MAE (SoH percentage points)', xaxis_title='Entire held-out dataset', height=320)
    st.plotly_chart(chart, use_container_width=True)
    st.warning('The cross-dataset failures are material. Performance on these laboratory cells does not establish accuracy on a new Baaz pack design or charging protocol.')
    st.subheader('RUL backtests')
    st.caption('First 70% reference crossing. Exact errors exclude right-censored cells and cells already below 70% at their first valid reference. Coverage and early predictions on censored cells are shown separately.')
    r = rul_metrics[rul_metrics.level.eq('dataset') & rul_metrics.signal.eq(signal_name)]
    st.dataframe(r[['validation', 'dataset', 'method', 'n_anchors', 'observed_crossing_anchors', 'n_scored', 'n_scored_cells', 'mae_cycles', 'rmse_cycles', 'forecast_coverage', 'n_censored_forecasts', 'censor_bound_violation_rate']], hide_index=True, use_container_width=True)
    st.caption('RUL backtest settings use the saved 12-point history and 70% threshold. Interactive fleet controls change the simulation, not the saved validation experiment.')
    alarm_path = ROOT / 'reports/review/alarm_metrics.csv'
    if alarm_path.exists():
        st.subheader('First replacement-alert timing')
        alarm = pd.read_csv(alarm_path)
        alarm = alarm[alarm.validation.eq('leave_one_cell_out') & alarm.signal.eq('raw')]
        st.dataframe(alarm[['dataset', 'observed_threshold_events', 'timely_first_alerts', 'premature_first_alerts',
                            'late_first_alerts', 'no_alerts', 'no_eligible_history', 'observation_false_positive_rate',
                            'observation_precision']], hide_index=True, use_container_width=True)
        st.caption('A first alarm more than 14 scenario days before the 70% laboratory capacity endpoint is premature. An eventual later in-window alarm does not erase that false early replacement. These are capacity endpoints, not field breakdowns.')
    st.subheader('Temperature and CV comparisons')
    st.dataframe(metrics[metrics.variant.ne('no_temperature')][['validation', 'variant', 'dataset', 'model', 'n_rows', 'n_cells', 'macro_cell_mae_raw_pp', 'macro_cell_mae_smoothed_pp']], hide_index=True, use_container_width=True)

with methods_tab:
    st.subheader('Feature and label coverage')
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.markdown('''The common model uses only charging times within a configurable upper voltage window and charging current divided by rated Ah. Temperature and CV taper features are separate variants. No discharge capacity, cell identifier, dataset identifier, cycle number or total charge capacity enters the SoH regressors.

The 3.9–4.18 V window is a laboratory configuration for these cells. A pack implementation requires Baaz’s cell specification and per-cell voltage telemetry. Cycles that begin above 3.9 V do not receive an estimate from this window.

The RUL models extrapolate recent predicted SoH. A flat, increasing or very long extrapolation returns an explicit unavailable result. Swaps-to-days conversion is an editable assumption, especially for partial swaps.''')
    inventory_text = (ROOT / 'reports/inventory_report.md').read_text()
    with st.expander('Complete data inventory and source documentation'):
        st.markdown(inventory_text)
    summary_path = ROOT / 'reports/results_summary.md'
    if summary_path.exists():
        st.download_button('Download results summary', summary_path.read_text(), file_name='results_summary.md', mime='text/markdown')

with improvement_tab:
    st.subheader('Preregistered E0–E9 experiments')
    st.write('The frozen baseline remains the recommended default. New assumptions, lost prediction coverage and negative results remain visible. All displayed experiments use held-out cells.')
    ablation = ROOT / 'reports/improvements/ablation.csv'
    if ablation.exists(): st.dataframe(pd.read_csv(ablation), hide_index=True, use_container_width=True)
    for title, relative in [('SoH interval calibration', 'E5_filter/interval_calibration.csv'),
            ('First alerts by horizon', 'E6_probabilistic/alert_metrics.csv'),
            ('Fast-degradation detection', 'E7_fast/classification_metrics.csv'),
            ('Calibration learning curve', 'E9_calibration/learning_curve.csv')]:
        path = ROOT / 'reports/improvements' / relative
        if path.exists():
            with st.expander(title): st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)
    learning = ROOT / 'reports/improvements/figures/calibration_learning_curve.png'
    if learning.exists(): st.image(str(learning), use_column_width=True)
    report = ROOT / 'reports/improvements/results.md'
    if report.exists(): st.download_button('Download improvement findings', report.read_text(), file_name='improvement_results.md')

with budgeting_tab:
    st.subheader('Fleet replacement-count backtest')
    st.warning('Coverage is incomplete and fleet sizes are small. This experiment does not validate replacement budgeting. Counts use deterministic raw-linear RUL forecasts; they are not sums of calibrated probabilities.')
    fleet_root = ROOT / 'reports/fleet_forecast'
    if (fleet_root / 'scenarios.csv').exists():
        scenarios = pd.read_csv(fleet_root / 'scenarios.csv')
        st.dataframe(scenarios[scenarios.validation.eq(protocol)], hide_index=True, use_container_width=True)
        st.image(str(fleet_root / 'replacement_counts.png'), use_column_width=True)
        st.download_button('Download fleet forecast report', (fleet_root / 'results.md').read_text(), file_name='fleet_forecast_results.md')
