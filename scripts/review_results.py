"""Additional skeptical review requested after the initial validation run.

No thresholds, features or model hyperparameters are tuned by this analysis.
Replacement alarms below concern a 70% laboratory-capacity endpoint, not field
breakdowns or safety events. Saved results retain every denominator.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT / 'src'))
from battery_analytics.common import load_config, write_json
from battery_analytics.rul import forecast
from battery_analytics.models import fit_estimator
import json

OUT = ROOT / 'reports/review'
OUT.mkdir(parents=True, exist_ok=True)


def accuracy_review(pred, cycles):
    selected = pred[pred.model.eq('training_selected') & pred.variant.eq('no_temperature')].copy()
    selected['band'] = pd.cut(selected.soh * 100, [-np.inf, 70, 80, 90, 110, np.inf],
                              labels=['below_70', '70_to_80', '80_to_90', '90_to_110', 'above_110'], right=False)
    rows = []
    for key, g in selected.groupby(['validation', 'dataset', 'band'], observed=True):
        row = dict(zip(['validation', 'dataset', 'band'], key), n_rows=len(g), n_cells=g.cell_id.nunique())
        for field, signal in [('prediction', 'raw'), ('prediction_smoothed', 'smoothed')]:
            temp = pd.DataFrame({'cell': g.cell_id, 'error': (g[field] - g.soh) * 100})
            cell_mae = temp.groupby('cell').error.apply(lambda s: s.abs().mean())
            cell_rmse = temp.groupby('cell').error.apply(lambda s: np.sqrt((s ** 2).mean()))
            row[signal + '_mae_pp'] = float(cell_mae.mean())
            row[signal + '_rmse_pp'] = float(cell_rmse.mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'accuracy_by_soh_band.csv', index=False)
    weighted = []
    for key, g in selected.groupby(['validation', 'dataset', 'cell_id']):
        weights = np.where(g.soh <= .80, 3., 1.)
        weighted.append(dict(zip(['validation', 'dataset', 'cell_id'], key),
            low_soh_weighted_mae_pp=float(np.average(np.abs(g.prediction - g.soh) * 100, weights=weights))))
    pd.DataFrame(weighted).to_csv(OUT / 'low_soh_weighted_error_per_cell.csv', index=False)
    selected['absolute_error_pp'] = (selected.prediction - selected.soh).abs() * 100
    selected.merge(cycles[['cycle_uid', 'ambient_temp_c', 'feature_reason', 'label_pairing', 'charge_source', 'label_source']], on='cycle_uid').sort_values(
        'absolute_error_pp', ascending=False).head(100).to_csv(OUT / 'worst_predictions.csv', index=False)
    # Matched rows prevent attributing cohort changes to temperature or CV features.
    base = selected[selected.validation.eq('leave_one_cell_out')].set_index('cycle_uid')
    matched = []
    for variant in ['temperature', 'cv_no_temperature']:
        subset = pred[pred.model.eq('training_selected') & pred.variant.eq(variant) & pred.validation.eq('leave_one_cell_out')]
        for dataset, g in subset.groupby('dataset'):
            ids = g.cycle_uid[g.cycle_uid.isin(base.index)]
            g = g.set_index('cycle_uid').loc[ids]
            b = base.loc[ids]
            for name, a in [('core_on_matched_rows', b), (variant, g)]:
                errors = pd.DataFrame({'cell': a.cell_id, 'error': (a.prediction - a.soh).abs() * 100})
                matched.append(dict(variant=variant, model=name, dataset=dataset, n_rows=len(a), n_cells=a.cell_id.nunique(),
                                    macro_cell_mae_pp=float(errors.groupby('cell').error.mean().mean())))
    pd.DataFrame(matched).to_csv(OUT / 'matched_feature_comparisons.csv', index=False)


def stage_review(pred, eols, config):
    selected = pred[pred.model.eq('training_selected') & pred.variant.eq('no_temperature')]
    rows = []
    # Truth locates retrospective evaluation anchors only. Forecast inputs are
    # still cut at each anchor; endpoint and reference capacity never enter fit.
    for _, truth in eols[eols.censoring.eq('observed_crossing')].iterrows():
        for validation in ['leave_one_cell_out', 'leave_one_dataset_out']:
            h = selected[selected.validation.eq(validation) & selected.dataset.eq(truth.dataset) & selected.cell_id.eq(truth.cell_id)]
            for stage, fraction in [('early', .20), ('mid', .50), ('late', .85)]:
                today = truth.first_reference_cycle + fraction * (truth.eol_cycle - truth.first_reference_cycle)
                for field, signal in [('prediction', 'raw'), ('prediction_smoothed', 'smoothed')]:
                    for method in ['linear', 'exponential']:
                        result = forecast(h, today, config['rul'], method, field)
                        actual = truth.eol_cycle - result['as_of_cycle']
                        forecast_value = result['predicted_rul_cycles']
                        error = forecast_value - actual
                        rows.append(dict(validation=validation, dataset=truth.dataset, cell_id=truth.cell_id, stage=stage,
                            signal=signal, method=method, actual_rul_cycles=actual, predicted_rul_cycles=forecast_value,
                            absolute_error_cycles=abs(error), absolute_percentage_error=100 * abs(error) / actual if actual > 0 else np.nan,
                            signed_error_cycles=error, forecast_reason=result['forecast_reason'], as_of_cycle=result['as_of_cycle'],
                            target_evaluation_cycle=today, eol_bracket_cycles=truth.eol_bracket_cycles))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'rul_by_stage_forecasts.csv', index=False)
    summaries = []
    for key, g in frame.groupby(['validation', 'dataset', 'signal', 'method', 'stage']):
        good = g[g.absolute_error_cycles.notna()]
        summaries.append(dict(zip(['validation', 'dataset', 'signal', 'method', 'stage'], key),
            requested_cells=len(g), scored_cells=len(good), forecast_coverage=float(len(good) / len(g)),
            mae_cycles=float(good.absolute_error_cycles.mean()), rmse_cycles=float(np.sqrt((good.signed_error_cycles ** 2).mean())),
            mean_ape_pct=float(good.absolute_percentage_error.mean()), median_ape_pct=float(good.absolute_percentage_error.median()),
            median_true_remaining_cycles=float(good.actual_rul_cycles.median())))
    pd.DataFrame(summaries).to_csv(OUT / 'rul_by_stage_metrics.csv', index=False)


def alarm_review(pred, eols, config):
    threshold_cycles = config['dashboard']['replace_soon_days'] * config['rul']['swaps_per_day'] * config['rul']['test_cycles_per_swap']
    selected = pred[pred.model.eq('training_selected') & pred.variant.eq('no_temperature')]
    points, events = [], []
    for validation in ['leave_one_cell_out', 'leave_one_dataset_out']:
        mode = selected[selected.validation.eq(validation)]
        for _, truth in eols.iterrows():
            h = mode[mode.dataset.eq(truth.dataset) & mode.cell_id.eq(truth.cell_id)].sort_values(
                ['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last')
            for field, signal in [('prediction', 'raw'), ('prediction_smoothed', 'smoothed')]:
                cell_points = []
                for j in range(len(h)):
                    today = float(h.cycle_index.iloc[j])
                    prefix = h.iloc[max(0, j - config['rul']['lookback_points'] + 1):j + 1]
                    f = forecast(prefix, today, config['rul'], 'linear', field)
                    alarm = bool(np.isfinite(f['predicted_rul_cycles']) and f['predicted_rul_cycles'] <= threshold_cycles)
                    label = 'unknown'
                    remaining = np.nan
                    if truth.censoring == 'observed_crossing':
                        remaining = truth.eol_cycle - today
                        if remaining > 0:
                            label = 'positive' if remaining <= threshold_cycles else 'negative'
                        else:
                            label = 'after_eol'
                    elif truth.censoring == 'right_censored' and truth.last_reference_cycle - today >= threshold_cycles:
                        label = 'negative'
                    r = dict(validation=validation, signal=signal, dataset=truth.dataset, cell_id=truth.cell_id,
                             today_cycle=today, alarm=alarm, truth_label=label, actual_remaining_cycles=remaining,
                             predicted_rul_cycles=f['predicted_rul_cycles'], forecast_reason=f['forecast_reason'],
                             censoring=truth.censoring)
                    points.append(r)
                    cell_points.append(r)
                if truth.censoring == 'observed_crossing':
                    alarms = [r for r in cell_points if r['alarm']]
                    within_window = [r for r in cell_points if r['truth_label'] == 'positive']
                    first = alarms[0] if alarms else None
                    lead = first['actual_remaining_cycles'] if first else np.nan
                    category = ('no_eligible_history' if not len(h) else 'no_alert')
                    if first:
                        category = 'premature_first_alert' if lead > threshold_cycles else ('timely_first_alert' if lead > 0 else 'late_first_alert')
                    timely = [r for r in within_window if r['alarm']]
                    events.append(dict(validation=validation, signal=signal, dataset=truth.dataset, cell_id=truth.cell_id,
                        first_alert_category=category, first_alert_lead_cycles=lead,
                        first_alert_lead_scenario_days=lead / (config['rul']['swaps_per_day'] * config['rul']['test_cycles_per_swap']),
                        has_observation_in_timely_window=bool(within_window), any_timely_alert=bool(timely),
                        timely_warning_lead_days=(max(r['actual_remaining_cycles'] for r in timely) /
                            (config['rul']['swaps_per_day'] * config['rul']['test_cycles_per_swap'])) if timely else np.nan))
            print(f'Alarm backtest {validation}: {truth.dataset}:{truth.cell_id}', flush=True)
    p, e = pd.DataFrame(points), pd.DataFrame(events)
    p.to_csv(OUT / 'alarm_observations.csv', index=False)
    e.to_csv(OUT / 'alarm_events.csv', index=False)
    summaries = []
    for key, g in p.groupby(['validation', 'signal', 'dataset']):
        known = g[g.truth_label.isin(['positive', 'negative'])]
        tp = int((known.alarm & known.truth_label.eq('positive')).sum())
        fp = int((known.alarm & known.truth_label.eq('negative')).sum())
        fn = int((~known.alarm & known.truth_label.eq('positive')).sum())
        tn = int((~known.alarm & known.truth_label.eq('negative')).sum())
        eg = e
        for col, value in zip(['validation', 'signal', 'dataset'], key):
            eg = eg[eg[col].eq(value)]
        counts = eg.first_alert_category.value_counts().to_dict() if len(eg) else {}
        summaries.append(dict(zip(['validation', 'signal', 'dataset'], key),
            tp_observations=tp, fp_observations=fp, fn_observations=fn, tn_observations=tn,
            observation_recall=tp / (tp + fn) if tp + fn else np.nan,
            observation_false_positive_rate=fp / (fp + tn) if fp + tn else np.nan,
            observation_precision=tp / (tp + fp) if tp + fp else np.nan,
            observed_threshold_events=len(eg), timely_first_alerts=counts.get('timely_first_alert', 0),
            premature_first_alerts=counts.get('premature_first_alert', 0), late_first_alerts=counts.get('late_first_alert', 0),
            no_alerts=counts.get('no_alert', 0), no_eligible_history=counts.get('no_eligible_history', 0),
            timely_first_alert_rate=counts.get('timely_first_alert', 0) / len(eg) if len(eg) else np.nan,
            any_timely_alert_events=int(eg.any_timely_alert.sum()) if len(eg) else 0,
            events_with_timely_observation=int(eg.has_observation_in_timely_window.sum()) if len(eg) else 0,
            median_first_alert_lead_days=float(eg.first_alert_lead_scenario_days.median()) if len(eg) else np.nan,
            median_timely_warning_days=float(eg.timely_warning_lead_days.median()) if len(eg) else np.nan))
    pd.DataFrame(summaries).to_csv(OUT / 'alarm_metrics.csv', index=False)
    write_json(OUT / 'alarm_definition.json', dict(horizon_cycles=threshold_cycles, horizon_scenario_days=config['dashboard']['replace_soon_days'],
        model='Predeclared linear trend on raw or trailing-median held-out SoH histories.',
        event='First alarm per cell. Before 14-day horizon = premature, within horizon = timely, at/after endpoint = late.',
        endpoint='First 70% nominal-relative laboratory discharge-capacity crossing, not a field failure.',
        right_censoring='Known negatives only when the last above-threshold reference is at least one horizon after today.',
        left_censoring='Already-below-threshold cells are not scored as incident events.',
        sampling='Oxford is sampled around every 100 ageing cycles, coarser than the 28-cycle alert window.'))


def feature_review(cycles, audit, config):
    rng = np.random.default_rng(config['seed'])
    rows = []
    for n, fold in enumerate(audit):
        if fold['validation'] != 'leave_one_cell_out':
            continue
        columns = fold['features']
        train = cycles[cycles.core_model_eligible & cycles.group_id.isin(fold['outer_train_cells']) & cycles[columns].notna().all(axis=1)]
        test = cycles[cycles.core_model_eligible & cycles.group_id.isin(fold['outer_test_cells']) & cycles[columns].notna().all(axis=1)]
        model = fit_estimator(fold['selected_model'], train, columns, config)
        baseline = np.abs(model.predict(test[columns]) - test.soh).mean() * 100
        for feature in columns:
            increases = []
            for repeat in range(3):
                x = test[columns].copy()
                x[feature] = rng.permutation(x[feature].to_numpy())
                increases.append(float(np.abs(model.predict(x) - test.soh).mean() * 100 - baseline))
            rows.append(dict(variant=fold['variant'], dataset=test.dataset.iloc[0], cell_id=test.cell_id.iloc[0],
                model=fold['selected_model'], feature=feature, n_rows=len(test), heldout_permutation_mae_increase_pp=float(np.mean(increases))))
        if n % 10 == 0:
            print(f'Feature review {n + 1}/{len(audit)} folds', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'feature_importance_per_cell.csv', index=False)
    frame.groupby(['variant', 'dataset', 'feature']).agg(mean_mae_increase_pp=('heldout_permutation_mae_increase_pp', 'mean'),
         n_cells=('cell_id', 'nunique')).reset_index().to_csv(OUT / 'feature_importance.csv', index=False)


if __name__ == '__main__':
    config = load_config()
    cycles = pd.read_parquet(ROOT / 'data/processed/cycles.parquet')
    pred = pd.read_parquet(ROOT / 'reports/evaluation/predictions.parquet')
    eols = pd.read_csv(ROOT / 'reports/rul/end_of_life.csv')
    accuracy_review(pred, cycles)
    stage_review(pred, eols, config)
    alarm_review(pred, eols, config)
    feature_review(cycles, json.loads((ROOT / 'reports/evaluation/fold_audit.json').read_text()), config)
    print('Review artifacts saved to', OUT, flush=True)
