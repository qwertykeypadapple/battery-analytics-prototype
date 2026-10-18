"""Causal, explainable forecasts and explicit right/left censoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import ROOT, load_config, write_json

METHODS = ['linear', 'exponential']


def reference_histories(references):
    good = references.loc[references.label_valid & np.isfinite(references.soh)].copy()
    # Some CALCE source cycle indices include multiple episodes. The final valid
    # discharge is the reference for that ageing index; preserve all in the cache.
    sort = ['dataset', 'cell_id', 'cycle_index']
    if 'label_end_time_s' in good:
        sort.append('label_end_time_s')
    return good.sort_values(sort).drop_duplicates(['dataset', 'cell_id', 'cycle_index'], keep='last')


def end_of_life(history, threshold=.70):
    h = history.sort_values('cycle_index').drop_duplicates('cycle_index', keep='last')
    if not len(h):
        return dict(censoring='no_valid_reference', eol_cycle=np.nan, last_reference_cycle=np.nan, eol_bracket_cycles=np.nan)
    x, y = h.cycle_index.to_numpy(), h.soh.to_numpy()
    common = dict(last_reference_cycle=float(x[-1]), first_reference_cycle=float(x[0]),
                  threshold_recrossings=int(np.sum((y[1:] > threshold) != (y[:-1] > threshold))))
    if y[0] <= threshold:
        return common | dict(censoring='left_censored_already_below_70pct', eol_cycle=np.nan, eol_bracket_cycles=np.nan)
    crossings = np.flatnonzero(y <= threshold)
    if not len(crossings):
        return common | dict(censoring='right_censored', eol_cycle=np.nan, eol_bracket_cycles=np.nan)
    j = int(crossings[0])
    crossing = x[j - 1] + (threshold - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])
    return common | dict(censoring='observed_crossing', eol_cycle=float(crossing), eol_bracket_cycles=float(x[j] - x[j - 1]))


def forecast(history, today, config, method='linear', value='prediction_smoothed'):
    """A forecast can read only model predictions at or before the requested cycle."""
    if method not in METHODS:
        raise ValueError(method)
    prefix = history.loc[history.cycle_index <= today].sort_values(['cycle_index', 'charge_end_time_s'])
    prefix = prefix.drop_duplicates('cycle_index', keep='last').tail(config['lookback_points'])
    result = dict(method=method, requested_today_cycle=float(today), as_of_cycle=np.nan, current_predicted_soh=np.nan,
                  predicted_rul_cycles=np.nan, predicted_rul_days=np.nan, forecast_reason='insufficient_history', history_points=len(prefix))
    if not len(prefix):
        return result
    x, y = prefix.cycle_index.to_numpy(float), prefix[value].to_numpy(float)
    if not np.isfinite(y).all():
        result['forecast_reason'] = 'nonfinite_soh_predictions'
        return result
    origin = float(x[-1])
    current = float(y[-1])
    result.update(as_of_cycle=origin, current_predicted_soh=current)
    threshold = config['end_of_life_soh']
    if current <= threshold:
        result.update(predicted_rul_cycles=0., predicted_rul_days=0., forecast_reason='predicted_at_or_below_threshold')
        return result
    if len(prefix) < config['min_points'] or np.ptp(x) <= 0:
        return result
    relative = x - origin
    if method == 'linear':
        slope, intercept = np.polyfit(relative, y, 1)
        result.update(fitted_slope=float(slope), fitted_current_soh=float(intercept))
        if slope >= -config['minimum_fade_per_cycle']:
            result['forecast_reason'] = 'flat_or_increasing_recent_trend'
            return result
        remaining = (threshold - intercept) / slope
    else:
        if (y <= 0).any():
            result['forecast_reason'] = 'nonpositive_soh_for_exponential'
            return result
        slope, intercept = np.polyfit(relative, np.log(y), 1)
        result.update(fitted_slope=float(slope), fitted_current_soh=float(np.exp(intercept)))
        if slope * current >= -config['minimum_fade_per_cycle']:
            result['forecast_reason'] = 'flat_or_increasing_recent_trend'
            return result
        remaining = (np.log(threshold) - intercept) / slope
    if not np.isfinite(remaining) or remaining > config['maximum_horizon_cycles']:
        result['forecast_reason'] = 'beyond_supported_forecast_horizon'
        return result
    remaining = max(0., float(remaining))
    cycles_per_day = config['swaps_per_day'] * config['test_cycles_per_swap']
    if cycles_per_day <= 0:
        raise ValueError('Cycle-to-day conversion must be positive')
    result.update(predicted_rul_cycles=remaining, predicted_rul_days=remaining / cycles_per_day, forecast_reason='estimated')
    return result


def evaluate(config=None):
    config = config or load_config()
    out = ROOT / 'reports/rul'
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(ROOT / 'reports/evaluation/predictions.parquet')
    pred = pred[pred.model.eq('training_selected') & pred.variant.eq('no_temperature')]
    references = reference_histories(pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet'))
    inventory = pd.read_csv(ROOT / 'reports/inventory/cells.csv')
    eols = []
    for _, cell in inventory.iterrows():
        h = references[references.dataset.eq(cell.dataset) & references.cell_id.eq(cell.cell_id)]
        eols.append(dict(dataset=cell.dataset, cell_id=cell.cell_id, **end_of_life(h, config['rul']['end_of_life_soh'])))
    eol = pd.DataFrame(eols)
    eol.to_csv(out / 'end_of_life.csv', index=False)
    results = []
    for (validation, dataset, cell), h in pred.groupby(['validation', 'dataset', 'cell_id']):
        h = h.sort_values(['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last')
        truth = eol[eol.dataset.eq(dataset) & eol.cell_id.eq(cell)].iloc[0].to_dict()
        candidates = sorted(set(int(round(frac * (len(h) - 1))) for frac in config['rul']['today_fractions']))
        for index in candidates:
            today = float(h.cycle_index.iloc[index])
            if truth['censoring'] == 'observed_crossing' and today >= truth['eol_cycle']:
                continue  # Avoid trivial evaluation after actual end of life.
            for value, signal in [('prediction', 'raw'), ('prediction_smoothed', 'smoothed')]:
                for method in METHODS:
                    result = forecast(h, today, config['rul'], method, value)
                    result.update(validation=validation, dataset=dataset, cell_id=cell, signal=signal,
                                  censoring=truth['censoring'], actual_eol_cycle=truth['eol_cycle'],
                                  actual_rul_cycles=np.nan, absolute_error_cycles=np.nan, signed_error_cycles=np.nan,
                                  censor_lower_bound_cycles=np.nan, premature_before_censor_bound=np.nan,
                                  eol_bracket_cycles=truth['eol_bracket_cycles'])
                    if truth['censoring'] == 'observed_crossing':
                        actual = truth['eol_cycle'] - result['as_of_cycle']
                        result['actual_rul_cycles'] = actual
                        if np.isfinite(result['predicted_rul_cycles']):
                            error = result['predicted_rul_cycles'] - actual
                            result.update(absolute_error_cycles=abs(error), signed_error_cycles=error)
                    elif truth['censoring'] == 'right_censored':
                        bound = max(0., truth['last_reference_cycle'] - result['as_of_cycle'])
                        result['censor_lower_bound_cycles'] = bound
                        if np.isfinite(result['predicted_rul_cycles']):
                            result['premature_before_censor_bound'] = float(result['predicted_rul_cycles'] < bound)
                    results.append(result)
    r = pd.DataFrame(results)
    r.to_csv(out / 'rul_forecasts.csv', index=False)
    summaries = []
    for level, columns in [('cell', ['dataset', 'cell_id']), ('dataset', ['dataset']), ('overall', [])]:
        groups = ['validation', 'signal', 'method'] + columns
        for key, g in r.groupby(groups):
            record = dict(zip(groups, key), level=level, n_anchors=len(g),
                          forecast_coverage=float(g.predicted_rul_cycles.notna().mean()),
                          observed_crossing_anchors=int(g.censoring.eq('observed_crossing').sum()),
                          n_cells=g[['dataset', 'cell_id']].drop_duplicates().shape[0])
            exact = g[g.absolute_error_cycles.notna()]
            right = g[g.censoring.eq('right_censored') & g.predicted_rul_cycles.notna()]
            record.update(n_scored=len(exact), mae_cycles=float(exact.absolute_error_cycles.mean()),
                          rmse_cycles=float(np.sqrt((exact.signed_error_cycles ** 2).mean())),
                          n_scored_cells=exact[['dataset', 'cell_id']].drop_duplicates().shape[0],
                          n_censored_forecasts=len(right), censor_bound_violation_rate=float(right.premature_before_censor_bound.mean()),
                          mean_bias_cycles=float(exact.signed_error_cycles.mean()))
            summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'rul_metrics.csv', index=False)
    # Compare the two methods on exactly the same available event anchors too.
    paired = r.pivot_table(index=['validation', 'signal', 'dataset', 'cell_id', 'requested_today_cycle'],
                           columns='method', values='absolute_error_cycles').dropna()
    paired.to_csv(out / 'paired_method_errors.csv')
    write_json(out / 'settings.json', dict(**config['rul'], primary_signal='smoothed',
               censoring='No exact RUL error for right-censored or already-below-threshold cells.',
               endpoint='First observed 70% reference crossing, interpolated between adjacent valid checks.',
               days='Scenario conversion only. Two swaps/day and one source laboratory cycle per swap by default.'))
    print(eol.groupby(['dataset', 'censoring']).size().to_string(), flush=True)
    print(summary[summary.level.eq('dataset') & summary.signal.eq('smoothed')].to_string(index=False), flush=True)
    return r, eol
