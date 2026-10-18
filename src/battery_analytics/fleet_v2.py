"""Frozen-rule resampled fleet experiment; no model fitting or report side effects."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .rul import forecast

KINDS = ('calce', 'nasa', 'oxford', 'mixed')
DATASETS = KINDS[:3]
SIZES = (50, 200, 1000)
SEEDS = tuple(range(2026101800, 2026101820))
TODAYS = (0, 15, 30, 45, 60)
HORIZONS = (30, 60, 90)
PROTOCOLS = ('leave_one_cell_out', 'leave_one_dataset_out')
AGE_SPANS = {'calce': 1008., 'nasa': 98.5, 'oxford': 4100.}
METHODS = ('charge', 'historical', 'cycle')
RUL_CONFIG = dict(end_of_life_soh=.7, lookback_points=12, min_points=6,
                  maximum_horizon_cycles=20000, minimum_fade_per_cycle=.000001,
                  swaps_per_day=2., test_cycles_per_swap=1.)


def make_layout(inventory, kind, seed):
    """One master layout; smaller cohorts use prefixes without drawing again."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, KINDS.index(kind)])))
    n = max(SIZES)
    datasets = (np.asarray(DATASETS)[rng.integers(0, 3, size=n)] if kind == 'mixed'
                else np.repeat(kind, n))
    cell_u, age_u = rng.random(n), rng.random(n)
    pools = {d: sorted(inventory.loc[inventory.dataset.eq(d), 'cell_id']) for d in DATASETS}
    cells = [pools[d][int(u * len(pools[d]))] for d, u in zip(datasets, cell_u)]
    return pd.DataFrame(dict(kind=kind, seed=seed, copy_index=np.arange(n), dataset=datasets,
                             cell_id=cells, group_id=[f'{d}:{c}' for d, c in zip(datasets, cells)],
                             commissioning_day=[-u * AGE_SPANS[d] for d, u in zip(datasets, age_u)]))


def audit_predictions(predictions, baseline_audit, cycle_audit, inventory):
    """Fail closed on a fitted-on-source prediction or dataset leakage."""
    bfolds = {(f['validation'], f['fold']): f for f in baseline_audit if f['variant'] == 'no_temperature'}
    cfolds = {(f['validation'], f['fold']): f for f in cycle_audit}
    records = []
    for method, p in predictions.items():
        assert not p.duplicated(['validation', 'cycle_uid']).any()
        assert (p.group_id == p.dataset + ':' + p.cell_id).all()
        for protocol in PROTOCOLS:
            for cell in inventory.itertuples(index=False):
                gid = f'{cell.dataset}:{cell.cell_id}'
                h = p[p.validation.eq(protocol) & p.group_id.eq(gid)]
                fold = gid if protocol == PROTOCOLS[0] else cell.dataset
                record = dict(method=method, validation=protocol, dataset=cell.dataset,
                              cell_id=cell.cell_id, group_id=gid, fold=fold if len(h) else '',
                              prediction_rows=len(h), source_held_out=False,
                              source_dataset_held_out=False, train_cells='[]', selection_cells='[]')
                if len(h):
                    bf = bfolds[protocol, fold]
                    f = bf if method == 'charge' else cfolds[protocol, fold]
                    train = f['outer_train_cells'] if method == 'charge' else f['train_cells']
                    test = f['outer_test_cells'] if method == 'charge' else f['test_cells']
                    selection = bf['inner_train_cells'] + bf['inner_validation_cells'] if method == 'charge' else []
                    assert gid in test and gid not in train and gid not in selection
                    if method == 'cycle':
                        assert set(train) == set(bf['outer_train_cells'])
                        assert set(test) == set(bf['outer_test_cells'])
                        assert h.fold.eq(fold).all()
                    ds_absent = all(g.split(':', 1)[0] != cell.dataset for g in train + selection)
                    if protocol == PROTOCOLS[1]:
                        assert ds_absent
                    record.update(source_held_out=True, source_dataset_held_out=ds_absent,
                                  train_cells=json.dumps(train), selection_cells=json.dumps(selection))
                records.append(record)
    return pd.DataFrame(records)


def prefix_crossings(history):
    """Cache the frozen RUL function at each prefix end, using predictions only.

    A later lookup shifts the cached crossing to a copy's continuous source age.
    This is identical to v1 asof_forecast, without millions of repeated fits.
    """
    h = history[['cycle_index', 'charge_end_time_s', 'prediction']].copy()
    h = h.sort_values(['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last')
    x = h.cycle_index.to_numpy(float)
    crossings, reasons = [], []
    for i, age in enumerate(x):
        prefix = h.iloc[max(0, i - RUL_CONFIG['lookback_points'] + 1):i + 1]
        f = forecast(prefix, age, RUL_CONFIG, 'linear', 'prediction')
        remaining = f['predicted_rul_cycles']
        crossings.append(age + remaining)
        reasons.append(f['forecast_reason'])
    return x, np.asarray(crossings), np.asarray(reasons, dtype=object)


def lookup_forecasts(cache, ages):
    ages = np.asarray(ages, dtype=float)
    remaining = np.full(len(ages), np.nan)
    reason = np.full(len(ages), 'no_prefix', dtype=object)
    x, crossings, reasons = cache
    if len(x):
        idx = np.searchsorted(x, ages, side='right') - 1
        ended = ages > x[-1]
        reason[ended] = 'history_ended'
        valid = (idx >= 0) & ~ended
        remaining[valid] = crossings[idx[valid]] - ages[valid]
        reason[valid] = reasons[idx[valid]]
        finite = np.isfinite(remaining)
        reason[finite] = np.where(remaining[finite] <= 0, 'predicted_overdue', 'estimated')
    return remaining, reason


def reference_asof(history, endpoint, ages):
    """Current eligibility/exposure uses only checks already observed by age.

    Only the explicitly named actual_* / retrospective columns use future truth.
    """
    ages = np.asarray(ages, dtype=float)
    n = len(ages)
    x, y = history.cycle_index.to_numpy(float), history.soh.to_numpy(float)
    already = np.zeros(n, dtype=bool)
    exposure = np.zeros(n)
    observed_events = np.zeros(n, dtype=int)
    no_prefix = np.ones(n, dtype=bool)
    if len(x):
        idx = np.searchsorted(x, ages, side='right') - 1
        valid = idx >= 0
        no_prefix = ~valid
        below = np.flatnonzero(y <= .7)
        below_age = x[below[0]] if len(below) else np.inf
        already = ages >= below_age
        if y[0] > .7:
            exposure[valid] = np.maximum(0, np.minimum(x[idx[valid]], below_age) - x[0]) / 2
        if endpoint.censoring == 'observed_crossing':
            observed_events = already.astype(int)
    observed = endpoint.censoring == 'observed_crossing'
    out = dict(already_below=already, no_reference_prefix=no_prefix, historical_events=observed_events,
               historical_exposure_days=exposure,
               initially_left_censored=np.repeat(endpoint.censoring == 'left_censored_already_below_70pct', n),
               retrospective_preexisting=(ages >= endpoint.eol_cycle) & ~already if observed else np.zeros(n, bool))
    for days in HORIZONS:
        actual = np.full(n, np.nan)
        if observed:
            actual = ((ages < endpoint.eol_cycle) & (endpoint.eol_cycle <= ages + days * 2)).astype(float)
        elif endpoint.censoring == 'right_censored':
            actual[ages + days * 2 <= endpoint.last_reference_cycle] = 0.
        out[f'actual_{days}'] = actual
    return pd.DataFrame(out)


def absolute_bounds(low, high):
    return (0. if low <= 0 <= high else min(abs(low), abs(high))), max(abs(low), abs(high))


def scenario(g, days, nominal):
    """All methods compare the same copies; retain missing-count intervals."""
    a = g.loc[~g.already_below].copy()
    actual = a[f'actual_{days}']
    lo = float(actual.sum())
    unknown = int(actual.isna().sum())
    exposure, events = g.historical_exposure_days.sum(), g.historical_events.sum()
    rate = events / exposure if exposure > 0 else 0.
    hp = min(1., rate * days)
    r = dict(nominal=nominal, unique_sources=g.group_id.nunique(),
             unique_sources_calce=g.loc[g.dataset.eq('calce'), 'group_id'].nunique(),
             unique_sources_nasa=g.loc[g.dataset.eq('nasa'), 'group_id'].nunique(),
             unique_sources_oxford=g.loc[g.dataset.eq('oxford'), 'group_id'].nunique(),
             already_below=int(g.already_below.sum()), at_risk=len(a), at_risk_unique=a.group_id.nunique(),
             initially_left_censored=int(g.initially_left_censored.sum()),
             no_reference_prefix=int(a.no_reference_prefix.sum()),
             retrospective_preexisting=int(a.retrospective_preexisting.sum()),
             actual_count_lower=lo, actual_count_upper=lo + unknown, unknown_actual=unknown,
             historical_events=int(events), historical_exposure_days=float(exposure),
             historical_rate_per_battery_day=float(rate), historical_zero_exposure=exposure == 0,
             historical_probability=float(hp))
    predictions = {'historical': np.repeat(hp, len(a))}
    for method in ('charge', 'cycle'):
        remaining = a[f'{method}_rul'].to_numpy()
        usable = np.isfinite(remaining)
        predictions[method] = np.where(usable, ((remaining > 0) & (remaining <= days * 2)).astype(float), np.nan)
        r[f'{method}_usable'] = int(usable.sum())
        r[f'{method}_usable_unique'] = a.loc[usable, 'group_id'].nunique()
        r[f'{method}_overdue'] = int(a[f'{method}_reason'].eq('predicted_overdue').sum())
    matched = actual.notna().to_numpy() & np.isfinite(predictions['charge']) & np.isfinite(predictions['cycle'])
    r.update(matched_n=int(matched.sum()), matched_unique=a.loc[matched, 'group_id'].nunique(),
             matched_actual=float(actual.iloc[np.flatnonzero(matched)].sum()) if matched.any() else np.nan)
    for method, pred in predictions.items():
        subtotal, missing = float(np.nansum(pred)), int(np.isnan(pred).sum())
        r.update({f'{method}_subtotal': subtotal, f'{method}_unknown': missing,
                  f'{method}_count_upper': subtotal + missing,
                  f'{method}_subtotal_error_low': subtotal - (lo + unknown),
                  f'{method}_subtotal_error_high': subtotal - lo})
        low, high = subtotal - (lo + unknown), subtotal + missing - lo
        alo, ahi = absolute_bounds(low, high)
        r.update({f'{method}_full_error_low': low, f'{method}_full_error_high': high,
                  f'{method}_full_abs_error_low': alo, f'{method}_full_abs_error_high': ahi})
        known = actual.notna().to_numpy() & np.isfinite(pred)
        r[f'{method}_known_n'] = int(known.sum())
        r[f'{method}_known_unique'] = a.loc[known, 'group_id'].nunique()
        r[f'{method}_known_actual'] = float(actual.iloc[np.flatnonzero(known)].sum()) if known.any() else np.nan
        r[f'{method}_known_prediction'] = float(pred[known].sum()) if known.any() else np.nan
        r[f'{method}_known_error'] = r[f'{method}_known_prediction'] - r[f'{method}_known_actual']
        prediction = float(pred[matched].sum()) if matched.any() else np.nan
        error = prediction - r['matched_actual']
        r.update({f'matched_{method}': prediction, f'{method}_matched_error': error,
                  f'{method}_matched_absolute_error': abs(error), f'{method}_matched_squared_error': error ** 2,
                  f'{method}_matched_error_per100_installed': error * 100 / nominal,
                  f'{method}_matched_error_per100_matched': error * 100 / matched.sum() if matched.any() else np.nan})
    return r


def summarize(scenarios, specific=False):
    rows = []
    keys = ['validation', 'kind', 'nominal', 'days']
    for key, g in scenarios.groupby(keys, sort=True):
        for method in METHODS:
            suffix = 'known_error' if specific else 'matched_error'
            error = g[f'{method}_{suffix}'].dropna()
            r = dict(zip(keys, key), method=method, scored_scenarios=len(error),
                     mean_unique_sources=g.unique_sources.mean(), min_unique_sources=g.unique_sources.min(),
                     max_unique_sources=g.unique_sources.max(), mean_at_risk=g.at_risk.mean(),
                     mean_matched_n=g.matched_n.mean(), mean_matched_unique=g.matched_unique.mean(),
                     mean_matched_actual=g.matched_actual.mean(), mae_count=error.abs().mean(),
                     rmse_count=np.sqrt((error ** 2).mean()), signed_bias_count=error.mean(),
                     mean_known_n=g[f'{method}_known_n'].mean(),
                     mean_known_unique=g[f'{method}_known_unique'].mean())
            if not specific:
                for norm in ('installed', 'matched'):
                    e = g[f'{method}_matched_error_per100_{norm}'].dropna()
                    r[f'mae_per100_{norm}'] = e.abs().mean()
                    r[f'bias_per100_{norm}'] = e.mean()
            rows.append(r)
    return pd.DataFrame(rows)
