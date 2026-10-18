"""Preregistered ranking rules and censoring-aware top-K evaluation; no fitting."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import numpy as np
import pandas as pd

from .fleet_v2 import KINDS

METHODS = ('charge_soh', 'charge_rul', 'cycle_soh', 'age', 'random')
SCORE_FIELDS = dict(charge_soh='charge_soh', charge_rul='charge_rul', cycle_soh='cycle_soh')
RANDOM_DRAWS = 2000
K_MODES = ('historical', 'oracle')


def current_soh(history, ages):
    """Latest eligible raw held-out point, without any reference-label columns."""
    h = history[['cycle_index', 'charge_end_time_s', 'prediction']].copy()
    h = h.sort_values(['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last')
    ages = np.asarray(ages, float)
    values = np.full(len(ages), np.nan)
    reasons = np.full(len(ages), 'no_prefix', dtype=object)
    if len(h):
        x, y = h.cycle_index.to_numpy(float), h.prediction.to_numpy(float)
        idx = np.searchsorted(x, ages, side='right') - 1
        ended = ages > x[-1]
        reasons[ended] = 'history_ended'
        valid = (idx >= 0) & ~ended
        values[valid] = y[idx[valid]]
        reasons[valid] = np.where(np.isfinite(values[valid]), 'available', 'nonfinite_last_prediction')
        values[~np.isfinite(values)] = np.nan
    return values, reasons


def tie_keys(representation, kind, seed, unit_ids):
    return np.asarray([hashlib.sha256(
        f'hybrid-fleet-v1|{representation}|{kind}|{seed}|{uid}'.encode()).hexdigest()
        for uid in unit_ids])


def ranking_order(predictors, method, ties):
    """Accept only causal score/age columns, never outcomes or follow-up flags."""
    age = predictors.source_age_cycles.to_numpy(float)
    if method == 'age':
        return np.lexsort((ties, -age)), np.zeros(len(age), dtype=bool)
    scores = predictors[SCORE_FIELDS[method]].to_numpy(float)
    missing = ~np.isfinite(scores)
    # Finite score first, score ascending; missing scores then age descending.
    order = np.lexsort((ties, np.where(missing, -age, 0), np.where(missing, 0, scores), missing.astype(int)))
    return order, missing


def historical_budget(installed, days):
    exposure = float(installed.historical_exposure_days.sum())
    events = int(installed.historical_events.sum())
    m = int((~installed.already_below).sum())
    rate = events / exposure if exposure > 0 else 0.
    count = m * min(1., rate * days)
    return dict(historical_events=events, historical_exposure_days=exposure,
                historical_zero_exposure=exposure == 0, historical_rate=rate,
                historical_count=count, historical_k=min(m, max(0, int(np.floor(count + .5)))))


def ratios(t, uf, k, a, u):
    """Vectorized exact/conditional metrics and sharp bounds under unknown truth."""
    t, uf = np.atleast_1d(t).astype(float), np.atleast_1d(uf).astype(float)
    n = len(t)
    def divide(numer, denom):
        numer, denom = np.broadcast_arrays(numer, denom)
        return np.divide(numer, denom, out=np.full(numer.shape, np.nan), where=denom > 0)
    precision = divide(t, k - uf)
    recall = divide(t, a)
    plo, phi = divide(t, k), divide(t + uf, k)
    if a + u == 0:
        rlo = rhi = np.full(n, np.nan)
    else:
        rlo = np.divide(t, a + u - uf, out=np.ones(n), where=a + u - uf > 0)
        rhi = np.divide(t + uf, a + uf, out=np.zeros(n), where=a + uf > 0)
    return dict(precision=precision, recall=recall,
                exact_precision=precision if u == 0 else np.full(n, np.nan),
                exact_recall=recall if u == 0 else np.full(n, np.nan),
                precision_lower=plo, precision_upper=phi, recall_lower=rlo, recall_upper=rhi)


def metric_means(t, uf, k, a, u):
    result = {}
    for name, values in ratios(t, uf, k, a, u).items():
        finite = np.isfinite(values)
        result[name] = float(values[finite].mean()) if finite.any() else np.nan
        result[name + '_draws_defined'] = int(finite.sum())
    return result


def source_mask(ids, source_bits):
    value = 0
    for gid in set(ids):
        value |= 1 << source_bits[gid]
    return hex(value)


def evaluate_selection(actual, ids, selected, missing, source_bits):
    y = np.asarray(actual, float)
    ids, selected = np.asarray(ids), np.asarray(selected, int)
    a, u, k = int(np.sum(y == 1)), int(np.isnan(y).sum()), len(selected)
    chosen = y[selected]
    t, uf = int(np.sum(chosen == 1)), int(np.isnan(chosen).sum())
    caught_ids = ids[selected][chosen == 1]
    fallback_ids = ids[selected][missing[selected]]
    r = dict(selected_known_positive=t, selected_unknown=uf, selected_known_negative=k - t - uf,
             selected_unique=len(set(ids[selected])), caught_unique=len(set(caught_ids)),
             caught_source_mask=source_mask(caught_ids, source_bits),
             fallback_selected=int(missing[selected].sum()), fallback_selected_unique=len(set(fallback_ids)),
             missing_scores=int(missing.sum()), missing_unique=len(set(ids[missing])),
             random_draws=0, random_expected_tp=np.nan)
    r.update(metric_means(t, uf, k, a, u))
    return r


@lru_cache(maxsize=100000)
def seen_probability(m, k, count):
    if not count or not k:
        return 0.
    if k > m - count:
        return 1.
    return 1 - math.comb(m - count, k) / math.comb(m, k)


def evaluate_random(actual, ids, k, representation, kind, seed, today, days, k_mode):
    """Evaluate a uniform top-K draw via its exact category-count distribution."""
    y, ids = np.asarray(actual, float), np.asarray(ids)
    m, a, u = len(y), int(np.sum(y == 1)), int(np.isnan(y).sum())
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        20261018, 77, 0 if representation == 'unique' else 1, KINDS.index(kind),
        seed, today, days, K_MODES.index(k_mode)])))
    draws = (rng.multivariate_hypergeometric([a, m - a - u, u], k, size=RANDOM_DRAWS)
             if m else np.zeros((RANDOM_DRAWS, 3), dtype=int))
    t, uf = draws[:, 0], draws[:, 2]
    all_counts = pd.Series(ids).value_counts().to_numpy()
    event_counts = pd.Series(ids[y == 1]).value_counts().to_numpy()
    r = dict(selected_known_positive=float(t.mean()), selected_unknown=float(uf.mean()),
             selected_known_negative=float(draws[:, 1].mean()),
             selected_unique=sum(seen_probability(m, k, int(c)) for c in all_counts),
             caught_unique=sum(seen_probability(m, k, int(c)) for c in event_counts), caught_source_mask='',
             fallback_selected=0, fallback_selected_unique=0, missing_scores=0, missing_unique=0,
             random_draws=RANDOM_DRAWS, random_expected_tp=k * a / m if m else 0.)
    r.update(metric_means(t, uf, k, a, u))
    return r


def summarize(results):
    keys = ['representation', 'validation', 'kind', 'nominal', 'days', 'k_mode', 'method']
    mean_fields = ['k', 'at_risk', 'at_risk_unique', 'known_positives', 'unknown_actual',
                   'missing_scores', 'missing_unique', 'fallback_selected', 'fallback_selected_unique',
                   'selected_unique', 'caught_unique', 'precision', 'recall', 'exact_precision', 'exact_recall',
                   'precision_lower', 'precision_upper', 'recall_lower', 'recall_upper',
                   'selected_known_positive', 'selected_unknown']
    rows = []
    for key, g in results.groupby(keys, sort=True):
        r = dict(zip(keys, key), cases=len(g), k_available_cases=int(g.k_available.sum()),
                 precision_cases=int(g.precision.notna().sum()), recall_cases=int(g.recall.notna().sum()),
                 exact_precision_cases=int(g.exact_precision.notna().sum()), exact_recall_cases=int(g.exact_recall.notna().sum()),
                 zero_k_cases=int(g.k.eq(0).sum()), zero_known_event_cases=int(g.known_positives.eq(0).sum()),
                 fully_observed_cases=int(g.unknown_actual.eq(0).sum()),
                 min_real_sources=int(g.unique_sources.min()), max_real_sources=int(g.unique_sources.max()))
        r.update({f: float(g[f].mean()) for f in mean_fields})
        positive_mask = 0
        for mask in g.actual_positive_source_mask.unique():
            positive_mask |= int(mask, 16)
        r['distinct_positive_source_histories'] = positive_mask.bit_count()
        r['actual_positive_source_mask'] = hex(positive_mask)
        rows.append(r)
    return pd.DataFrame(rows)


def paired_comparisons(results):
    keys = ['representation', 'validation', 'kind', 'nominal', 'days', 'k_mode']
    rows = []
    for key, g in results.groupby(keys, sort=True):
        for model in ('charge_soh', 'charge_rul'):
            for other in ('cycle_soh', 'age', 'random'):
                for metric in ('precision', 'recall'):
                    pivot = g.pivot(index='case_id', columns='method', values=metric)
                    paired = pivot[[model, other]].dropna()
                    difference = paired[model] - paired[other]
                    support = g[g.case_id.isin(paired.index)].actual_positive_source_mask.unique()
                    mask = 0
                    for value in support:
                        mask |= int(value, 16)
                    rows.append(dict(zip(keys, key), model=model, baseline=other, metric=metric,
                                     paired_cases=len(paired), mean_difference=difference.mean(),
                                     model_mean=paired[model].mean(), baseline_mean=paired[other].mean(),
                                     distinct_positive_sources=mask.bit_count(), positive_source_mask=hex(mask)))
    return pd.DataFrame(rows)
