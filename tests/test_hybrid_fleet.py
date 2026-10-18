"""Scientific checks for ranking causality, selection budgets and unknown truth."""
from itertools import product
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from battery_analytics.common import ROOT
from battery_analytics.hybrid_fleet import (
    current_soh, tie_keys, ranking_order, historical_budget, ratios,
    evaluate_selection, evaluate_random,
)


def test_point_score_is_past_only_label_free_and_rejects_ended_history():
    h = pd.DataFrame(dict(cycle_index=[0., 10., 10., 20.],
                          charge_end_time_s=[0., 10., 11., 20.],
                          prediction=[1., .9, .85, .8], soh=[99.] * 4))
    values, reasons = current_soh(h, [-1., 10., 15., 20., 21.])
    np.testing.assert_allclose(values, [np.nan, .85, .85, .8, np.nan], equal_nan=True)
    assert reasons.tolist() == ['no_prefix', 'available', 'available', 'available', 'history_ended']
    h.loc[h.cycle_index > 15, 'prediction'] = -999
    h['soh'] = -999
    after, _ = current_soh(h, [15.])
    assert after[0] == values[2]


def test_fallback_ranks_every_member_and_overdue_rul_gets_high_priority():
    p = pd.DataFrame(dict(source_age_cycles=[10., 40., 30., 20., 50.],
                          charge_soh=[.8, np.nan, .7, np.nan, .8],
                          charge_rul=[10., np.nan, -2., np.nan, 3.]))
    # Deliberately fix tie order: finite-score ties do not use age.
    ties = np.array(['a', 'b', 'c', 'd', 'e'])
    order, missing = ranking_order(p, 'charge_soh', ties)
    assert order.tolist() == [2, 0, 4, 1, 3]
    assert missing.tolist() == [False, True, False, True, False]
    order, _ = ranking_order(p, 'charge_rul', ties)
    assert order.tolist() == [2, 4, 0, 1, 3]
    order, _ = ranking_order(p, 'age', ties)
    assert order.tolist() == [4, 1, 2, 3, 0]
    assert tie_keys('unique', 'calce', 123, ['calce:a'])[0] == hashlib.sha256(
        b'hybrid-fleet-v1|unique|calce|123|calce:a').hexdigest()


def test_historical_budget_rounds_half_up_and_includes_retired_past_exposure():
    g = pd.DataFrame(dict(historical_events=[1, 0, 0, 0],
                          historical_exposure_days=[60., 60., 30., 30.],
                          already_below=[True, False, False, False]))
    # Three at risk * (1 event / 180 battery-days) * 30 = 0.5, rounded to 1.
    budget = historical_budget(g, 30)
    assert budget['historical_count'] == .5 and budget['historical_k'] == 1
    g.historical_exposure_days = 0
    assert historical_budget(g, 30)['historical_k'] == 0
    assert historical_budget(g, 30)['historical_zero_exposure']


@pytest.mark.parametrize('y,selected', [
    ([1., 0., np.nan, np.nan], [0, 2]),
    ([0., np.nan, np.nan], [1]),
    ([np.nan, np.nan], [0, 1]),
    ([np.nan, np.nan], []),
    ([1., np.nan], []),
    ([0., 0.], [0]),
])
def test_censoring_bounds_match_every_possible_unknown_assignment(y, selected):
    y = np.asarray(y)
    selected = np.asarray(selected, int)
    uidx = np.flatnonzero(np.isnan(y))
    known_a, unknown = int(np.sum(y == 1)), len(uidx)
    t, uf = np.sum(y[selected] == 1), np.isnan(y[selected]).sum()
    scores = ratios(t, uf, len(selected), known_a, unknown)
    p_values, r_values = [], []
    for values in product([0., 1.], repeat=unknown):
        actual = y.copy()
        actual[uidx] = values
        tp = actual[selected].sum()
        if len(selected):
            p_values.append(tp / len(selected))
        if actual.sum():
            r_values.append(tp / actual.sum())
    for metric, values in [('precision', p_values), ('recall', r_values)]:
        if values:
            assert scores[metric + '_lower'][0] == pytest.approx(min(values))
            assert scores[metric + '_upper'][0] == pytest.approx(max(values))
        else:
            assert np.isnan(scores[metric + '_lower'][0]) and np.isnan(scores[metric + '_upper'][0])


def test_unknown_flags_and_missing_predictions_are_not_dropped_from_k():
    y = [1., 0., np.nan, 1.]
    ids = ['a', 'b', 'c', 'a']
    r = evaluate_selection(y, ids, [0, 2], np.array([False, False, True, False]), {'a': 0, 'b': 1, 'c': 2})
    assert r['selected_known_positive'] == r['selected_unknown'] == 1
    assert r['precision'] == 1 and r['recall'] == .5
    assert (r['precision_lower'], r['precision_upper']) == (.5, 1.)
    assert r['fallback_selected'] == r['fallback_selected_unique'] == 1
    assert r['caught_unique'] == 1 and r['caught_source_mask'] == '0x1'
    assert np.isnan(r['exact_precision'])


def test_random_floor_is_reproducible_and_matches_uniform_selection_expectation():
    y = [1.] * 4 + [0.] * 3 + [np.nan] * 3
    args = (y, list('abcdefghij'), 3, 'unique', 'mixed', 2026101800, 0, 60, 'historical')
    r, again = evaluate_random(*args), evaluate_random(*args)
    assert r['selected_known_positive'] == again['selected_known_positive']
    assert r['random_draws'] == 2000
    assert r['random_expected_tp'] == 1.2
    assert abs(r['selected_known_positive'] - 1.2) < .08
    assert r['selected_unique'] == pytest.approx(3)
    assert r['caught_unique'] == pytest.approx(1.2)


def test_saved_hybrid_populations_budgets_and_earlier_artifacts_reconcile():
    out = ROOT / 'reports/hybrid_fleet'
    if not (out / 'run_manifest.json').exists():
        pytest.skip('Requires the preregistered hybrid run')
    r = pd.read_csv(out / 'scenario_metrics.csv')
    assert len(r) == 73200
    assert not r.duplicated(['case_id', 'k_mode', 'method']).any()
    assert (r.nominal == r.at_risk + r.already_below).all()
    assert r.groupby(['case_id', 'k_mode']).k.nunique().le(1).all()
    available = r[r.k_available]
    np.testing.assert_allclose(available.selected_known_positive + available.selected_unknown + available.selected_known_negative, available.k)
    assert r.loc[r.k_mode.eq('oracle') & r.unknown_actual.gt(0), 'k'].isna().all()
    for field in ['precision', 'recall', 'precision_lower', 'precision_upper', 'recall_lower', 'recall_upper']:
        assert r[field].dropna().between(0, 1).all()
    assert (r.precision_lower.dropna() <= r.loc[r.precision_lower.notna(), 'precision_upper']).all()
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['protected_sha256_before'] == manifest['protected_sha256_after']
    for path, sha in manifest['protected_sha256_before'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
