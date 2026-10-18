"""Leakage, censoring, risk-set and first-alert checks for the survival study."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from battery_analytics.common import ROOT
from battery_analytics.survival import History, KaplanMeier, person_periods, fit_bundle, ipcw, event_class, Cox
from battery_analytics.survival_training import choose_threshold


def endpoints():
    return pd.DataFrame([
        dict(group_id='a', censoring='observed_crossing', first_reference_cycle=0., eol_cycle=15.5, last_reference_cycle=40.),
        dict(group_id='b', censoring='right_censored', first_reference_cycle=0., eol_cycle=np.nan, last_reference_cycle=13.),
    ])


def test_survival_features_read_only_past_predictions_and_preserve_signed_rate():
    x = np.arange(41.)
    f = pd.DataFrame(dict(cycle_index=x, charge_end_time_s=x, prediction=1 - .003*x, soh=999.))
    h = History(f)
    before, last = h.at([4., 20.5, 80.])
    assert np.isnan(before[0, 1])
    np.testing.assert_allclose(before[1], [.94, .003, 20.5])
    assert last.tolist() == [4., 20., 40.]
    f.loc[f.cycle_index > 20.5, 'prediction'] = 1000
    f['soh'] = -1000
    after, _ = History(f).at([20.5])
    np.testing.assert_allclose(after[0], before[1])
    f.prediction = 1 + .001*x
    assert History(f).at([20.])[0][0, 1] == pytest.approx(-.001)


def test_person_periods_do_not_turn_partial_censoring_into_a_non_event_label():
    x = np.arange(41.)
    f = pd.DataFrame(dict(cycle_index=x, charge_end_time_s=x, prediction=1 - .003*x))
    p = person_periods({'a': History(f), 'b': History(f)}, endpoints())
    a, b = p[p.group_id.eq('a')], p[p.group_id.eq('b')]
    assert a.start.tolist() == [5., 15.]
    assert a.stop.tolist() == [15., 15.5]
    assert a.event.tolist() == [0, 1]
    assert a.logistic_eligible.all()
    assert b.stop.tolist() == [13.] and not b.logistic_eligible.any()
    assert b.event.sum() == 0


def test_delayed_entry_km_and_training_only_ipcw_have_known_hand_values():
    e = endpoints(); e.loc[0, 'eol_cycle'] = 10.; e.loc[1, 'last_reference_cycle'] = 20.
    km, g = KaplanMeier(e), KaplanMeier(e, censoring=True)
    assert km.risk([0.], 15.)[0] == .5
    assert np.isnan(km.risk([0.], 25.)[0])
    y, w, support = ipcw(g, [0.], 10., True, 15.)
    assert y[0] == w[0] == 1 and support[0]
    y, w, support = ipcw(g, [0.], 12., False, 15.)
    assert np.isnan(y[0]) and w[0] == 0 and support[0]
    _, _, support = ipcw(g, [0.], 30., False, 25.)
    assert not support[0]
    # Before a delayed entrant joins, the first event removes the sole at-risk cell.
    e.loc[1, 'first_reference_cycle'] = 12.
    assert KaplanMeier(e).at([10.])[0] == 0


def test_no_event_training_returns_explicit_zero_risk_with_feature_coverage():
    x = np.arange(41.)
    f = pd.DataFrame(dict(cycle_index=x, charge_end_time_s=x, prediction=1 - .003*x))
    e = endpoints(); e.censoring = 'right_censored'; e.last_reference_cycle = 40.
    bundle = fit_bundle({'a': History(f), 'b': History(f)}, e)
    features, _ = History(f).at([20.])
    for name in ('hazard', 'hazard_known_history', 'cox'):
        assert bundle.models[name].status == 'no_training_events'
        assert bundle.models[name].risk([20.], 10., features)[0] == 0


def test_time_varying_cox_matches_independent_phreg_penalized_fit():
    sm = pytest.importorskip('statsmodels.duration.hazard_regression')
    rows = []
    for gid, end, event, soh in [('a', 5.5, 1, .7), ('b', 8.3, 1, .8), ('c', 12., 0, .9), ('d', 13., 0, 1.)]:
        for start in np.arange(0., end, 3.):
            stop = min(start+3, end)
            rows.append(dict(group_id=gid, start=start, stop=stop, event=int(event and stop==end),
                             charge_soh=soh-.005*start, fade_rate=.001+soh*.001, cycle=start,
                             logistic_eligible=True))
    p = pd.DataFrame(rows)
    fitted = Cox(p)
    assert fitted.status == 'fitted'
    x = fitted.scaler.transform(p[['charge_soh', 'fade_rate']].to_numpy())
    reference = sm.PHReg(p.stop.to_numpy(), x, status=p.event.to_numpy(), entry=p.start.to_numpy(), ties='breslow')
    independent = reference.fit_regularized(alpha=p.event.sum()/len(p), L1_wt=0., maxiter=1000, cnvrg_tol=1e-10)
    np.testing.assert_allclose(fitted.beta, independent.params, atol=2e-5)
    prob = fitted.risk([3.], 3., np.array([[.8, .002, 3.]]))[0]
    assert 0 <= prob <= 1


def test_first_alarm_categories_and_inner_threshold_tie_break_are_fixed():
    assert [event_class(first, 100., 28.) for first in [20., 90., 110., np.nan]] == ['premature', 'timely', 'late', 'none']
    frame = pd.DataFrame(dict(group_id=['a'], cycle_index=[1.], p_28=[0.]))
    chosen, _, _ = choose_threshold(frame, endpoints().iloc[:1], 28)
    assert chosen == 1.01  # all candidate thresholds tie on a missed event


def test_saved_survival_artifacts_preserve_holdouts_and_earlier_experiments():
    out = ROOT / 'reports/survival'
    if not (out / 'run_manifest.json').exists():
        pytest.skip('Requires the preregistered survival run')
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['protected_sha256_before'] == manifest['protected_sha256_after']
    for path, sha in manifest['protected_sha256_before'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
    audit = json.loads((out / 'nested_charge_audit.json').read_text())
    for a in audit:
        assert set(a['train_cells']).isdisjoint(a['test_cells'])
        assert set(a['train_cells'] + a['test_cells']).isdisjoint(a['outer_test_cells'])
        for name in ['inner_train_cells', 'inner_validation_cells']:
            assert set(a['selection'][name]) <= set(a['train_cells'])
