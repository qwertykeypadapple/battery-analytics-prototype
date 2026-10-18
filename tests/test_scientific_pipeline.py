from copy import deepcopy
from pathlib import Path
import json
import inspect
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
from battery_analytics.common import ROOT, load_config
from battery_analytics.features import extract_charge_features, FEATURE_SETS, FORBIDDEN_FEATURES
from battery_analytics.evaluation import causal_median, errors
from battery_analytics.rul import forecast, end_of_life


@pytest.fixture
def config():
    return load_config()


def synthetic_charge():
    t = np.arange(0., 6001., 5.)
    v = np.minimum(3.5 + .0002 * t, 4.2)
    i = np.where(t <= 3500, 1.5, 1.5 * np.exp(-(t - 3500) / 600))
    temp = 25 + (v - 3.5) * 4
    return t, v, i, temp


def test_upper_window_invariance_to_partial_start(config):
    t, v, i, temp = synthetic_charge()
    full = extract_charge_features(t, v, i, temp, 2., config['features'], 'nasa')
    keep = v >= 3.8
    partial = extract_charge_features(t[keep] - t[keep][0], v[keep], i[keep], temp[keep], 2., config['features'], 'nasa')
    assert full['feature_valid'] and partial['feature_valid']
    for name in set(sum(FEATURE_SETS.values(), [])) - {'ambient_temp_c'}:
        assert partial[name] == pytest.approx(full[name], rel=1e-9, abs=1e-9)


def test_above_window_start_and_missing_temperature_are_explicit(config):
    t, v, i, temp = synthetic_charge()
    keep = v > 3.95
    above = extract_charge_features(t[keep], v[keep], i[keep], temp[keep], 2., config['features'], 'nasa')
    assert not above['feature_valid']
    assert above['feature_reason'] == 'started_above_lower_window'
    result = extract_charge_features(t, v, i, None, 2., config['features'], 'nasa')
    assert result['feature_valid'] and not result['temperature_available']
    assert np.isnan(result['window_mean_temp_c'])
    assert np.isnan(result['window_temp_rise_c'])


def test_feature_interface_and_allowlist_exclude_labels():
    parameters = set(inspect.signature(extract_charge_features).parameters)
    assert not parameters.intersection({'capacity', 'soh', 'label', 'discharge', 'cycle_index'})
    for names in FEATURE_SETS.values():
        assert not FORBIDDEN_FEATURES.intersection(names)


def test_metric_units_and_past_only_median():
    result = errors([1., .8], [.99, .82])
    assert result['mae_pp'] == pytest.approx(1.5)
    assert result['rmse_pp'] == pytest.approx(np.sqrt(2.5))
    frame = pd.DataFrame(dict(dataset=['x'] * 8, cell_id=['a'] * 8, cycle_index=range(8),
                              charge_end_time_s=range(8), prediction=np.arange(8, dtype=float)))
    before = causal_median(frame, window=3)
    frame.loc[5:, 'prediction'] = 999
    after = causal_median(frame, window=3)
    np.testing.assert_equal(before.prediction_smoothed[:5].to_numpy(), after.prediction_smoothed[:5].to_numpy())


def test_rul_is_causal_and_matches_known_linear_trend(config):
    h = pd.DataFrame(dict(cycle_index=np.arange(1., 41.), charge_end_time_s=np.arange(1., 41.),
                          prediction_smoothed=1 - .01 * np.arange(1., 41.)))
    first = forecast(h, 20, config['rul'], 'linear')
    assert first['predicted_rul_cycles'] == pytest.approx(10)
    assert first['predicted_rul_days'] == pytest.approx(5)
    h.loc[h.cycle_index > 20, 'prediction_smoothed'] = 999
    second = forecast(h, 20, config['rul'], 'linear')
    assert first == second


def test_exponential_and_censoring(config):
    h = pd.DataFrame(dict(cycle_index=np.arange(1., 21.), charge_end_time_s=np.arange(1., 21.),
                          prediction_smoothed=.95 * np.exp(-.01 * np.arange(1., 21.))))
    actual = np.log(.70 / .95) / -.01 - 10
    assert forecast(h, 10, config['rul'], 'exponential')['predicted_rul_cycles'] == pytest.approx(actual)
    right = end_of_life(pd.DataFrame({'cycle_index': [0, 100], 'soh': [1, .8]}))
    assert right['censoring'] == 'right_censored' and np.isnan(right['eol_cycle'])
    crossing = end_of_life(pd.DataFrame({'cycle_index': [0, 100], 'soh': [.8, .6]}))
    assert crossing['eol_cycle'] == pytest.approx(50)
    left = end_of_life(pd.DataFrame({'cycle_index': [0, 100], 'soh': [.6, .5]}))
    assert left['censoring'].startswith('left_censored') and np.isnan(left['eol_cycle'])


def test_real_cache_nominal_normalisation_and_temperature():
    cycles = pd.read_parquet(ROOT / 'data/processed/cycles.parquet')
    assert cycles.cycle_uid.is_unique
    expected = cycles.measured_capacity_ah / cycles.rated_capacity_ah
    np.testing.assert_allclose(cycles.soh, expected, equal_nan=True)
    assert cycles.loc[cycles.dataset.eq('calce'), 'window_mean_temp_c'].isna().all()
    assert cycles.loc[cycles.dataset.eq('oxford'), 'cv_duration_s'].isna().all()
    refs = pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet')
    bad = refs[refs.dataset.eq('calce') & (refs.measured_capacity_ah > 2.025)]
    assert len(bad) >= 1 and not bad.label_valid.any()


def test_oxford_observed_time_units_and_discharge_label(config):
    path = next((Path(config['data_root']) / 'oxford_battery/extracted').glob('*.mat'))
    data = loadmat(path, simplify_cells=True, variable_names=['Cell1'])['Cell1']['cyc0000']
    ch, dis = data['C1ch'], data['C1dc']
    seconds = (ch['t'] - ch['t'][0]) * 86400
    current = np.diff(ch['q']) / 1000 * 3600 / np.diff(seconds)
    assert .73 < np.median(current) < .76
    assert 3000 < seconds[-1] < 4000
    refs = pd.read_parquet(ROOT / 'data/processed/capacity_references.parquet')
    label = refs[refs.dataset.eq('oxford') & refs.cell_id.eq('Cell1') & refs.cycle_index.eq(0)].iloc[0]
    assert label.measured_capacity_ah == pytest.approx(abs(dis['q'][-1] - dis['q'][0]) / 1000)
    assert dis['t'][-1] < ch['t'][-1]


def test_actual_validation_splits_and_predictions_are_disjoint():
    audit = json.loads((ROOT / 'reports/evaluation/fold_audit.json').read_text())
    for fold in audit:
        outer_train, outer_test = set(fold['outer_train_cells']), set(fold['outer_test_cells'])
        inner_train, inner_validation = set(fold['inner_train_cells']), set(fold['inner_validation_cells'])
        assert outer_train.isdisjoint(outer_test)
        assert inner_train.isdisjoint(inner_validation)
        assert inner_train | inner_validation == outer_train
        assert inner_train.isdisjoint(outer_test) and inner_validation.isdisjoint(outer_test)
        assert fold['selected_model'] == min(fold['validation_mae_pp'], key=fold['validation_mae_pp'].get)
        if fold['validation'] == 'leave_one_dataset_out':
            assert {x.split(':')[0] for x in outer_train}.isdisjoint(x.split(':')[0] for x in outer_test)
    p = pd.read_parquet(ROOT / 'reports/evaluation/predictions.parquet')
    assert not p.duplicated(['validation', 'variant', 'model', 'cycle_uid']).any()
    selected = p[p.model.eq('training_selected') & p.variant.eq('no_temperature')]
    assert set(selected.validation) == {'leave_one_cell_out', 'leave_one_dataset_out'}


def test_censored_rul_has_no_invented_exact_error():
    r = pd.read_csv(ROOT / 'reports/rul/rul_forecasts.csv')
    censored = r[r.censoring.ne('observed_crossing')]
    assert censored.actual_rul_cycles.isna().all()
    assert censored.absolute_error_cycles.isna().all()
    assert r.loc[r.actual_rul_cycles.notna(), 'actual_rul_cycles'].ge(0).all()
