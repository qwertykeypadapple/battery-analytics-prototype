"""Leakage, censoring, provenance and count-denominator regression checks."""
import importlib.util
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from battery_analytics.common import ROOT
from battery_analytics.fleet_v2 import (
    AGE_SPANS, RUL_CONFIG, make_layout, audit_predictions, prefix_crossings,
    lookup_forecasts, reference_asof, scenario,
)
from battery_analytics.rul import end_of_life

spec = importlib.util.spec_from_file_location('frozen_fleet_v1', ROOT / 'scripts/fleet_forecast.py')
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


def test_cache_matches_frozen_v1_and_ignores_future_predictions_and_labels():
    x = np.arange(1., 201., 2.)
    h = pd.DataFrame(dict(cycle_index=x, charge_end_time_s=x, prediction=1 - x * .002,
                          soh=np.ones(len(x))))
    ages = [-1, 1, 5, 18, 40.5, 90, 150, 198, 199, 200]
    cache = prefix_crossings(h)
    remaining, reason = lookup_forecasts(cache, ages)
    for i, age in enumerate(ages):
        expected, expected_reason = v1.asof_forecast(h, age, RUL_CONFIG)
        assert remaining[i] == pytest.approx(expected, nan_ok=True)
        assert reason[i] == expected_reason
    h.loc[h.cycle_index > 40.5, 'prediction'] = 999
    h['soh'] = -999
    modified = lookup_forecasts(prefix_crossings(h), [40.5])
    assert modified[0][0] == pytest.approx(remaining[4])
    assert modified[1][0] == reason[4]


def test_future_reference_checks_do_not_enter_eligibility_or_historical_rate():
    h = pd.DataFrame(dict(cycle_index=[0., 10., 20., 30.], soh=[1., .9, .75, .65]))
    e = pd.Series(end_of_life(h))
    before = reference_asof(h, e, [15.])
    h.loc[h.cycle_index > 15, 'soh'] = .1
    after = reference_asof(h, pd.Series(end_of_life(h)), [15.])
    cols = ['already_below', 'no_reference_prefix', 'historical_events', 'historical_exposure_days']
    pd.testing.assert_frame_equal(before[cols], after[cols])
    assert before.historical_exposure_days.iloc[0] == 5
    assert before.historical_events.iloc[0] == 0


def test_censored_followup_and_left_censored_early_prefix_are_unknown():
    h = pd.DataFrame(dict(cycle_index=[0., 100.], soh=[1., .8]))
    result = reference_asof(h, pd.Series(end_of_life(h)), [0., 80., 120.])
    assert result.actual_30.iloc[0] == 0
    assert result.actual_30.iloc[1:].isna().all()
    left = pd.DataFrame(dict(cycle_index=[10., 20.], soh=[.6, .5]))
    result = reference_asof(left, pd.Series(end_of_life(left)), [0., 10.])
    assert result.actual_30.isna().all()
    assert result.already_below.tolist() == [False, True]


def test_counts_exposure_and_unknown_ranges_on_hand_calculated_cohort():
    g = pd.DataFrame(dict(
        group_id=['calce:a', 'calce:a', 'calce:b', 'nasa:c'],
        dataset=['calce', 'calce', 'calce', 'nasa'], already_below=[True, False, False, False],
        initially_left_censored=False, no_reference_prefix=False, retrospective_preexisting=False,
        historical_exposure_days=10., historical_events=[1, 0, 0, 0],
        charge_rul=[0., 10., np.nan, 100.], cycle_rul=[0., 20., 20., 20.],
        charge_reason=['predicted_overdue', 'estimated', 'history_ended', 'estimated'],
        cycle_reason=['predicted_overdue', 'estimated', 'estimated', 'estimated'],
        actual_30=[0., 1., 0., np.nan]))
    r = scenario(g, 30, 4)
    assert r['at_risk'] == 3 and r['unique_sources'] == 3
    assert r['historical_rate_per_battery_day'] == .025
    assert r['historical_subtotal'] == 2.25
    assert (r['actual_count_lower'], r['actual_count_upper']) == (1., 2.)
    assert (r['charge_subtotal'], r['charge_count_upper']) == (1., 2.)
    assert (r['charge_full_error_low'], r['charge_full_error_high']) == (-1., 1.)
    assert (r['charge_full_abs_error_low'], r['charge_full_abs_error_high']) == (0., 1.)
    assert r['matched_n'] == 1 and r['matched_unique'] == 1
    assert r['matched_actual'] == r['matched_charge'] == r['matched_cycle'] == 1
    assert r['matched_historical'] == .75
    assert r['charge_known_n'] == 1 and r['cycle_known_n'] == 2


def test_fixed_layouts_preserve_source_support_and_stagger_ages():
    inv = pd.DataFrame(dict(dataset=['calce'] * 2 + ['nasa'] * 2 + ['oxford'] * 2,
                            cell_id=['a', 'b'] * 3))
    for kind in ['calce', 'mixed']:
        layout = make_layout(inv, kind, 2026101800)
        pd.testing.assert_frame_equal(layout, make_layout(inv, kind, 2026101800))
        assert len(layout) == layout.commissioning_day.nunique() == 1000
        assert layout.group_id.nunique() <= (2 if kind == 'calce' else 6)
        assert (layout.commissioning_day <= 0).all()
        assert (-layout.commissioning_day <= layout.dataset.map(AGE_SPANS)).all()
        assert set(layout.iloc[:50].group_id) <= set(layout.iloc[:200].group_id)


def test_provenance_audit_rejects_a_source_in_training():
    p = pd.DataFrame(dict(validation=['leave_one_dataset_out'], cycle_uid=['x'],
                          group_id=['calce:a'], dataset=['calce'], cell_id=['a']))
    inventory = pd.DataFrame(dict(dataset=['calce'], cell_id=['a']))
    f = dict(validation='leave_one_dataset_out', fold='calce', variant='no_temperature',
             outer_train_cells=['calce:b'], outer_test_cells=['calce:a'],
             inner_train_cells=[], inner_validation_cells=[])
    # A different cell from the same dataset also violates LODO.
    with pytest.raises(AssertionError):
        audit_predictions({'charge': p}, [f], [], inventory)
    f['outer_train_cells'] = ['calce:a']
    with pytest.raises(AssertionError):
        audit_predictions({'charge': p}, [f], [], inventory)


def test_saved_v2_results_reconcile_populations_bounds_and_preserved_artifacts():
    out = ROOT / 'reports/fleet_forecast_v2'
    if not (out / 'scenarios.csv').exists():
        pytest.skip('Run the preregistered experiment to create integration artifacts')
    s = pd.read_csv(out / 'scenarios.csv')
    assert len(s) == 7200
    assert (s.nominal == s.at_risk + s.already_below).all()
    assert (s.matched_actual.dropna() <= s.loc[s.matched_actual.notna(), 'matched_n']).all()
    assert (s.actual_count_upper == s.actual_count_lower + s.unknown_actual).all()
    for method in ('charge', 'cycle'):
        assert (s[f'{method}_unknown'] + s[f'{method}_usable'] == s.at_risk).all()
        assert (s.matched_n <= s[f'{method}_usable']).all()
        assert (s[f'{method}_subtotal'] <= s[f'{method}_usable']).all()
        assert (s[f'{method}_count_upper'] <= s.at_risk).all()
        assert s.loc[s.matched_n.eq(0), f'{method}_matched_error'].isna().all()
        assert (s[f'{method}_full_error_low'] <= s[f'{method}_full_error_high']).all()
        assert (s[f'{method}_full_abs_error_low'] <= s[f'{method}_full_abs_error_high']).all()
    assert s[s.kind.eq('calce') & s.nominal.eq(1000)].unique_sources.eq(7).all()
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['protected_sha256_before'] == manifest['protected_sha256_after']
    for path, sha in manifest['protected_sha256_before'].items():
        if path.startswith(('reports/fleet_forecast/', 'reports/improvements/baseline/')):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
