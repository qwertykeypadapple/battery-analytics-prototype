import inspect
import json
import numpy as np
import pandas as pd
import pytest
from battery_analytics.common import load_config, ROOT
from battery_analytics.improvements.charge import throughput, incremental_capacity, Q_FEATURES
from battery_analytics.improvements.stages import relative_features, anchor_correction
from battery_analytics.improvements.core import verify_baseline
from battery_analytics.features import CORE_FEATURES


def signals():
    t=np.arange(0.,4001.,5.); v=3.5+.0002*t; i=1.5+0*t
    return t,v,i


def test_throughput_charge_only_units_partial_window():
    t,v,i=signals(); cfg=load_config()['features']
    a=throughput(t,v,i,2.,cfg,'nasa'); mask=v>=3.8
    b=throughput(t[mask]-t[mask][0],v[mask],i[mask],2.,cfg,'nasa')
    for c in Q_FEATURES: assert a[c]==pytest.approx(b[c],rel=1e-10)
    assert a['q_lower_upper']==pytest.approx(1400*1.5/(3600*2),rel=1e-8)
    assert a['q_lower_upper']==pytest.approx(sum(a[c] for c in Q_FEATURES[:3]))
    for function in [throughput,incremental_capacity]:
        assert not set(inspect.signature(function).parameters)&{'soh','capacity','discharge','cycle_index'}


def test_relative_features_ignore_labels_and_future_charges():
    d=pd.DataFrame({'group_id':['a']*10,'cycle_index':np.arange(10),'charge_end_time_s':np.arange(10),'soh':np.arange(10)})
    for c in CORE_FEATURES: d[c]=np.arange(10.)
    before=relative_features(d)
    d.loc[8:,CORE_FEATURES]=9999; d['soh']=-999
    after=relative_features(d)
    np.testing.assert_equal(before.filter(like='delta_').iloc[:8].to_numpy(),after.filter(like='delta_').iloc[:8].to_numpy())
    assert before.delta_time_lower_upper_s.iloc[:5].isna().all()
    np.testing.assert_allclose(before.delta_time_lower_upper_s.iloc[5:], np.arange(5.,10.)-2)


def test_entry_anchor_uses_explicit_single_reference():
    assert anchor_correction(np.array([.8,.7]),.95,.9)==pytest.approx([.85,.75])
    assert 'soh' not in inspect.signature(relative_features).parameters
    verify_baseline()


def test_ica_uses_current_charge_and_is_partial_start_invariant():
    t,v,i=signals(); cfg=load_config()['features']
    # Make a resolved bump in dQ/dV, wholly inside the allowed window.
    i=i*(1+.4*np.exp(-((v-4.08)/.02)**2))
    a=incremental_capacity(t,v,i,2.,cfg,'nasa'); mask=v>=3.8
    b=incremental_capacity(t[mask]-t[mask][0],v[mask],i[mask],2.,cfg,'nasa')
    assert a['ica_peak_present']==1
    for c in a: assert a[c]==pytest.approx(b[c],rel=1e-8,abs=1e-10)
    assert 4.04<a['ica_peak_v']<4.12


def test_kalman_filter_is_causal_and_uses_no_reference_labels():
    from battery_analytics.improvements.tracking import filter_cell
    x=np.arange(1.,101.); z=1-.002*x
    pars=dict(time_scale=100.,observation_variance=.0004,rate_mean=-.2,rate_variance=.01,process_std=.001)
    a=filter_cell(x,z,pars); z[50:]=999
    b=filter_cell(x,z,pars)
    np.testing.assert_equal(a[:50],b[:50])
    assert np.all(a[:,2]>=0) and np.all(a[:,4]>=0)
    assert not set(inspect.signature(filter_cell).parameters)&{'soh','reference','labels','capacity'}


def test_probability_forecasts_do_not_read_truth_or_future_rows():
    from battery_analytics.improvements.tracking import risk_columns
    h=pd.DataFrame({'prediction':[.8,.75], 'rate_per_cycle':[-.001,-.001],
        'variance_soh':[.0001,.0001],'covariance_soh_rate':[0.,0.],'variance_rate':[1e-10,1e-10],'soh':[.8,.7]})
    a=risk_columns(h); h['soh']=-999; h.loc[1,'prediction']=99
    b=risk_columns(h)
    for c in [c for c in a if c.startswith(('probability_','rul_'))]: assert a[c].iloc[0]==b[c].iloc[0]
    assert a.probability_28.iloc[0]<a.probability_120.iloc[0]<=a.probability_180.iloc[0]


def test_improvement_selection_and_noise_have_no_test_cells():
    for name in ['E4_combined','E8_forced_ica','E8_optional_ica']:
        audit=json.loads((ROOT/'reports/improvements'/name/'audit.json').read_text())
        for f in audit:
            if 'winner' not in f: continue
            assert set(f['inner_train_cells']).isdisjoint(f['inner_validation_cells'])
            assert set(f['inner_train_cells']+f['inner_validation_cells']).isdisjoint(f['outer_test_cells'])
            scores=f['inner_scores_pp']; minimum=min(scores.values())
            assert f['winner']==next(k for k,v in scores.items() if v<=minimum+.20)
    for f in json.loads((ROOT/'reports/improvements/E5_filter/audit.json').read_text()):
        pars=f['parameters']; noise=set(pars['noise_training_cells'])
        assert noise<=set(f['inner_train_cells'])
        assert noise.isdisjoint(f['inner_validation_cells']+f['outer_test_cells'])
        assert set(pars['prior_cells'])<=noise
        for split in pars['noise_crossfit_splits']:
            assert set(split['train_cells']).isdisjoint(split['validation_cells'])
            assert set(split['train_cells']+split['validation_cells'])<=noise


def test_alarm_thresholds_and_calibration_choices_keep_holdouts_separate():
    for f in json.loads((ROOT/'reports/improvements/E6_probabilistic/threshold_audit.json').read_text()):
        assert set(f['selection_cells']).isdisjoint(f['outer_test_cells'])
    for f in json.loads((ROOT/'reports/improvements/E9_calibration/choices.json').read_text()):
        assert len(f['calibration_cells'])==f['k']
        assert set(f['calibration_cells']).isdisjoint(f['development_test_cells'])
        assert f['locked_cell'] not in f['calibration_cells']+f['development_test_cells']
        assert set(f['source_training_cells']).isdisjoint(f['calibration_cells']+f['development_test_cells']+[f['locked_cell']])


def test_calibration_offset_uses_only_designated_cells():
    from battery_analytics.improvements.stages import calibration_offset
    a=pd.DataFrame({'group_id':['a','a','b'],'soh':[.8,.8,.9]})
    assert calibration_offset(a,[.7,.7,.7])==pytest.approx(.15)
    assert calibration_offset(a.iloc[:0],[])==0


def test_fast_flag_is_unavailable_until_registered_decision_cycle():
    p=pd.read_parquet(ROOT/'reports/improvements/E7_fast/predictions.parquet')
    assert not p.loc[p.cycle_index<p.decision_cycle,'fast_degradation_flag'].any()
    assert not p.loc[p.decision_cycle.isna(),'fast_degradation_flag'].any()
