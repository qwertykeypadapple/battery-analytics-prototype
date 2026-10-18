import inspect
import numpy as np
import pandas as pd
import pytest
from battery_analytics.common import load_config
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
    np.testing.assert_allclose(before.delta_time_lower_upper_s, np.arange(10.)-2)


def test_entry_anchor_uses_explicit_single_reference():
    assert anchor_correction(np.array([.8,.7]),.95,.9)==pytest.approx([.85,.75])
    assert 'soh' not in inspect.signature(relative_features).parameters
    verify_baseline()
