import importlib.util
import numpy as np
import pandas as pd
from battery_analytics.common import ROOT, load_config

spec = importlib.util.spec_from_file_location('fleet_count_script', ROOT/'scripts/fleet_forecast.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fleet_count_forecast_ignores_future_predictions_and_labels():
    h=pd.DataFrame({'cycle_index':np.arange(1.,101.), 'charge_end_time_s':np.arange(1.,101.),
                    'prediction':1-np.arange(1.,101.)*.003, 'soh':np.ones(100)})
    before=module.asof_forecast(h,40,load_config()['rul'])
    h.loc[h.cycle_index>40,'prediction']=999
    h['soh']=-999
    assert module.asof_forecast(h,40,load_config()['rul'])==before


def test_unknown_followup_is_not_a_zero_count():
    e=pd.Series({'censoring':'right_censored','last_reference_cycle':100})
    assert np.isnan(module.known_outcome(e,80,60))
    assert module.known_outcome(e,20,60)==0
