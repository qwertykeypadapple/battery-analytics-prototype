"""Features have no access to a capacity label, cell age, or discharge trace."""
from __future__ import annotations
import numpy as np
import pandas as pd

CORE_FEATURES = ['time_lower_middle_s', 'time_middle_upper_middle_s',
                 'time_upper_middle_upper_s', 'time_lower_upper_s', 'cc_current_c']
TEMPERATURE_FEATURES = CORE_FEATURES + ['window_mean_temp_c', 'window_temp_rise_c', 'ambient_temp_c']
CV_FEATURES = CORE_FEATURES + ['time_lower_cv_s', 'cv_duration_s', 'cv_taper_slope_c_per_s', 'cv_tau_s']
FEATURE_SETS = {'no_temperature': CORE_FEATURES, 'temperature': TEMPERATURE_FEATURES, 'cv_no_temperature': CV_FEATURES}
FORBIDDEN_FEATURES = {'soh', 'measured_capacity_ah', 'discharge_capacity', 'label_integral_ah',
                      'cycle_index', 'cell_id', 'dataset', 'label_cycle_index', 'rated_capacity_ah'}


def first_crossing(t, v, level):
    """Interpolate a voltage crossing supported by subsequent samples."""
    indices = np.flatnonzero(v >= level)
    for k in indices:
        tail = v[k:min(k + 3, len(v))]
        if len(tail) >= 2 and np.all(tail >= level - .003):
            if k and v[k] != v[k - 1]:
                fraction = np.clip((level - v[k - 1]) / (v[k] - v[k - 1]), 0, 1)
                return float(t[k - 1] + fraction * (t[k] - t[k - 1]))
            return float(t[k])
    return np.nan


def extract_charge_features(time_s, voltage_v, current_a, temperature_c, rated_capacity_ah, config, dataset):
    """Return upper-window charge features and explicit eligibility reasons.

    Only the five signal/specification arguments above enter feature calculations.
    Neither full charge quantity nor any discharge channel enters this function.
    """
    levels = config['datasets'][dataset]
    output = {k: np.nan for k in set(TEMPERATURE_FEATURES + CV_FEATURES) - {'ambient_temp_c'}}
    output.update(feature_valid=False, feature_reason='', cv_available=False, cv_duration_available=False, cv_reason='not_observed',
                  temperature_available=False, starting_charge_voltage_v=np.nan, window_sample_count=0)
    t, v, i = (np.asarray(a, dtype=float) for a in (time_s, voltage_v, current_a))
    temp = np.asarray(temperature_c, float) if temperature_c is not None else np.full(len(t), np.nan)
    if not (len(t) == len(v) == len(i) == len(temp)):
        output['feature_reason'] = 'signal_length_mismatch'
        return output
    valid = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if valid.sum() < config['min_samples']:
        output['feature_reason'] = 'short_or_nonfinite_charge'
        return output
    t, v, i, temp = (a[valid] for a in (t, v, i, temp))
    if np.any(np.diff(t) < 0):
        output['feature_reason'] = 'charge_time_runs_backwards'
        return output
    # Duplicate samples at cycler step boundaries do not create elapsed time.
    keep = np.r_[np.diff(t) > 0, True]
    t, v, i, temp = (a[keep] for a in (t, v, i, temp))
    positive = np.flatnonzero(i > config['min_positive_current_c'] * rated_capacity_ah)
    if len(positive) < config['min_samples']:
        output['feature_reason'] = 'no_substantial_positive_charge'
        return output
    t, v, i, temp = (a[positive[0]:positive[-1] + 1] for a in (t, v, i, temp))
    output['starting_charge_voltage_v'] = float(v[0])
    if v[0] > levels['lower_v']:
        output['feature_reason'] = 'started_above_lower_window'
        return output
    if ((v < 0) | (v > levels['voltage_limit_v'] + .3)).any():
        output['feature_reason'] = 'charge_voltage_outside_physical_bounds'
        return output
    smooth_v = pd.Series(v).rolling(config['voltage_smoothing_samples'], center=True, min_periods=1).median().to_numpy()
    crossings = {name: first_crossing(t, smooth_v, levels[name]) for name in ['lower_v', 'middle_v', 'upper_middle_v', 'upper_v', 'cv_v']}
    required = [crossings[k] for k in ['lower_v', 'middle_v', 'upper_middle_v', 'upper_v']]
    if not np.all(np.isfinite(required)) or not np.all(np.diff(required) > 0):
        output['feature_reason'] = 'incomplete_upper_voltage_window'
        return output
    lo, middle, high_middle, hi = required
    mask = (t >= lo) & (t <= hi)
    output['window_sample_count'] = int(mask.sum())
    if mask.sum() < 6:
        output['feature_reason'] = 'too_few_window_samples'
        return output
    if hi - lo > config['max_window_duration_s'] or np.max(np.diff(np.r_[lo, t[mask], hi])) > config['max_gap_s']:
        output['feature_reason'] = 'window_duration_or_gap_too_large'
        return output
    cc = (t >= lo) & (t <= high_middle) & (i > .01 * rated_capacity_ah)
    icc = float(np.median(i[cc])) if cc.any() else np.nan
    if not np.isfinite(icc) or icc <= 0 or np.std(i[cc]) / icc > .20:
        output['feature_reason'] = 'unstable_or_missing_cc_current'
        return output
    output.update(time_lower_middle_s=middle - lo, time_middle_upper_middle_s=high_middle - middle,
                  time_upper_middle_upper_s=hi - high_middle, time_lower_upper_s=hi - lo,
                  cc_current_c=icc / rated_capacity_ah, feature_valid=True, feature_reason='ok')
    if np.isfinite(temp[mask]).mean() >= .95:
        good_temp = np.isfinite(temp)
        output.update(window_mean_temp_c=float(np.mean(temp[mask & good_temp])),
                      window_temp_rise_c=float(np.interp(hi, t[good_temp], temp[good_temp]) - np.interp(lo, t[good_temp], temp[good_temp])),
                      temperature_available=True)
    cv_start = crossings['cv_v']
    if np.isfinite(cv_start):
        output['time_lower_cv_s'] = cv_start - lo
        cmask = (t >= cv_start) & (i > config['min_positive_current_c'] * rated_capacity_ah) & (v >= levels['cv_v'] - .02)
        if cmask.sum() >= 2:
            ct, ci = t[cmask], i[cmask]
            if np.min(ci) < .8 * icc and v[-1] >= levels['cv_v'] - .02:
                duration = float(ct[-1] - cv_start)
                if duration > 10:
                    output.update(cv_duration_s=duration, cv_duration_available=True, cv_reason='duration_observed_taper_undersampled')
                # CALCE logs many CV points on current-change events rather than
                # a fixed cadence. Assess relative coverage, and weight by elapsed
                # time so densely logged high-current points do not dominate.
                gap_bound = max(config['max_gap_s'], config['max_cv_gap_fraction'] * duration)
                if len(ct) >= config['min_cv_samples'] and duration > 10 and np.max(np.diff(ct)) <= gap_bound:
                    weights = np.sqrt(np.maximum(np.gradient(ct), 1e-6))
                    linear_slope = float(np.polyfit(ct - ct[0], ci / rated_capacity_ah, 1, w=weights)[0])
                    log_slope = float(np.polyfit(ct - ct[0], np.log(ci), 1, w=weights)[0])
                    if log_slope < 0 and linear_slope < 0:
                        output.update(cv_duration_s=duration, cv_taper_slope_c_per_s=linear_slope,
                                      cv_tau_s=-1 / log_slope, cv_available=True, cv_reason='observed_taper')
    return output
