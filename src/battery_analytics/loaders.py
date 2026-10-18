"""Dataset-specific parsers, with independent charge inputs and capacity labels."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .common import ChargeRecord, capacity_quality


def nasa(root: Path, config):
    base = root / 'nasa_battery/extracted/cleaned_dataset'
    md = pd.read_csv(base / 'metadata.csv')
    references = []
    records = []
    for cid, cell in md.groupby('battery_id', sort=True):
        cell = cell.sort_values('test_id').reset_index(drop=True)
        ref_by_index = {}
        discharge_number = 0
        for j, row in cell.iterrows():
            if row.type != 'discharge':
                continue
            discharge_number += 1
            d = pd.read_csv(base / 'data' / row.filename)
            q = pd.to_numeric(pd.Series([row.Capacity]), errors='coerce').iloc[0]
            valid, reason, integral = capacity_quality(q, 2., d.Time, d.Voltage_measured, d.Current_measured, config['quality'])
            ref = dict(dataset='nasa', cell_id=cid, cycle_index=float(discharge_number), reference_id=f'nasa:{cid}:{row.filename}',
                       measured_capacity_ah=q, rated_capacity_ah=2., soh=q / 2., ambient_temp_c=float(row.ambient_temperature),
                       label_valid=valid, label_reason=reason, label_integral_ah=integral,
                       label_source=str((base / 'data' / row.filename).relative_to(root)),
                       source_test_id=int(row.test_id), label_pairing='next_discharge',
                       low_capacity_review=bool(np.isfinite(q) and 0 < q < .2))
            references.append(ref)
            ref_by_index[j] = ref
        for j, row in cell.iterrows():
            if row.type != 'charge':
                continue
            d = pd.read_csv(base / 'data' / row.filename)
            later = cell.iloc[j + 1:]
            relevant = later[later.type.isin(['charge', 'discharge'])]
            ref = None
            notes = []
            if len(relevant) and relevant.iloc[0].type == 'discharge':
                ref = ref_by_index[relevant.index[0]].copy()
                if abs(ref['ambient_temp_c'] - row.ambient_temperature) > config['quality']['maximum_label_temperature_difference_c']:
                    ref.update(label_valid=False, label_reason='charge_discharge_ambient_mismatch')
            elif len(relevant):
                notes.append('superseded_by_later_charge')
            else:
                notes.append('no_following_discharge')
            cycle_number = float(ref['cycle_index']) if ref else float(cell.iloc[:j + 1].type.eq('discharge').sum() + 1)
            meta = dict(dataset='nasa', cell_id=cid, cycle_index=cycle_number, source_cycle_index=int(row.test_id), episode_index=1,
                        cycle_uid=f'nasa:{cid}:{row.filename}', rated_capacity_ah=2., ambient_temp_c=float(row.ambient_temperature),
                        ambient_source='measured_test_setting', charge_source=str((base / 'data' / row.filename).relative_to(root)),
                        current_source='measured', charge_start_time_s=float(d.Time.iloc[0]), charge_end_time_s=float(d.Time.iloc[-1]))
            records.append(ChargeRecord(meta, d.Time.to_numpy(), d.Voltage_measured.to_numpy(), d.Current_measured.to_numpy(),
                                        d.Temperature_measured.to_numpy(), ref, notes))
    return records, references


def oxford(root: Path, config):
    path = next((root / 'oxford_battery/extracted').glob('*.mat'))
    mat = loadmat(path, simplify_cells=True)
    records, references = [], []
    for cid, checkpoints in mat.items():
        if cid.startswith('__'):
            continue
        for key, check in checkpoints.items():
            cyc = int(key.removeprefix('cyc'))
            dis = check.get('C1dc')
            ref = None
            if dis is not None:
                dt = (np.asarray(dis['t']) - dis['t'][0]) * 86400
                dq = np.asarray(dis['q']) / 1000
                di = np.gradient(dq, dt) * 3600
                q = float(abs(dq[-1] - dq[0]))
                valid, reason, integral = capacity_quality(q, .740, dt, dis['v'], di, config['quality'])
                ref = dict(dataset='oxford', cell_id=cid, cycle_index=float(cyc), reference_id=f'oxford:{cid}:{key}:C1dc',
                           measured_capacity_ah=q, rated_capacity_ah=.740, soh=q / .740, ambient_temp_c=40.,
                           label_valid=valid, label_reason=reason, label_integral_ah=integral,
                           label_source=f'{path.relative_to(root)}::{cid}.{key}.C1dc',
                           label_pairing='nearest_same_checkpoint_C1dc',
                           label_start_time_s=float(dis['t'][0] * 86400), label_end_time_s=float(dis['t'][-1] * 86400),
                           low_capacity_review=False)
                references.append(ref)
            charge = check.get('C1ch')
            if charge is None:
                continue
            t = (np.asarray(charge['t']) - charge['t'][0]) * 86400
            # Current reconstruction uses ONLY this charge trace's q(t).
            i = np.gradient(np.asarray(charge['q']) / 1000, t) * 3600
            meta = dict(dataset='oxford', cell_id=cid, cycle_index=float(cyc), source_cycle_index=cyc, episode_index=1,
                        cycle_uid=f'oxford:{cid}:{key}', rated_capacity_ah=.740, ambient_temp_c=40.,
                        ambient_source='documented_chamber_setting', charge_source=f'{path.relative_to(root)}::{cid}.{key}.C1ch',
                        current_source='derived_from_charging_q_only', charge_start_time_s=float(charge['t'][0] * 86400),
                        charge_end_time_s=float(charge['t'][-1] * 86400))
            if ref:
                ref = ref.copy()
                ref['label_lag_s'] = ref['label_end_time_s'] - meta['charge_end_time_s']
            records.append(ChargeRecord(meta, t, np.asarray(charge['v']), i, np.asarray(charge['T']), ref))
    return records, references


def active_runs(d, nominal):
    """Sign episodes ignore zero-current rests but retain measured chronological order."""
    current = d['Current (A)'].to_numpy()
    active = np.flatnonzero(np.abs(current) > .02 * nominal)
    if not len(active):
        return []
    signs = np.sign(current[active])
    boundaries = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1]) + 1, len(active)]
    return [(int(signs[a]), int(active[a]), int(active[b - 1])) for a, b in zip(boundaries[:-1], boundaries[1:])]


def calce(root: Path, config):
    base = root / 'calce_battery/extracted'
    records, references = [], []
    for path in sorted(base.glob('*timeseries.csv')):
        cid = re.search(r'CALCE_(CX2-\d+)_', path.name).group(1)
        d = pd.read_csv(path)
        runs = active_runs(d, 1.35)
        ref_by_run = {}
        episode_counts = {}
        for n, (sign, first, last) in enumerate(runs):
            if sign >= 0:
                continue
            g = d.iloc[first:last + 1]
            q_values = g['Discharge_Capacity (Ah)'].to_numpy()
            q = float(np.max(q_values) - np.min(q_values))
            valid, reason, integral = capacity_quality(q, 1.35, g['Test_Time (s)'], g['Voltage (V)'], g['Current (A)'],
                                                     config['quality'], check_integral=True)
            if (np.diff(q_values) < -.02).any():
                valid, reason = False, (reason + ';discharge_counter_reset_inside_episode').strip(';')
            if len(g) > 1 and np.diff(g['Test_Time (s)']).max() > 600:
                valid, reason = False, (reason + ';discharge_large_time_gap').strip(';')
            cyc = float(g.Cycle_Index.iloc[0])
            ref = dict(dataset='calce', cell_id=cid, cycle_index=cyc, reference_id=f'calce:{cid}:discharge_run{n}',
                       measured_capacity_ah=q, rated_capacity_ah=1.35, soh=q / 1.35, ambient_temp_c=25.,
                       label_valid=valid, label_reason=reason, label_integral_ah=integral,
                       label_source=f'{path.relative_to(root)}::rows{first}-{last}',
                       label_pairing='next_discharge_episode', label_start_time_s=float(g['Test_Time (s)'].iloc[0]),
                       label_end_time_s=float(g['Test_Time (s)'].iloc[-1]), low_capacity_review=bool(0 < q < .135))
            references.append(ref)
            ref_by_run[n] = ref
        for n, (sign, first, last) in enumerate(runs):
            if sign <= 0:
                continue
            g = d.iloc[first:last + 1]
            cyc = float(g.Cycle_Index.iloc[0])
            episode_counts[cyc] = episode_counts.get(cyc, 0) + 1
            ref = ref_by_run.get(n + 1)
            if ref:
                ref = ref.copy()
                if ref['label_start_time_s'] < float(g['Test_Time (s)'].iloc[-1]):
                    ref.update(label_valid=False, label_reason='discharge_precedes_charge_end')
                if ref['cycle_index'] - cyc > 1:
                    ref.update(label_valid=False, label_reason='capacity_check_too_far_from_charge')
            meta = dict(dataset='calce', cell_id=cid, cycle_index=cyc, source_cycle_index=cyc,
                        episode_index=episode_counts[cyc], cycle_uid=f'calce:{cid}:charge_run{n}', rated_capacity_ah=1.35,
                        ambient_temp_c=25., ambient_source='filename_only_not_measured',
                        charge_source=f'{path.relative_to(root)}::rows{first}-{last}', current_source='measured',
                        charge_start_time_s=float(g['Test_Time (s)'].iloc[0]), charge_end_time_s=float(g['Test_Time (s)'].iloc[-1]))
            records.append(ChargeRecord(meta, g['Test_Time (s)'].to_numpy(), g['Voltage (V)'].to_numpy(),
                                        g['Current (A)'].to_numpy(), None, ref))
    return records, references
