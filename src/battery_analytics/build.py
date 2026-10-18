"""One-time parsing and auditable feature/label cache."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .common import ROOT, load_config, write_json, completed_charge
from .features import extract_charge_features, FEATURE_SETS
from . import loaders


def fingerprint(config):
    data = Path(config['data_root'])
    items = [(str(p.relative_to(data)), p.stat().st_size, p.stat().st_mtime_ns)
             for d in data.glob('*_battery/extracted') for p in sorted(d.rglob('*')) if p.is_file()]
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(loaders.__file__), ROOT / 'src/battery_analytics/features.py', ROOT / 'src/battery_analytics/common.py']}
    return hashlib.sha256(json.dumps(dict(config=config, inputs=items, code=code), sort_keys=True).encode()).hexdigest()


def build(config=None, force=False):
    config = config or load_config()
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(config)
    meta_path = out / 'cache_metadata.json'
    if not force and meta_path.exists() and json.loads(meta_path.read_text()).get('fingerprint') == signature:
        print('Using verified feature cache', flush=True)
        return pd.read_parquet(out / 'cycles.parquet'), pd.read_parquet(out / 'capacity_references.parquet')
    rows, refs = [], []
    incomplete_reference_ids = set()
    for name in ['nasa', 'oxford', 'calce']:
        started = time.monotonic()
        records, references = getattr(loaders, name)(Path(config['data_root']), config)
        refs.extend(references)
        for record in records:
            row = record.metadata.copy()
            # Labels are attached AFTER the isolated charge feature function returns.
            features = extract_charge_features(record.time_s, record.voltage_v, record.current_a,
                                               record.temperature_c, row['rated_capacity_ah'], config['features'], name)
            row.update(features)
            row['charge_complete_to_protocol'] = completed_charge(record, config)
            if record.reference:
                ref = record.reference
                row.update(measured_capacity_ah=ref['measured_capacity_ah'], soh=ref['soh'], label_valid=ref['label_valid'],
                           label_reason=ref['label_reason'], reference_id=ref['reference_id'], label_cycle_index=ref['cycle_index'],
                           label_source=ref['label_source'], label_pairing=ref['label_pairing'], label_lag_s=ref.get('label_lag_s', np.nan))
            else:
                row.update(measured_capacity_ah=np.nan, soh=np.nan, label_valid=False, label_reason=';'.join(record.notes) or 'no_capacity_pair',
                           reference_id='', label_cycle_index=np.nan, label_source='', label_pairing='', label_lag_s=np.nan)
            if not row['charge_complete_to_protocol'] and record.reference:
                row['label_valid'] = False
                row['label_reason'] = (row['label_reason'] + ';pre_discharge_charge_incomplete').strip(';')
                incomplete_reference_ids.add(row['reference_id'])
            row['notes'] = ';'.join(record.notes)
            row['group_id'] = name + ':' + row['cell_id']
            row['core_model_eligible'] = bool(row['feature_valid'] and row['label_valid'])
            rows.append(row)
        print(f'{name}: {len(records)} charge episodes, {len(references)} capacity references ({time.monotonic() - started:.1f}s)', flush=True)
        del records
    cycles = pd.DataFrame(rows).sort_values(['dataset', 'cell_id', 'cycle_index', 'charge_end_time_s']).reset_index(drop=True)
    capacity = pd.DataFrame(refs).sort_values(['dataset', 'cell_id', 'cycle_index']).reset_index(drop=True)
    incomplete = capacity.reference_id.isin(incomplete_reference_ids)
    capacity.loc[incomplete, 'label_valid'] = False
    capacity.loc[incomplete, 'label_reason'] = capacity.loc[incomplete, 'label_reason'].fillna('').map(
        lambda reason: (reason + ';pre_discharge_charge_incomplete').strip(';'))
    assert cycles.cycle_uid.is_unique
    cycles.to_parquet(out / 'cycles.parquet', index=False)
    cycles.to_csv(out / 'cycles.csv', index=False)
    capacity.to_parquet(out / 'capacity_references.parquet', index=False)
    capacity.to_csv(out / 'capacity_references.csv', index=False)
    rejects = cycles.loc[~cycles.core_model_eligible, ['dataset', 'cell_id', 'cycle_uid', 'cycle_index', 'feature_reason', 'label_reason']]
    rejects.to_csv(out / 'excluded_charge_episodes.csv', index=False)
    coverage = []
    for name, g in cycles.groupby('dataset'):
        r = dict(dataset=name, charge_episodes=len(g), cells=g.cell_id.nunique(), valid_labels=int(g.label_valid.sum()),
                 valid_upper_windows=int(g.feature_valid.sum()), core_eligible=int(g.core_model_eligible.sum()),
                 started_above_window=int(g.feature_reason.eq('started_above_lower_window').sum()),
                 observed_cv_tapers=int(g.cv_available.sum()), observed_cv_durations=int(g.cv_duration_available.sum()),
                 measured_temperature=int(g.temperature_available.sum()))
        for variant, columns in FEATURE_SETS.items():
            r[variant + '_eligible'] = int((g.core_model_eligible & g[columns].notna().all(axis=1)).sum())
        coverage.append(r)
    pd.DataFrame(coverage).to_csv(out / 'feature_coverage.csv', index=False)
    write_json(meta_path, dict(fingerprint=signature, config=config, rows=len(cycles), capacity_references=len(capacity),
                               feature_sets=FEATURE_SETS, coverage=coverage))
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    return cycles, capacity
