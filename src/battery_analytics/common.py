from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / 'config.yaml'
    config = yaml.safe_load(path.read_text())
    config['data_root'] = str((path.parent / config['data_root']).resolve())
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert) + '\n')


@dataclass
class ChargeRecord:
    metadata: dict
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray | None
    reference: dict | None = None
    notes: list[str] = field(default_factory=list)


def completed_charge(record, config):
    """Check documented CC-CV termination without seeing the capacity label.

    Oxford's stored C1 reference is CC-only, with its separately measured same-
    checkpoint discharge label. It must not inherit a made-up CV requirement.
    """
    dataset = record.metadata['dataset']
    if dataset == 'oxford':
        return True
    cutoff = config['quality']['charge_cutoff_a'][dataset]
    limit = config['features']['datasets'][dataset]['voltage_limit_v']
    i, v = np.asarray(record.current_a), np.asarray(record.voltage_v)
    termination = np.isfinite(i) & np.isfinite(v) & (v >= limit - .03) & (i >= .5 * cutoff) & (i <= 1.5 * cutoff + .005)
    return bool(termination.any())


def capacity_quality(q, nominal, t, v, i, config, *, check_integral=False):
    """Label QC is isolated from charge feature extraction."""
    reasons = []
    if not np.isfinite(q) or q <= 0:
        reasons.append('capacity_missing_or_nonpositive')
    elif q > nominal * config['maximum_capacity_multiple']:
        reasons.append('capacity_above_physical_review_bound')
    t, v, i = map(lambda a: np.asarray(a, dtype=float), (t, v, i))
    valid = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if valid.sum() < config['min_discharge_samples'] or valid.mean() < .95:
        reasons.append('discharge_missing_or_short_trace')
        return False, ';'.join(reasons), np.nan
    t, v, i = t[valid], v[valid], i[valid]
    if (np.diff(t) < 0).any():
        reasons.append('discharge_time_runs_backwards')
    if np.nanmax(v) < 3.8 or np.nanmin(v) > 2.85 or np.ptp(v) < .7:
        reasons.append('discharge_not_full_voltage_range')
    integral = float(np.trapz(np.maximum(-i, 0), t) / 3600)
    if check_integral and np.isfinite(q):
        tolerance = config['capacity_absolute_tolerance_ah'] + config['maximum_integration_relative_error'] * abs(q)
        if abs(integral - q) > tolerance:
            reasons.append('discharge_capacity_counter_integral_mismatch')
    return not reasons, ';'.join(reasons), integral
