"""Run the preregistered survival study; never rewrite earlier experiments."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import joblib
import numpy as np
import pandas as pd
from battery_analytics.common import load_config
from battery_analytics.improvements.core import baseline, folds, verify_baseline, BASE
from battery_analytics.survival_training import fit_outer_folds

OUT = ROOT / 'reports/survival'
PREREG = 'reports/survival/preregistration.md'
NEW = ('reports/survival/', 'scripts/survival.py', 'src/battery_analytics/survival', 'tests/test_survival.py')
INPUTS = [
    'config.yaml', 'data/processed/cycles.parquet', 'data/processed/capacity_references.parquet',
    'reports/improvements/baseline/predictions.parquet', 'reports/improvements/baseline/end_of_life.csv',
    'reports/improvements/baseline/fold_audit.json', 'reports/improvements/baseline/manifest.json',
    'reports/improvements/E6_probabilistic/predictions.parquet', 'reports/improvements/E6_probabilistic/alert_events.csv',
    'reports/review/alarm_observations.csv', 'reports/inventory/cells.csv',
    'reports/fleet_forecast_v2/layouts.csv', 'reports/fleet_forecast_v2/copy_date_audit.parquet',
    'reports/fleet_forecast_v2/scenarios.csv', 'reports/hybrid_fleet/rank_audit.parquet',
    'reports/hybrid_fleet/case_audit.csv', 'reports/hybrid_fleet/scenario_metrics.csv',
    'reports/hybrid_fleet/unique_layouts.csv',
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def prereg_commit():
    commit = git('log', '--diff-filter=A', '--format=%H', '--', PREREG).decode().strip()
    assert len(commit) == 40 and '\n' not in commit, 'Commit preregistration first'
    assert git('show', f'{commit}:{PREREG}') == (ROOT / PREREG).read_bytes(), 'Preregistration was changed'
    return commit


def load_endpoints():
    e = pd.read_csv(BASE / 'end_of_life.csv')
    e['group_id'] = e.dataset + ':' + e.cell_id
    return e


def fit_stage():
    verify_baseline()
    commit = prereg_commit()
    protected = [p for p in git('ls-files', '-z').decode().split('\0') if p and not p.startswith(NEW)]
    state = dict(started_utc=datetime.now(timezone.utc).isoformat(), preregistration_commit=commit,
                 preregistration_sha256=sha(PREREG), repository_head=git('rev-parse', 'HEAD').decode().strip(),
                 input_sha256={p: sha(p) for p in INPUTS},
                 protected_sha256_before={p: sha(p) for p in protected},
                 fit_code_sha256={p: sha(p) for p in ['scripts/survival.py', 'src/battery_analytics/survival.py',
                                                      'src/battery_analytics/survival_training.py']},
                 python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    (OUT / 'execution_start.json').write_text(json.dumps(state, indent=2) + '\n')
    e, p = load_endpoints(), baseline()
    result = fit_outer_folds(folds(), pd.read_parquet(ROOT / 'data/processed/cycles.parquet'), p, e, load_config())
    joblib.dump(result['bundles'], OUT / 'models.joblib', compress=3)
    for name, key in [('outer_probabilities', 'outer'), ('inner_probabilities', 'inner'),
                      ('training_person_periods', 'periods'), ('nested_charge_predictions', 'charge_predictions')]:
        result[key].to_parquet(OUT / f'{name}.parquet', index=False)
    for name, key in [('threshold_audit', 'thresholds'), ('model_audit', 'models'), ('nested_charge_audit', 'charge_audit')]:
        (OUT / f'{name}.json').write_text(json.dumps(result[key], indent=2) + '\n')
    pd.DataFrame(result['models']).to_csv(OUT / 'model_audit.csv', index=False)
    e['has_frozen_predictions'] = e.group_id.isin(p.group_id)
    e.to_csv(OUT / 'source_inventory.csv', index=False)
    state['fit_completed_utc'] = datetime.now(timezone.utc).isoformat()
    assert state['protected_sha256_before'] == {p: sha(p) for p in protected}
    (OUT / 'execution_start.json').write_text(json.dumps(state, indent=2) + '\n')
    print('Saved all fixed fold fits and threshold audits', flush=True)


def evaluate_stage():
    from battery_analytics.survival_evaluation import evaluate
    state = json.loads((OUT / 'execution_start.json').read_text())
    assert state['preregistration_commit'] == prereg_commit()
    assert state['preregistration_sha256'] == sha(PREREG)
    for p, digest in state['input_sha256'].items():
        assert sha(p) == digest, p
    for p, digest in state['fit_code_sha256'].items():
        if p != 'scripts/survival.py':
            assert sha(p) == digest, 'Fit code changed after fitting: ' + p
    evaluate(ROOT, OUT, joblib.load(OUT / 'models.joblib'), load_endpoints(), baseline())
    finish_manifest(state)


def finish_manifest(state):
    protected_after = {p: sha(p) for p in state['protected_sha256_before']}
    assert state['protected_sha256_before'] == protected_after
    verify_baseline()
    state.update(completed_utc=datetime.now(timezone.utc).isoformat(), protected_files_unchanged=True,
                 protected_sha256_after=protected_after,
                 code_sha256={str(p.relative_to(ROOT)): sha(str(p.relative_to(ROOT)))
                              for p in sorted((ROOT / 'src/battery_analytics').glob('survival*.py'))} | {'scripts/survival.py': sha('scripts/survival.py')},
                 result_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                                if p.is_file() and p.name != 'run_manifest.json'})
    (OUT / 'run_manifest.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'fit', 'evaluate', 'report'], default='all')
    args = parser.parse_args()
    if args.stage in ('all', 'fit'):
        fit_stage()
    if args.stage in ('all', 'evaluate'):
        evaluate_stage()
    if args.stage == 'report':
        from battery_analytics.survival_reporting import report
        previous = json.loads((OUT / 'run_manifest.json').read_text())
        raw_names = [name for name in previous['result_sha256'] if name not in
                     ('results.md', 'calibration.png', 'calibration_support.csv')]
        raw_hashes = {name: sha(f'reports/survival/{name}') for name in raw_names}
        assert raw_hashes == {name: previous['result_sha256'][name] for name in raw_names}
        report(ROOT, OUT)
        assert raw_hashes == {name: sha(f'reports/survival/{name}') for name in raw_names}
        state = json.loads((OUT / 'execution_start.json').read_text())
        state['evaluation_code_sha256'] = previous.get('evaluation_code_sha256', previous['code_sha256'])
        state['report_refresh'] = dict(utc=datetime.now(timezone.utc).isoformat(),
            reason='Clarify protocol-specific results, censored-history use, positive known-history timing and negative fleet/hybrid transfer; no scoring changes',
            raw_artifacts_sha256_unchanged=raw_hashes)
        finish_manifest(state)
