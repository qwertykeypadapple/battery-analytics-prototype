from __future__ import annotations
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from .build import build
from .common import ROOT, load_config, write_json
from .evaluation import causal_median, metric_tables
from .features import FEATURE_SETS, FORBIDDEN_FEATURES
from .models import MODEL_NAMES, inner_selection, fit_estimator


def train(config=None):
    config = config or load_config()
    cycles, _ = build(config)
    out = ROOT / 'reports/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / 'models'
    model_dir.mkdir(exist_ok=True)
    predictions, audit, omissions = [], [], []
    started = time.monotonic()
    for variant, columns in FEATURE_SETS.items():
        assert not FORBIDDEN_FEATURES.intersection(columns), 'Label/identifier found in model features'
        data = cycles.loc[cycles.core_model_eligible & cycles[columns].notna().all(axis=1)].copy()
        # A CV-taper sensitivity analysis needs more than isolated resolved episodes.
        counts = data.groupby('group_id').size()
        if variant == 'cv_no_temperature':
            enough = counts[counts >= 6].index
            for group in counts[counts < 6].index:
                omissions.append(dict(variant=variant, group_id=group, reason='fewer_than_six_resolved_cv_episodes'))
            data = data[data.group_id.isin(enough)]
        protocols = ['leave_one_cell_out'] + (['leave_one_dataset_out'] if variant == 'no_temperature' else [])
        for protocol in protocols:
            folds = []
            if protocol == 'leave_one_cell_out':
                for dataset, g in data.groupby('dataset'):
                    groups = sorted(g.group_id.unique())
                    if len(groups) < 4:
                        omissions.append(dict(variant=variant, dataset=dataset, reason='fewer_than_four_cells_for_nested_selection'))
                        continue
                    for group in groups:
                        folds.append((group, g[g.group_id.ne(group)], g[g.group_id.eq(group)]))
            else:
                assert set(data.dataset.unique()) == {'nasa', 'oxford', 'calce'}
                for dataset in sorted(data.dataset.unique()):
                    folds.append((dataset, data[data.dataset.ne(dataset)], data[data.dataset.eq(dataset)]))
            for fold_n, (fold, training, test) in enumerate(folds):
                assert set(training.group_id).isdisjoint(test.group_id)
                if protocol == 'leave_one_dataset_out':
                    assert training.dataset.nunique() == 2 and test.dataset.nunique() == 1
                winner, selection = inner_selection(training, columns, config)
                assert set(selection['inner_train_cells'] + selection['inner_validation_cells']).isdisjoint(test.group_id)
                audit.append(dict(validation=protocol, variant=variant, fold=fold, features=columns,
                                  outer_train_cells=sorted(training.group_id.unique()), outer_test_cells=sorted(test.group_id.unique()),
                                  selected_model=winner, train_rows=len(training), test_rows=len(test), **selection))
                for name in MODEL_NAMES:
                    fitted = fit_estimator(name, training, columns, config)
                    keep = ['cycle_uid', 'dataset', 'cell_id', 'group_id', 'cycle_index', 'charge_end_time_s',
                            'soh', 'reference_id', 'measured_capacity_ah', 'rated_capacity_ah']
                    result = test[keep].copy()
                    result['prediction'] = fitted.predict(test[columns])
                    result['validation'] = protocol
                    result['variant'] = variant
                    result['model'] = name
                    result['selected_family'] = winner
                    result = causal_median(result, window=config['validation']['smoothing_points'])
                    predictions.append(result)
                    if name == winner:
                        selected = result.copy()
                        selected['model'] = 'training_selected'
                        predictions.append(selected)
                if (fold_n + 1) % 5 == 0 or fold_n + 1 == len(folds):
                    print(f'{variant} / {protocol}: {fold_n + 1}/{len(folds)} folds, {time.monotonic() - started:.0f}s', flush=True)
    result = pd.concat(predictions, ignore_index=True)
    assert not result.duplicated(['validation', 'variant', 'model', 'cycle_uid']).any()
    result.to_parquet(out / 'predictions.parquet', index=False)
    per_cell, per_dataset, overall = metric_tables(result)
    for name, table in [('metrics_per_cell', per_cell), ('metrics_per_dataset', per_dataset), ('metrics_overall', overall)]:
        table.to_csv(out / f'{name}.csv', index=False)
    write_json(out / 'fold_audit.json', audit)
    write_json(out / 'omissions.json', omissions)
    # This refit is for a future external battery. Dashboard accuracy uses only OOF predictions.
    columns = FEATURE_SETS['no_temperature']
    all_data = cycles.loc[cycles.core_model_eligible & cycles[columns].notna().all(axis=1)].copy()
    winner, selection = inner_selection(all_data, columns, config)
    fitted = fit_estimator(winner, all_data, columns, config)
    artifact = dict(estimator=fitted, features=columns, selected_model=winner, config=config,
                    feature_ranges={c: [float(all_data[c].min()), float(all_data[c].max())] for c in columns},
                    use='laboratory reference only; not calibrated on Baaz packs')
    joblib.dump(artifact, model_dir / 'lab_reference_model.joblib')
    write_json(model_dir / 'model_card.json', {k: v for k, v in artifact.items() if k != 'estimator'} | dict(selection=selection))
    write_json(out / 'run_metadata.json', dict(elapsed_s=time.monotonic() - started,
               feature_cache_fingerprint=json.loads((ROOT / 'data/processed/cache_metadata.json').read_text())['fingerprint'],
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'src/battery_analytics').glob('*.py')},
               hyperparameters='Fixed before evaluation. No test-cell tuning. A separate whole-cell inner validation split selects model family.',
               smoothing='Trailing median, current and past predictions only; no centred smoothing.', seed=config['seed']))
    print(overall.loc[overall.variant.eq('no_temperature'), ['validation', 'model', 'n_rows', 'n_cells', 'macro_cell_mae_raw_pp', 'macro_cell_mae_smoothed_pp']].to_string(index=False), flush=True)
    return result
