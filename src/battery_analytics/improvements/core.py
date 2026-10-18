from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from battery_analytics.common import ROOT, write_json
from battery_analytics.evaluation import causal_median
from battery_analytics.features import CORE_FEATURES
from battery_analytics.models import cell_balanced_weights

OUT = ROOT / 'reports/improvements'
BASE = OUT / 'baseline'
META = ['cycle_uid', 'dataset', 'cell_id', 'group_id', 'cycle_index',
        'charge_end_time_s', 'soh', 'reference_id', 'rated_capacity_ah']


def verify_baseline():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    inputs=OUT/'frozen_inputs.json'
    if inputs.exists():
        for name,digest in json.loads(inputs.read_text())['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name


def baseline():
    p = pd.read_parquet(BASE / 'predictions.parquet')
    return p[p.model.eq('training_selected') & p.variant.eq('no_temperature')].copy()


def data():
    verify_baseline()
    d = pd.read_parquet(ROOT / 'data/processed/cycles.parquet')
    d = d[d.core_model_eligible].sort_values(['group_id', 'cycle_index', 'charge_end_time_s']).copy()
    d['entry_rank'] = d.groupby('group_id').cumcount()
    d['cycle_squared'] = d.cycle_index ** 2
    return d


def folds():
    return [x for x in json.loads((BASE / 'fold_audit.json').read_text()) if x['variant'] == 'no_temperature']


def fit(frame, columns):
    # Missing ICA peaks are explicit and imputed ONLY using training medians.
    model = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                      ('scale', StandardScaler()), ('ridge', Ridge(alpha=1.))])
    model.fit(frame[columns], frame.soh, ridge__sample_weight=cell_balanced_weights(frame))
    return model


def macro_mae(frame, predictions):
    return float(pd.DataFrame({'cell': frame.group_id.to_numpy(),
            'e': np.abs(np.asarray(predictions) - frame.soh.to_numpy()) * 100}).groupby('cell').e.mean().mean())


def prediction_frame(test, p, fold, experiment):
    r = test[META].copy()
    r['prediction'] = p
    r['validation'] = fold['validation']
    r['experiment'] = experiment
    r['fold'] = fold['fold']
    return r


def aggregate(g):
    e = (g.prediction - g.soh) * 100
    a = pd.DataFrame({'group_id': g.group_id, 'dataset': g.dataset, 'ae': e.abs(), 'se': e ** 2})
    cm = a.groupby(['dataset', 'group_id']).agg(ae=('ae', 'mean'), se=('se', 'mean')).reset_index()
    return dict(n_rows=len(g), n_cells=g.group_id.nunique(),
                cell_macro_mae_pp=float(cm.ae.mean()), cell_macro_rmse_pp=float(np.sqrt(cm.se).mean()),
                pooled_mae_pp=float(e.abs().mean()), pooled_rmse_pp=float(np.sqrt((e ** 2).mean())),
                equal_dataset_mae_pp=float(cm.groupby('dataset').ae.mean().mean()))


def metric_rows(pred):
    result = []
    for (experiment, protocol), g in pred.groupby(['experiment', 'validation']):
        for signal, field in [('raw', 'prediction'), ('median', 'prediction_smoothed')]:
            h = g.copy()
            h['prediction'] = h[field]
            h = h[np.isfinite(h.prediction)]
            for level, cols in [('overall', []), ('dataset', ['dataset']), ('cell', ['dataset', 'cell_id'])]:
                groups = [((), h)] if not cols else h.groupby(cols)
                for key, subset in groups:
                    if not isinstance(key, tuple): key = (key,)
                    if len(subset):
                        result.append(dict(experiment=experiment, validation=protocol, signal=signal, level=level,
                                           **dict(zip(cols, key)), **aggregate(subset)))
            for dataset, d in h.groupby('dataset'):
                for band, mask in [('below_70', d.soh < .70), ('70_to_80', (d.soh >= .70) & (d.soh < .80))]:
                    subset = d[mask]
                    if len(subset): result.append(dict(experiment=experiment, validation=protocol, signal=signal,
                            level='band', dataset=dataset, band=band, **aggregate(subset)))
    return pd.DataFrame(result)


def save(experiment, pred, audit=None):
    out = OUT / experiment
    out.mkdir(parents=True, exist_ok=True)
    if 'prediction_smoothed' not in pred:
        pred = pd.concat([causal_median(g) for _, g in pred.groupby('validation')], ignore_index=True)
    assert not pred.duplicated(['validation', 'cycle_uid']).any()
    pred.to_parquet(out / 'predictions.parquet', index=False)
    metrics = metric_rows(pred)
    metrics.to_csv(out / 'metrics.csv', index=False)
    b = baseline()
    matched = b.merge(pred[['validation', 'cycle_uid']], on=['validation', 'cycle_uid'])
    matched['experiment'] = experiment + '_matched_baseline'
    metric_rows(matched).to_csv(out / 'matched_baseline_metrics.csv', index=False)
    expected = b.groupby(['validation', 'dataset', 'cell_id']).size().rename('baseline_rows')
    actual = pred.groupby(['validation', 'dataset', 'cell_id']).prediction.count().rename('predicted_rows')
    cov = expected.to_frame().join(actual).fillna({'predicted_rows': 0}).reset_index()
    cov['row_coverage'] = cov.predicted_rows / cov.baseline_rows
    cov.to_csv(out / 'coverage.csv', index=False)
    if audit is not None: write_json(out / 'audit.json', audit)
    print(metrics[(metrics.level == 'overall') & metrics.signal.eq('raw')].to_string(index=False), flush=True)
    return pred


def note(experiment, text):
    p = OUT / 'progress.md'
    with p.open('a') as f: f.write(f'\n## {experiment}\n\n{text}\n')


def e0():
    out = OUT / 'E0'; out.mkdir(exist_ok=True)
    p = baseline(); p['experiment'] = 'baseline'
    counts = p.groupby(['validation', 'group_id']).cycle_uid.transform('size')
    sensitivity = p[counts >= 20].copy(); sensitivity['experiment'] = 'E0_min20_posthoc'
    metric_rows(sensitivity).to_csv(out / 'minimum_history_metrics.csv', index=False)
    p.loc[counts < 20, ['validation', 'dataset', 'cell_id']].drop_duplicates().to_csv(out / 'excluded_cells.csv', index=False)
    c = data()[['cycle_uid', 'ambient_temp_c']]
    n = p.merge(c, on='cycle_uid'); n = n[n.dataset.eq('nasa') & (n.soh < .7)].copy()
    n['temperature_group'] = pd.cut(n.ambient_temp_c, [-np.inf, 10, 35, np.inf], right=False, labels=['cold', 'room', 'warm'])
    n['error_pp'] = (n.prediction - n.soh) * 100
    rows = []
    for key, h in n.groupby(['validation', 'temperature_group'], observed=True):
        rows.append(dict(validation=key[0], temperature_group=key[1], **aggregate(h)))
    pd.DataFrame(rows).to_csv(out / 'nasa_low_soh_temperature.csv', index=False)
    n.groupby(['validation', 'temperature_group', 'cell_id'], observed=True).agg(
        n_rows=('error_pp', 'size'), mae_pp=('error_pp', lambda x: x.abs().mean()),
        mean_reference_soh=('soh', 'mean'), ambient_c=('ambient_temp_c', 'median')).reset_index().to_csv(out / 'nasa_temperature_per_cell.csv', index=False)
    print(pd.read_csv(out / 'minimum_history_metrics.csv').query("level == 'overall' and signal == 'raw'").to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    note('E0', 'Post-hoc minimum-history sensitivity and fixed ambient groups saved. All baseline cells remain in subsequent unrestricted evaluations.')


def e1():
    d = data(); output = []; audits = []
    cols = ['cycle_index', 'cycle_squared']
    for f in folds():
        train = d[d.group_id.isin(f['outer_train_cells'])]
        test = d[d.group_id.isin(f['outer_test_cells'])]
        model = fit(train, cols)
        output.append(prediction_frame(test, model.predict(test[cols]), f, 'E1_cycle'))
        audits.append(dict(fold=f['fold'], validation=f['validation'], features=cols,
                train_cells=f['outer_train_cells'], test_cells=f['outer_test_cells'], fixed='Ridge alpha1; no tuning'))
    return save('E1_cycle', pd.concat(output, ignore_index=True), audits)
