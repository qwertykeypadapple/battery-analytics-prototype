from __future__ import annotations
import numpy as np
import pandas as pd


def causal_median(frame, value='prediction', window=5):
    """A trailing filter. Current and past predictions only, never future labels."""
    result = frame.sort_values(['dataset', 'cell_id', 'cycle_index', 'charge_end_time_s']).copy()
    result['prediction_smoothed'] = result.groupby(['dataset', 'cell_id'], sort=False)[value].transform(
        lambda s: s.rolling(window, min_periods=1).median())
    return result


def errors(y, p):
    error = (np.asarray(p) - np.asarray(y)) * 100
    return dict(mae_pp=float(np.abs(error).mean()), rmse_pp=float(np.sqrt(np.mean(error ** 2))))


def metric_record(g):
    result = dict(n_rows=len(g), n_cells=g.group_id.nunique())
    for col, suffix in [('prediction', 'raw'), ('prediction_smoothed', 'smoothed')]:
        result.update({f'{k.removesuffix("_pp")}_{suffix}_pp': v for k, v in errors(g.soh, g[col]).items()})
    return result


def metric_tables(predictions):
    base = ['validation', 'variant', 'model']
    cell_records, dataset_records, overall_records = [], [], []
    for key, g in predictions.groupby(base + ['dataset', 'cell_id'], sort=True):
        cell_records.append(dict(zip(base + ['dataset', 'cell_id'], key), **metric_record(g)))
    per_cell = pd.DataFrame(cell_records)
    for key, g in predictions.groupby(base + ['dataset'], sort=True):
        r = dict(zip(base + ['dataset'], key), **metric_record(g))
        cm = per_cell
        for col, val in zip(base + ['dataset'], key):
            cm = cm[cm[col].eq(val)]
        for col in ['mae_raw_pp', 'rmse_raw_pp', 'mae_smoothed_pp', 'rmse_smoothed_pp']:
            r['macro_cell_' + col] = float(cm[col].mean())
        dataset_records.append(r)
    per_dataset = pd.DataFrame(dataset_records)
    for key, g in predictions.groupby(base, sort=True):
        r = dict(zip(base, key), **metric_record(g))
        cm, dm = per_cell, per_dataset
        for col, val in zip(base, key):
            cm, dm = cm[cm[col].eq(val)], dm[dm[col].eq(val)]
        for col in ['mae_raw_pp', 'rmse_raw_pp', 'mae_smoothed_pp', 'rmse_smoothed_pp']:
            r['macro_cell_' + col] = float(cm[col].mean())
            r['macro_dataset_' + col] = float(dm['macro_cell_' + col].mean())
        overall_records.append(r)
    return per_cell, per_dataset, pd.DataFrame(overall_records)
