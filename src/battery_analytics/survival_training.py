"""Nested charge cross-fitting and inner-only survival alert thresholds."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import numpy as np
import pandas as pd

from .features import FEATURE_SETS
from .models import inner_selection, fit_estimator
from .survival import (History, SURVIVAL_METHODS, ALL_HORIZONS, ALERT_HORIZONS, THRESHOLDS,
                       fit_bundle, first_alert, event_class)


class ChargeCrossfit:
    def __init__(self, data, config):
        columns = FEATURE_SETS['no_temperature']
        self.data = data[data.core_model_eligible & data[columns].notna().all(axis=1)].sort_values(
            ['group_id', 'cycle_index', 'charge_end_time_s']).copy()
        self.columns, self.config = columns, config
        self.cache, self.audit, self.predictions = OrderedDict(), [], []

    def predict(self, train_ids, test_ids, outer_test, protocol, fold, context):
        train_ids, test_ids, outer_test = sorted(train_ids), sorted(test_ids), set(outer_test)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(train_ids + test_ids).isdisjoint(outer_test)
        if protocol == 'leave_one_dataset_out':
            assert all(g.split(':', 1)[0] != fold for g in train_ids + test_ids)
        key = tuple(train_ids)
        if key not in self.cache:
            tr = self.data[self.data.group_id.isin(train_ids)]
            if len(train_ids) < 3:
                family, selection = 'linear_ridge', dict(inner_train_cells=train_ids, inner_validation_cells=[],
                                                        reason='fixed_small_pool_ridge')
            else:
                family, selection = inner_selection(tr, self.columns, self.config)
            model = fit_estimator(family, tr, self.columns, self.config)
            self.cache[key] = (model, family, selection)
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        model, family, selection = self.cache[key]
        self.cache.move_to_end(key)
        va = self.data[self.data.group_id.isin(test_ids)]
        out = va[['cycle_uid', 'dataset', 'cell_id', 'group_id', 'cycle_index', 'charge_end_time_s']].copy()
        out['prediction'] = model.predict(va[self.columns])
        self.audit.append(dict(validation=protocol, fold=fold, context=context, family=family,
                               train_cells=train_ids, test_cells=test_ids, outer_test_cells=sorted(outer_test),
                               selection=selection))
        self.predictions.append(out.assign(validation=protocol, fold=fold, context=context))
        return out

    def crossfit(self, pool, outer_test, protocol, fold, context):
        sides = {0: [], 1: []}
        for dataset in sorted({g.split(':', 1)[0] for g in pool}):
            ids = [g for g in pool if g.split(':', 1)[0] == dataset]
            ids.sort(key=lambda gid: hashlib.sha256(f'survival-charge-crossfit|42|{gid}'.encode()).hexdigest())
            for i, gid in enumerate(ids):
                sides[i % 2].append(gid)
        parts = [self.predict(sides[1-i], sides[i], outer_test, protocol, fold, f'{context}_{i}') for i in (0, 1)]
        return pd.concat(parts, ignore_index=True)


def histories(frame):
    return {gid: History(h) for gid, h in frame.groupby('group_id')}


def probability_frames(bundle, hist, group_ids):
    parts = []
    for gid in group_ids:
        h = hist.get(gid)
        if h is None:
            continue
        x, last = h.at(h.x)
        for name, model in bundle.models.items():
            out = pd.DataFrame(dict(group_id=gid, cycle_index=h.x, last_charge_cycle=last,
                                    charge_soh=x[:, 0], fade_rate=x[:, 1], method=name))
            for horizon in ALL_HORIZONS:
                out[f'p_{horizon}'] = model.risk(h.x, horizon, x)
            parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def choose_threshold(frame, endpoints, horizon):
    incident = endpoints[endpoints.censoring.eq('observed_crossing')]
    if not len(incident):
        return 1.01, {}, 'no_observed_inner_endpoint_abstain'
    scores = {}
    grouped = {gid: g for gid, g in frame.groupby('group_id')}
    for threshold in THRESHOLDS:
        costs = []
        for e in endpoints.itertuples():
            if e.censoring not in ('observed_crossing', 'right_censored'):
                continue
            h = grouped.get(e.group_id)
            first = first_alert(h, horizon, threshold) if h is not None and len(h) else np.nan
            if e.censoring == 'observed_crossing':
                category = event_class(first, e.eol_cycle, horizon)
                costs.append(dict(timely=0., premature=2., late=3., none=3.)[category])
            else:
                costs.append(2. if np.isfinite(first) and first + horizon <= e.last_reference_cycle else 0.)
        scores[str(threshold)] = float(np.mean(costs))
    best = min(scores.values())
    chosen = max(t for t in THRESHOLDS if scores[str(t)] == best)
    return chosen, scores, 'minimum_fixed_first_event_cost'


def fit_outer_folds(folds, data, frozen_predictions, endpoints, config, progress=print):
    crossfit = ChargeCrossfit(data, config)
    bundles, threshold_records, model_records, period_parts, outer_parts, inner_parts = {}, [], [], [], [], []
    for number, f in enumerate(folds, 1):
        protocol, fold = f['validation'], f['fold']
        outer_test = f['outer_test_cells']
        tr_ids, inner_ids, val_ids = f['outer_train_cells'], f['inner_train_cells'], f['inner_validation_cells']
        assert set(tr_ids).isdisjoint(outer_test)
        assert set(inner_ids).isdisjoint(val_ids) and set(inner_ids + val_ids) == set(tr_ids)
        outer_features = crossfit.crossfit(tr_ids, outer_test, protocol, fold, 'outer_training')
        inner_features = crossfit.crossfit(inner_ids, outer_test, protocol, fold, 'inner_training')
        inner_validation = crossfit.predict(inner_ids, val_ids, outer_test, protocol, fold, 'inner_validation')
        outer_bundle = fit_bundle(histories(outer_features), endpoints[endpoints.group_id.isin(tr_ids)])
        inner_bundle = fit_bundle(histories(inner_features), endpoints[endpoints.group_id.isin(inner_ids)])
        bundles[protocol, fold] = outer_bundle
        inner_p = probability_frames(inner_bundle, histories(inner_validation), val_ids)
        inner_parts.append(inner_p.assign(validation=protocol, fold=fold))
        for name in SURVIVAL_METHODS:
            p = inner_p[inner_p.method.eq(name)]
            for horizon in ALERT_HORIZONS:
                threshold, scores, reason = choose_threshold(p, endpoints[endpoints.group_id.isin(val_ids)], horizon)
                threshold_records.append(dict(validation=protocol, fold=fold, method=name, horizon_cycles=horizon,
                                              threshold=threshold, reason=reason, costs=scores,
                                              survival_training_cells=inner_ids, selection_cells=val_ids,
                                              outer_test_cells=outer_test))
        for context, bundle, ids in [('outer', outer_bundle, tr_ids), ('inner', inner_bundle, inner_ids)]:
            period_parts.append(bundle.periods.assign(validation=protocol, fold=fold, context=context))
            for name, model in bundle.models.items():
                eligible = endpoints[endpoints.group_id.isin(ids) & endpoints.censoring.isin(['observed_crossing', 'right_censored'])]
                record = dict(validation=protocol, fold=fold, context=context, method=name,
                              training_real_cells=len(eligible), training_observed_events=int(eligible.censoring.eq('observed_crossing').sum()),
                              period_cells=bundle.periods.group_id.nunique(), period_events=int(bundle.periods.event.sum()),
                              model_status=model.status, detail=getattr(model, 'detail', ''),
                              person_periods=len(bundle.periods), logistic_periods=int(bundle.periods.logistic_eligible.sum()))
                if name.startswith('hazard') and model.status == 'fitted':
                    record.update(coefficients=model.model.coef_[0].tolist(), intercept=float(model.model.intercept_[0]),
                                  scaler_mean=model.scaler.mean_.tolist(), scaler_scale=model.scaler.scale_.tolist())
                elif name == 'cox' and model.status in ('fitted', 'no_training_events'):
                    record.update(coefficients=model.beta.tolist(), scaler_mean=model.scaler.mean_.tolist(),
                                  scaler_scale=model.scaler.scale_.tolist())
                model_records.append(record)
        test = frozen_predictions[frozen_predictions.validation.eq(protocol) & frozen_predictions.group_id.isin(outer_test)]
        outer_p = probability_frames(outer_bundle, histories(test), outer_test)
        outer_parts.append(outer_p.assign(validation=protocol, fold=fold))
        progress(f'Fitted survival fold {number}/{len(folds)}: {protocol} / {fold}', flush=True)
    return dict(bundles=bundles, thresholds=threshold_records, models=model_records,
                periods=pd.concat(period_parts, ignore_index=True), outer=pd.concat(outer_parts, ignore_index=True),
                inner=pd.concat(inner_parts, ignore_index=True), charge_audit=crossfit.audit,
                charge_predictions=pd.concat(crossfit.predictions, ignore_index=True))
