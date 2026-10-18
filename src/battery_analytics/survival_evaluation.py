"""Locked evaluation of survival probabilities, first alerts and fleet uses."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .improvements.core import folds
from .improvements.tracking import crossing_samples
from .fleet_v2 import PROTOCOLS, KINDS, SIZES, HORIZONS, absolute_bounds
from .survival import (History, SURVIVAL_METHODS, PROB_HORIZONS, ALL_HORIZONS, ALERT_HORIZONS,
                       ipcw, first_alert, event_class)
from .hybrid_fleet import evaluate_selection, ranking_order, tie_keys, summarize as hybrid_summary

METHODS = (*SURVIVAL_METHODS, 'linear', 'e6')


class Queries:
    def __init__(self, bundles, frozen, endpoints, e6, linear):
        self.bundles = bundles
        self.endpoints = endpoints.set_index('group_id')
        self.history = {(p, gid): History(g) for (p, gid), g in frozen.groupby(['validation', 'group_id'])}
        self.fold = {(f['validation'], gid): f['fold'] for f in folds() for gid in f['outer_test_cells']}
        self.e6 = {(p, gid): g.sort_values('cycle_index') for (p, gid), g in e6.groupby(['validation', 'group_id'])}
        self.linear = {(p, gid): g.sort_values('cycle_index') for (p, gid), g in linear.groupby(['validation', 'group_id'])}

    def get(self, protocol, gid, method, ages, horizon):
        ages = np.atleast_1d(ages).astype(float)
        key = (protocol, gid)
        out = np.full(len(ages), np.nan)
        if key not in self.fold:
            return out
        if method in SURVIVAL_METHODS:
            features, _ = self.history[key].at(ages)
            out = self.bundles[protocol, self.fold[key]].models[method].risk(ages, horizon, features)
        else:
            h = self.e6.get(key) if method == 'e6' else self.linear.get(key)
            if h is not None and len(h):
                x = h.cycle_index.to_numpy(float)
                idx = np.searchsorted(x, ages, side='right') - 1
                valid = idx >= 0
                if method == 'e6':
                    out[valid] = h[f'p_{horizon}'].to_numpy()[idx[valid]]
                else:
                    remaining = h.predicted_rul_cycles.to_numpy()[idx[valid]] + x[idx[valid]] - ages[valid]
                    good = np.isfinite(remaining)
                    out[np.flatnonzero(valid)[good]] = (remaining[good] <= horizon).astype(float)
        out[ages < self.endpoints.loc[gid, 'first_reference_cycle']] = np.nan
        return out


def comparators(root, out, queries_fold):
    states = pd.read_parquet(root / 'reports/improvements/E6_probabilistic/predictions.parquet')
    parts = []
    for _, g in states.groupby(['validation', 'group_id']):
        g = g.sort_values(['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last').copy()
        samples = crossing_samples(g)
        for h in ALL_HORIZONS:
            g[f'p_{h}'] = (samples <= h).mean(axis=1)
            if h != 60:
                assert np.allclose(g[f'p_{h}'], g[f'probability_{h}'])
        keep = ['validation', 'fold', 'dataset', 'cell_id', 'group_id', 'cycle_index']
        keep += [f'p_{h}' for h in ALL_HORIZONS] + [f'alert_threshold_{h}' for h in ALERT_HORIZONS]
        parts.append(g[keep])
    e6 = pd.concat(parts, ignore_index=True)
    e6.to_parquet(out / 'e6_horizon_extension.parquet', index=False)
    linear = pd.read_csv(root / 'reports/review/alarm_observations.csv')
    linear = linear[linear.signal.eq('raw')].rename(columns={'today_cycle': 'cycle_index'})
    linear['group_id'] = linear.dataset + ':' + linear.cell_id
    linear = linear.drop_duplicates(['validation', 'group_id', 'cycle_index'], keep='last').copy()
    linear['fold'] = [queries_fold.get((p, gid), '') for p, gid in zip(linear.validation, linear.group_id)]
    for h in ALL_HORIZONS:
        linear[f'p_{h}'] = np.where(linear.predicted_rul_cycles.notna(), (linear.predicted_rul_cycles <= h).astype(float), np.nan)
    return e6, linear


def probability_evaluation(root, out, bundles, endpoints, frozen, e6, linear):
    p = pd.read_parquet(out / 'outer_probabilities.parquet')
    fields = ['validation', 'fold', 'group_id', 'cycle_index', 'method'] + [f'p_{h}' for h in ALL_HORIZONS]
    p = pd.concat([p[fields], e6.assign(method='e6')[fields], linear.assign(method='linear')[fields]], ignore_index=True)
    assert not p.duplicated(['validation', 'group_id', 'method', 'cycle_index']).any()
    ei = endpoints.set_index('group_id')
    rows = []
    for (protocol, gid, method), g in p.groupby(['validation', 'group_id', 'method']):
        e = ei.loc[gid]
        if e.censoring not in ('observed_crossing', 'right_censored'):
            continue
        event = e.censoring == 'observed_crossing'
        stop = e.eol_cycle if event else e.last_reference_cycle
        g = g[g.cycle_index.ge(e.first_reference_cycle) & g.cycle_index.lt(stop)]
        if not len(g):
            continue
        fold = g.fold.iloc[0]
        censoring = bundles[protocol, fold].censoring
        ages = g.cycle_index.to_numpy(float)
        for h in PROB_HORIZONS:
            truth, weight, support = ipcw(censoring, ages, stop, event, h)
            probability = g[f'p_{h}'].to_numpy(float)
            available = np.isfinite(probability) & support
            contribution = np.where(np.isfinite(truth), weight * (probability - truth)**2, 0.)
            contribution[~available] = np.nan
            rows.append(pd.DataFrame(dict(validation=protocol, dataset=e.dataset, group_id=gid,
                method=method, cycle_index=ages, horizon_cycles=h, days=h//2, probability=probability,
                actual=truth, ipcw_weight=weight, censoring_supported=support, supported_probability=available,
                brier_contribution=contribution, known_outcome_error=(probability-truth)**2,
                source_has_observed_event=event)))
    r = pd.concat(rows, ignore_index=True)
    r.to_parquet(out / 'probability_scores.parquet', index=False)
    keys = ['validation', 'dataset', 'days', 'method']
    records = []
    for key, g in r.groupby(keys):
        cell = g.groupby('group_id').brier_contribution.mean().dropna()
        scored = g[g.brier_contribution.notna()]
        records.append(dict(zip(keys, key), real_cells_requested=g.group_id.nunique(), real_cells_scored=len(cell),
                            real_event_sources=scored[scored.source_has_observed_event].group_id.nunique(),
                            real_sources_with_horizon_incidents=scored[scored.actual.eq(1)].group_id.nunique(),
                            snapshots=len(g), predictions=int(g.probability.notna().sum()),
                            ipcw_supported_snapshots=int(g.supported_probability.sum()),
                            unknown_outcome_snapshots=int(scored.actual.isna().sum()),
                            ipcw_cell_macro_brier=cell.mean(), pooled_ipcw_brier=scored.brier_contribution.mean(),
                            conditional_known_cell_macro_brier=g.groupby('group_id').known_outcome_error.mean().mean()))
    pd.DataFrame(records).to_csv(out / 'probability_metrics.csv', index=False)
    paired = []
    for (protocol, dataset, days), g in r.groupby(['validation', 'dataset', 'days']):
        pivot = g.pivot(index=['group_id', 'cycle_index'], columns='method', values='brier_contribution')
        for other in METHODS:
            if other == 'hazard' or other not in pivot:
                continue
            both = pivot[['hazard', other]].dropna()
            bycell = both.groupby('group_id').mean()
            paired.append(dict(validation=protocol, dataset=dataset, days=days, baseline=other,
                               common_real_cells=len(bycell), common_snapshots=len(both),
                               hazard_brier=bycell.hazard.mean(), baseline_brier=bycell[other].mean(),
                               hazard_minus_baseline=(bycell.hazard-bycell[other]).mean()))
    pd.DataFrame(paired).to_csv(out / 'paired_brier.csv', index=False)
    calibration_rows = []
    for dataset in ['all', 'calce', 'nasa', 'oxford']:
        subset = r if dataset == 'all' else r[r.dataset.eq(dataset)]
        for (protocol, days, method), g in subset.groupby(['validation', 'days', 'method']):
            g = g[g.supported_probability].copy()
            g['bin'] = np.minimum(4, np.floor(g.probability * 5)).astype(int)
            for b in range(5):
                a = g[g.bin.eq(b)].copy()
                n = a.group_id.nunique()
                event_n = a[a.actual.eq(1)].group_id.nunique()
                cell_weight = 1 / a.groupby('group_id').group_id.transform('size') if len(a) else np.array([])
                weight = cell_weight * a.ipcw_weight
                total = float(weight.sum())
                calibration_rows.append(dict(dataset=dataset, validation=protocol, days=days, method=method,
                    bin=b, bin_low=b*.2, bin_high=(b+1)*.2, snapshots=len(a), unique_real_cells=n,
                    unique_horizon_event_cells=event_n, sufficiently_populated=n>=5 and event_n>=2,
                    predicted_mean=float((weight*a.probability).sum()/total) if total else np.nan,
                    observed_frequency=float((weight*a.actual.fillna(0)).sum()/total) if total else np.nan))
    pd.DataFrame(calibration_rows).to_csv(out / 'calibration.csv', index=False)
    return p


def alert_evaluation(root, out, all_native, endpoints, frozen):
    thresholds = {(a['validation'], a['fold'], a['method'], a['horizon_cycles']): a['threshold']
                  for a in json.loads((out / 'threshold_audit.json').read_text())}
    e6states = pd.read_parquet(out / 'e6_horizon_extension.parquet')
    frozen_events = pd.read_csv(root / 'reports/improvements/E6_probabilistic/alert_events.csv')
    frames = {(p, gid, m): g.sort_values('cycle_index') for (p, gid, m), g in all_native.groupby(['validation', 'group_id', 'method'])}
    old = {(r.validation, f'{r.dataset}:{r.cell_id}', 'e6' if r.method=='probabilistic' else 'linear', r.horizon_cycles): r
           for r in frozen_events.itertuples()}
    eligibility = set(frozen.group_id)
    event_records, censor_records = [], []
    for protocol in PROTOCOLS:
        for e in endpoints.itertuples():
            if e.censoring not in ('observed_crossing', 'right_censored'):
                continue
            for method in METHODS:
                g = frames.get((protocol, e.group_id, method))
                for h in ALERT_HORIZONS:
                    threshold = 1.01
                    if g is not None and len(g):
                        fold = g.fold.iloc[0]
                        if method == 'linear':
                            threshold = .5
                        elif method == 'e6':
                            es = e6states[e6states.validation.eq(protocol) & e6states.group_id.eq(e.group_id)]
                            threshold = float(es[f'alert_threshold_{h}'].iloc[0])
                        else:
                            threshold = thresholds[protocol, fold, method, h]
                    first = first_alert(g, h, threshold) if g is not None else np.nan
                    available = int(g[f'p_{h}'].notna().sum()) if g is not None else 0
                    shared = dict(validation=protocol, dataset=e.dataset, group_id=e.group_id, method=method,
                                  days=h//2, horizon_cycles=h, threshold=threshold, first_alert_cycle=first,
                                  available_native_points=available, no_usable_history=available==0,
                                  has_frozen_predictions=e.group_id in eligibility)
                    if e.censoring == 'observed_crossing':
                        category = event_class(first, e.eol_cycle, h)
                        if method in ('linear', 'e6'):
                            old_row = old[protocol, e.group_id, method, h]
                            assert (np.isnan(first) and np.isnan(old_row.first_alert_cycle)) or np.isclose(first, old_row.first_alert_cycle)
                            assert category == ('none' if old_row.outcome in ('none', 'no_eligible_history') else old_row.outcome)
                        opportunity = bool(g is not None and ((g.cycle_index < e.eol_cycle) &
                            (g.cycle_index >= e.eol_cycle-h) & g[f'p_{h}'].notna()).any())
                        event_records.append(dict(**shared, outcome=category, eol_cycle=e.eol_cycle,
                                                  lead_days=(e.eol_cycle-first)/2, timely_opportunity=opportunity))
                    else:
                        category = ('none' if not np.isfinite(first) else
                                    ('verified_false_alert' if first+h <= e.last_reference_cycle else 'censor_unresolved'))
                        censor_records.append(dict(**shared, outcome=category, last_reference_cycle=e.last_reference_cycle))
    events, censored = pd.DataFrame(event_records), pd.DataFrame(censor_records)
    events.to_csv(out / 'alert_events.csv', index=False)
    censored.to_csv(out / 'censored_alerts.csv', index=False)
    summaries = []
    for cohort in ['all_19_events', 'eligible_17_events']:
        full = events if cohort == 'all_19_events' else events[events.has_frozen_predictions]
        for dataset in ['all', 'calce', 'nasa', 'oxford']:
            a = full if dataset == 'all' else full[full.dataset.eq(dataset)]
            for (protocol, method, days), g in a.groupby(['validation', 'method', 'days']):
                summaries.append(dict(cohort=cohort, dataset=dataset, validation=protocol, method=method, days=days,
                                      real_events=len(g), timely=int(g.outcome.eq('timely').sum()),
                                      premature=int(g.outcome.eq('premature').sum()), late=int(g.outcome.eq('late').sum()),
                                      none=int(g.outcome.eq('none').sum()), no_history=int(g.no_usable_history.sum()),
                                      timely_opportunities=int(g.timely_opportunity.sum()), timely_rate=g.outcome.eq('timely').mean()))
    pd.DataFrame(summaries).to_csv(out / 'alert_metrics.csv', index=False)
    cs = censored.groupby(['validation', 'dataset', 'method', 'days', 'outcome']).size().unstack('outcome', fill_value=0).reset_index()
    cs.to_csv(out / 'censored_alert_metrics.csv', index=False)


def concordance_evaluation(out, queries, endpoints):
    eligible = endpoints[endpoints.censoring.isin(['observed_crossing', 'right_censored'])].copy()
    eligible['stop'] = np.where(eligible.censoring.eq('observed_crossing'), eligible.eol_cycle, eligible.last_reference_cycle)
    rows, summaries = [], []
    for protocol in PROTOCOLS:
        for dataset, group in eligible.groupby('dataset'):
            for horizon in PROB_HORIZONS:
                local = []
                for i in group[group.censoring.eq('observed_crossing')].itertuples():
                    today = i.eol_cycle-horizon
                    key_i = (protocol, i.group_id)
                    if today < i.first_reference_cycle or key_i not in queries.fold:
                        continue
                    censor = queries.bundles[protocol, queries.fold[key_i]].censoring
                    gt = censor.at([today])[0]
                    ratio = censor.at([i.eol_cycle], left=True)[0] / gt if gt > 0 else 0.
                    support = i.eol_cycle <= censor.max_time and ratio >= .05
                    weight = 1/ratio**2 if support else np.nan
                    for j in group[group.stop.gt(i.eol_cycle)].itertuples():
                        if today < j.first_reference_cycle or (protocol, j.group_id) not in queries.fold:
                            continue
                        for method in METHODS:
                            pi = queries.get(protocol, i.group_id, method, [today], horizon)[0]
                            pj = queries.get(protocol, j.group_id, method, [today], horizon)[0]
                            available = bool(support and np.isfinite(pi) and np.isfinite(pj))
                            score = (.5 if abs(pi-pj) <= 1e-8 else float(pi>pj)) if available else np.nan
                            h_i = queries.history[key_i]
                            _, last = h_i.at([today])
                            record = dict(validation=protocol, dataset=dataset, days=horizon//2, method=method,
                                          cell_i=i.group_id, cell_j=j.group_id, today=today, event_cycle=i.eol_cycle,
                                          p_i=pi, p_j=pj, censoring_supported=support, ipcw_pair_weight=weight,
                                          available=available, concordance=score, feature_staleness_i=today-last[0])
                            local.append(record); rows.append(record)
                for method in METHODS:
                    g = pd.DataFrame([r for r in local if r['method']==method])
                    a = g[g.available] if len(g) else g
                    summaries.append(dict(validation=protocol, dataset=dataset, days=horizon//2, method=method,
                        candidate_pairs=len(g), comparable_pairs=len(a),
                        real_cells=len(set(a.cell_i)|set(a.cell_j)) if len(a) else 0,
                        earlier_event_sources=a.cell_i.nunique() if len(a) else 0,
                        c_index=float((a.concordance*a.ipcw_pair_weight).sum()/a.ipcw_pair_weight.sum()) if len(a) else np.nan,
                        unweighted_c_index=float(a.concordance.mean()) if len(a) else np.nan))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out / 'concordance_pairs.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'concordance_metrics.csv', index=False)
    matched = []
    keys = ['validation', 'dataset', 'days']
    for key, g in pairs.groupby(keys):
        pivot = g.pivot(index=['cell_i', 'cell_j', 'today'], columns='method', values='concordance')
        weights = g.drop_duplicates(['cell_i', 'cell_j', 'today']).set_index(['cell_i', 'cell_j', 'today']).ipcw_pair_weight
        for other in METHODS:
            if other == 'hazard':
                continue
            common = pivot[['hazard', other]].dropna()
            w = weights.reindex(common.index)
            matched.append(dict(zip(keys, key), baseline=other, common_pairs=len(common),
                                common_earlier_events=len(set(common.index.get_level_values('cell_i'))),
                                hazard_c=float((common.hazard*w).sum()/w.sum()) if len(w) else np.nan,
                                baseline_c=float((common[other]*w).sum()/w.sum()) if len(w) else np.nan))
    pd.DataFrame(matched).to_csv(out / 'paired_concordance.csv', index=False)


def fleet_evaluation(root, out, queries):
    base = pd.read_parquet(root / 'reports/fleet_forecast_v2/copy_date_audit.parquet')
    probability_fields = [f'{method}_p_{h}' for method in SURVIVAL_METHODS for h in PROB_HORIZONS]
    for field in probability_fields:
        base[field] = np.nan
    base['last_charge_cycle_survival'] = np.nan
    for (protocol, gid), g in base.groupby(['validation', 'group_id'], sort=False):
        ages = g.source_age_cycles.to_numpy()
        for method in SURVIVAL_METHODS:
            for h in PROB_HORIZONS:
                base.loc[g.index, f'{method}_p_{h}'] = queries.get(protocol, gid, method, ages, h)
        if (protocol, gid) in queries.history:
            _, last = queries.history[protocol, gid].at(ages)
            base.loc[g.index, 'last_charge_cycle_survival'] = last
    save = ['kind', 'seed', 'copy_index', 'group_id', 'validation', 'today', 'source_age_cycles', 'last_charge_cycle_survival'] + probability_fields
    predictions = base[save]
    predictions.to_parquet(out / 'fleet_probabilities.parquet', index=False)
    old = pd.read_csv(root / 'reports/fleet_forecast_v2/scenarios.csv').set_index(['kind','seed','validation','nominal','today','days'])
    records = []
    for (kind, seed, protocol, today), master in base.groupby(['kind','seed','validation','today']):
        for size in SIZES:
            installed = master[master.copy_index<size]
            risk = installed[~installed.already_below]
            for days in HORIZONS:
                earlier = old.loc[kind,seed,protocol,size,today,days]
                y = risk[f'actual_{days}']
                lo, unknown = float(y.sum()), int(y.isna().sum())
                for method in SURVIVAL_METHODS:
                    p = risk[f'{method}_p_{days*2}']
                    matched = y.notna() & p.notna()
                    subtotal, missing = float(p.sum()), int(p.isna().sum())
                    err_lo, err_hi = subtotal-(lo+unknown), subtotal+missing-lo
                    abs_lo, abs_hi = absolute_bounds(err_lo,err_hi)
                    actual = float(y[matched].sum()) if matched.any() else np.nan
                    model_count = float(p[matched].sum()) if matched.any() else np.nan
                    history_count = float(earlier.historical_probability*matched.sum()) if matched.any() else np.nan
                    records.append(dict(kind=kind,seed=seed,validation=protocol,nominal=size,today=today,days=days,method=method,
                        unique_sources=installed.group_id.nunique(), at_risk=len(risk), at_risk_unique=risk.group_id.nunique(),
                        observed_incident_unique=risk.loc[y.eq(1),'group_id'].nunique(), already_below=int(installed.already_below.sum()),
                        actual_lower=lo,actual_upper=lo+unknown,unknown_actual=unknown, probability_subtotal=subtotal,
                        probability_missing=missing, probability_upper=subtotal+missing, usable_unique=risk.loc[p.notna(),'group_id'].nunique(),
                        full_signed_error_lower=err_lo,full_signed_error_upper=err_hi,full_abs_error_lower=abs_lo,full_abs_error_upper=abs_hi,
                        matched_n=int(matched.sum()),matched_unique=risk.loc[matched,'group_id'].nunique(), matched_actual=actual,
                        matched_model=model_count,matched_historical=history_count,model_error=model_count-actual,
                        historical_error=history_count-actual))
    r = pd.DataFrame(records)
    assert len(r)==28800
    r.to_csv(out/'fleet_scenarios.csv',index=False)
    summaries=[]
    for key,g in r.groupby(['validation','kind','nominal','days','method']):
        summaries.append(dict(zip(['validation','kind','nominal','days','method'],key),
            real_sources_min=g.unique_sources.min(),real_sources_max=g.unique_sources.max(),
            mean_at_risk=g.at_risk.mean(),mean_missing=g.probability_missing.mean(),mean_usable_unique=g.usable_unique.mean(),
            matched_scenarios=int(g.model_error.notna().sum()),mean_matched=g.matched_n.mean(),mean_matched_unique=g.matched_unique.mean(),
            model_mae=g.model_error.abs().mean(),historical_mae=g.historical_error.abs().mean(),
            model_signed_bias=g.model_error.mean(),historical_signed_bias=g.historical_error.mean()))
    pd.DataFrame(summaries).to_csv(out/'fleet_metrics.csv',index=False)
    return predictions


def hybrid_evaluation(root,out,predictions):
    old = pd.read_csv(root/'reports/hybrid_fleet/scenario_metrics.csv')
    age_rows = {(r['case_id'],r['k_mode']):r for r in old[old.method.eq('age')].to_dict('records')}
    case_rows = pd.read_csv(root/'reports/hybrid_fleet/case_audit.csv')
    key=['representation','kind','seed','validation','nominal','today','days']
    cases={tuple(r[k] for k in key):r for r in case_rows.to_dict('records')}
    audit = pd.read_parquet(root/'reports/hybrid_fleet/rank_audit.parquet')
    audit['lookup_kind']=np.where(audit.representation.eq('unique'),audit.dataset,audit.kind)
    probs=predictions.rename(columns={'kind':'lookup_kind'}).drop(columns=['source_age_cycles','last_charge_cycle_survival'])
    audit=audit.merge(probs,on=['lookup_kind','seed','copy_index','group_id','validation','today'],how='left',validate='many_to_one')
    source_bits=json.loads((root/'reports/hybrid_fleet/source_id_map.json').read_text())
    records=[]
    for cohort,g in audit.groupby(key[:-1]):
        representation,kind,seed,protocol,nominal,today=cohort
        ties=tie_keys(representation,kind,seed,g.unit_id)
        for days in HORIZONS:
            case=cases[(*cohort,days)]
            y=g[f'actual_{days}'].to_numpy()
            assert len(g)==case['at_risk']
            for method in SURVIVAL_METHODS:
                inputs=pd.DataFrame(dict(source_age_cycles=g.source_age_cycles.to_numpy(),charge_soh=-g[f'{method}_p_{days*2}'].to_numpy()))
                order,missing=ranking_order(inputs,'charge_soh',ties)
                for mode in ['historical','oracle']:
                    row=age_rows[case['case_id'],mode].copy()
                    row['method']=method
                    if row['k_available']:
                        row.update(evaluate_selection(y,g.group_id,order[:int(row['k'])],missing,source_bits))
                    else:
                        row['missing_scores']=int(missing.sum())
                        row['missing_unique']=g.loc[missing,'group_id'].nunique()
                    records.append(row)
    r=pd.DataFrame(records)
    assert len(r)==58560
    r.to_csv(out/'hybrid_scenarios.csv',index=False)
    combined=pd.concat([old,r],ignore_index=True)
    hybrid_summary(combined).to_csv(out/'hybrid_metrics.csv',index=False)
    pairs=[]
    for k,g in combined.groupby(['representation','validation','kind','nominal','days','k_mode']):
        for metric in ['precision','recall']:
            pivot=g.pivot(index='case_id',columns='method',values=metric)
            for method in SURVIVAL_METHODS:
                for other in ['charge_soh','charge_rul','cycle_soh','age','random']:
                    both=pivot[[method,other]].dropna()
                    pairs.append(dict(zip(['representation','validation','kind','nominal','days','k_mode'],k),
                        method=method,baseline=other,metric=metric,common_cases=len(both),
                        model_mean=both[method].mean(),baseline_mean=both[other].mean(),
                        difference=(both[method]-both[other]).mean()))
    pd.DataFrame(pairs).to_csv(out/'hybrid_paired.csv',index=False)


def evaluate(root, out, bundles, endpoints, frozen):
    mapping={(f['validation'],gid):f['fold'] for f in folds() for gid in f['outer_test_cells']}
    e6,linear=comparators(root,out,mapping)
    queries=Queries(bundles,frozen,endpoints,e6,linear)
    native=probability_evaluation(root,out,bundles,endpoints,frozen,e6,linear)
    print('Probability, IPCW Brier and sparse-bin calibration tables saved',flush=True)
    alert_evaluation(root,out,native,endpoints,frozen)
    print('All-19-event first alerts replayed; frozen linear/E6 results match',flush=True)
    concordance_evaluation(out,queries,endpoints)
    print('Event-landmark time-dependent concordance saved',flush=True)
    predictions=fleet_evaluation(root,out,queries)
    print('All v2 fleet probability-count scenarios saved',flush=True)
    hybrid_evaluation(root,out,predictions)
    print('Survival rankings replayed on saved unique and resampled hybrid cases',flush=True)
    from .survival_reporting import report
    report(root,out)
