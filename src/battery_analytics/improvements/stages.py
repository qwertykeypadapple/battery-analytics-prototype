from __future__ import annotations
import numpy as np
import pandas as pd
from battery_analytics.common import ROOT, write_json
from battery_analytics.features import CORE_FEATURES
from .core import OUT, BASE, META, data, baseline, folds, fit, macro_mae, prediction_frame, save, note, metric_rows
from .charge import Q_FEATURES, ICA_FEATURES, build_features

K=5
CYCLE=['cycle_index','cycle_squared']
REPRESENTATIONS={'raw':CORE_FEATURES, 'throughput':Q_FEATURES+['cc_current_c'],
                 'alongside':CORE_FEATURES+Q_FEATURES, 'delta':['delta_'+c for c in CORE_FEATURES]}


def relative_features(frame, columns=CORE_FEATURES, k=K):
    """Only entry charge FEATURES: no capacity, true SoH or future reference."""
    frame=frame.sort_values(['group_id','cycle_index','charge_end_time_s']).copy()
    for _,h in frame.groupby('group_id'):
        reference=h.iloc[:k][columns].median()
        for c in columns: frame.loc[h.index,'delta_'+c]=h[c]-reference[c]
    return frame


def enhanced_data(ica=False):
    d=data().merge(pd.read_parquet(ROOT/'data/improvements/throughput_features.parquet'),on='cycle_uid',validate='1:1')
    d=relative_features(d)
    if ica: d=d.merge(pd.read_parquet(ROOT/'data/improvements/ica_features.parquet'),on='cycle_uid',validate='1:1')
    return d


def run_fixed(name,d,columns,min_rank=0):
    results=[]; audit=[]
    for f in folds():
        train=d[d.group_id.isin(f['outer_train_cells']) & (d.entry_rank>=min_rank)]
        test=d[d.group_id.isin(f['outer_test_cells']) & (d.entry_rank>=min_rank)]
        if not len(test):
            audit.append(dict(fold=f['fold'],validation=f['validation'],reason='no_post_warmup_history',test_cells=f['outer_test_cells']))
            continue
        fitted=fit(train,columns)
        results.append(prediction_frame(test,fitted.predict(test[columns]),f,name))
        audit.append(dict(fold=f['fold'],validation=f['validation'],features=columns,
                train_cells=sorted(train.group_id.unique()),test_cells=f['outer_test_cells'],min_rank=min_rank))
    return save(name,pd.concat(results,ignore_index=True),audit)


def e2():
    d=data(); features=build_features(d,'throughput'); d=d.merge(features,on='cycle_uid',validate='1:1')
    features.throughput_fallback.value_counts(dropna=False).rename_axis('fallback_intervals').reset_index(name='n_rows').to_csv(OUT/'E2_fallbacks.csv',index=False)
    run_fixed('E2_replace',d,Q_FEATURES+['cc_current_c'])
    run_fixed('E2_alongside',d,CORE_FEATURES+Q_FEATURES)


def anchor_correction(predictions, entry_measurements, entry_predictions):
    """The caller explicitly supplies exactly one commissioning label per cell."""
    offsets=entry_measurements-entry_predictions
    return predictions+offsets


def e3():
    d=enhanced_data()
    run_fixed('E3_delta',d,REPRESENTATIONS['delta'],K)
    results=[]; audit=[]
    for f in folds():
        train=d[d.group_id.isin(f['outer_train_cells'])]
        test=d[d.group_id.isin(f['outer_test_cells'])]
        model=fit(train,CORE_FEATURES)
        p=pd.Series(model.predict(test[CORE_FEATURES]),index=test.index)
        first=test.groupby('group_id',sort=False).head(1)
        offsets=dict(zip(first.group_id,first.soh-p.loc[first.index]))
        after=test[test.entry_rank>0]
        if len(after):
            out=prediction_frame(after,p.loc[after.index]+after.group_id.map(offsets),f,'E3_anchor')
            out['entry_reference_soh']=after.group_id.map(dict(zip(first.group_id,first.soh)))
            results.append(out)
        audit.append(dict(fold=f['fold'],validation=f['validation'],train_cells=f['outer_train_cells'],
                test_cells=f['outer_test_cells'],entry_cycle_uids=first.cycle_uid.tolist(),
                allowed_label_count_per_test_cell=1,excluded='first eligible observation'))
    save('E3_anchor',pd.concat(results,ignore_index=True),audit)
    entries=d.groupby('group_id').head(1)[['group_id','soh']].rename(columns={'soh':'entry_reference_soh'})
    strata=[]
    for name in ['E3_delta','E3_anchor']:
        p=pd.read_parquet(OUT/name/'predictions.parquet').drop(columns='entry_reference_soh',errors='ignore').merge(entries,on='group_id')
        p['entry_band']=pd.cut(p.entry_reference_soh,[-np.inf,.7,.9,np.inf],labels=['already_below_70','70_to_90','at_least_90'],right=False)
        for band,g in p.groupby('entry_band',observed=True):
            m=metric_rows(g); m['entry_band']=str(band); strata.append(m)
    pd.concat(strata).to_csv(OUT/'E3_entry_health_metrics.csv',index=False)


def candidate_columns(name,ica=False):
    return REPRESENTATIONS[name]+CYCLE+(ICA_FEATURES if ica else [])


def choose(train,validation,ica=False,forced=False):
    options=[(name,False) for name in REPRESENTATIONS]
    if ica: options+= [(name,True) for name in REPRESENTATIONS]
    if forced: options=[x for x in options if x[1]]
    scores={}
    for name,add in options:
        cols=candidate_columns(name,add); model=fit(train,cols)
        scores[name+('_ica' if add else '')]=macro_mae(validation,model.predict(validation[cols]))
    best=min(scores.values())
    winner=next(k for k,v in scores.items() if v<=best+.20)
    return winner,scores


def columns_for(winner):
    return candidate_columns(winner.removesuffix('_ica'),winner.endswith('_ica'))


def combined(name='E4_combined',ica=False,forced=False):
    d=enhanced_data(ica); d=d[d.entry_rank>=K]
    result=[]; audit=[]
    for f in folds():
        it=d[d.group_id.isin(f['inner_train_cells'])]; iv=d[d.group_id.isin(f['inner_validation_cells'])]
        train=d[d.group_id.isin(f['outer_train_cells'])]; test=d[d.group_id.isin(f['outer_test_cells'])]
        if not len(test):
            audit.append(dict(validation=f['validation'],fold=f['fold'],reason='no_post_warmup_history',test_cells=f['outer_test_cells'])); continue
        winner,scores=choose(it,iv,ica,forced); cols=columns_for(winner)
        model=fit(train,cols)
        result.append(prediction_frame(test,model.predict(test[cols]),f,name))
        audit.append(dict(validation=f['validation'],fold=f['fold'],winner=winner,features=cols,inner_scores_pp=scores,
                    inner_train_cells=f['inner_train_cells'],inner_validation_cells=f['inner_validation_cells'],
                    outer_train_cells=f['outer_train_cells'],outer_test_cells=f['outer_test_cells']))
    return save(name,pd.concat(result,ignore_index=True),audit)


def e4(): return combined()


def e5():
    import json
    from .tracking import fit_noise,track
    d=enhanced_data(); d=d[d.entry_rank>=K]
    refs=pd.read_parquet(ROOT/'data/processed/capacity_references.parquet')
    choices=json.loads((OUT/'E4_combined/audit.json').read_text())
    byfold={(a['validation'],a['fold']):a for a in choices}
    results=[]; validation_states=[]; audit=[]
    for index,f in enumerate(folds()):
        chosen=byfold[f['validation'],f['fold']]
        if 'winner' not in chosen: continue
        cols=chosen['features']
        it=d[d.group_id.isin(f['inner_train_cells'])]; iv=d[d.group_id.isin(f['inner_validation_cells'])]
        tr=d[d.group_id.isin(f['outer_train_cells'])]; te=d[d.group_id.isin(f['outer_test_cells'])]
        parameters=fit_noise(it,cols,refs)
        inner_model=fit(it,cols); inner=track(iv,inner_model.predict(iv[cols]),parameters)
        inner['validation']=f['validation']; inner['fold']=f['fold']; validation_states.append(inner)
        model=fit(tr,cols); outer=track(te,model.predict(te[cols]),parameters)
        result=prediction_frame(te,outer.loc[te.index,'prediction'],f,'E5_filter')
        for col in outer.columns:
            if col.startswith(('lower_','upper_','variance_','covariance_','rate_per_','observation_')):
                result[col]=outer.loc[te.index,col]
        results.append(result)
        audit.append(dict(validation=f['validation'],fold=f['fold'],observation_features=cols,parameters=parameters,
            inner_train_cells=f['inner_train_cells'],inner_validation_cells=f['inner_validation_cells'],outer_test_cells=f['outer_test_cells']))
        print(f'E5 filter {index+1}/{len(choices)}',flush=True)
    p=save('E5_filter',pd.concat(results,ignore_index=True),audit)
    pd.concat(validation_states,ignore_index=True).to_parquet(OUT/'E5_filter/inner_validation_states.parquet',index=False)
    interval_metrics(p,OUT/'E5_filter/interval_calibration.csv')


def interval_metrics(p,path):
    rows=[]
    for (protocol,dataset),g in p.groupby(['validation','dataset']):
        for level in [50,80,95]:
            inside=(g.soh>=g[f'lower_{level}'])&(g.soh<=g[f'upper_{level}'])
            rows.append(dict(validation=protocol,dataset=dataset,nominal=level/100,n_rows=len(g),n_cells=g.cell_id.nunique(),
                pooled_coverage=float(inside.mean()),cell_macro_coverage=float(pd.DataFrame({'cell':g.cell_id,'ok':inside}).groupby('cell').ok.mean().mean()),
                mean_width_pp=float((g[f'upper_{level}']-g[f'lower_{level}']).mean()*100)))
    pd.DataFrame(rows).to_csv(path,index=False)
