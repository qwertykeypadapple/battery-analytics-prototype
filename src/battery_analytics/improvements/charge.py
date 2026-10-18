"""New charge-only features. No discharge arrays, labels, or cell age accepted."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from battery_analytics.features import first_crossing
from battery_analytics.common import ROOT, load_config
from battery_analytics import loaders

Q_FEATURES = ['q_lower_middle', 'q_middle_upper_middle', 'q_upper_middle_upper', 'q_lower_upper']
ICA_FEATURES = ['ica_peak_v', 'ica_peak_height', 'ica_peak_area', 'ica_peak_present']


def window_arrays(time_s, voltage_v, current_a, rated_capacity_ah, config, dataset):
    t,v,i = [np.asarray(a,float) for a in [time_s,voltage_v,current_a]]
    good=np.isfinite(t)&np.isfinite(v)&np.isfinite(i)
    t,v,i=t[good],v[good],i[good]
    keep=np.r_[np.diff(t)>0,True]; t,v,i=t[keep],v[keep],i[keep]
    positive=np.flatnonzero(i>config['min_positive_current_c']*rated_capacity_ah)
    if not len(positive): raise ValueError('No positive charge')
    sl=slice(positive[0],positive[-1]+1); t,v,i=t[sl],v[sl],i[sl]
    smooth=pd.Series(v).rolling(config['voltage_smoothing_samples'],center=True,min_periods=1).median().to_numpy()
    levels=config['datasets'][dataset]
    crossings=[first_crossing(t,smooth,levels[x]) for x in ['lower_v','middle_v','upper_middle_v','upper_v']]
    if not np.isfinite(crossings).all() or np.any(np.diff(crossings)<=0): raise ValueError('Incomplete window')
    return t,smooth,i,crossings


def integral_between(t, i, lo, hi, nominal):
    use=(t>lo)&(t<hi)
    tt=np.r_[lo,t[use],hi]; ii=np.interp(tt,t,np.maximum(i,0))
    return float(np.trapz(ii,tt)/(3600*nominal))


def throughput(time_s, voltage_v, current_a, rated_capacity_ah, config, dataset):
    t,v,i,c=window_arrays(time_s,voltage_v,current_a,rated_capacity_ah,config,dataset)
    pairs=[(c[0],c[1]),(c[1],c[2]),(c[2],c[3]),(c[0],c[3])]
    out={}; fallback=[]
    for name,(lo,hi) in zip(Q_FEATURES,pairs):
        q=integral_between(t,i,lo,hi,rated_capacity_ah)
        if not np.isfinite(q):
            q=(hi-lo)*np.nanmedian(i[(t>=lo)&(t<=hi)])/(3600*rated_capacity_ah)
            fallback.append(name)
        out[name]=q
    return out | {'throughput_fallback':';'.join(fallback)}


def incremental_capacity(time_s, voltage_v, current_a, rated_capacity_ah, config, dataset):
    t,v,i,c=window_arrays(time_s,voltage_v,current_a,rated_capacity_ah,config,dataset)
    use=(t>c[0])&(t<c[-1]); tt=np.r_[c[0],t[use],c[-1]]
    ii=np.interp(tt,t,np.maximum(i,0)); vv=np.interp(tt,t,v)
    q=np.r_[0,np.cumsum(np.diff(tt)*(ii[:-1]+ii[1:])/2)]/(3600*rated_capacity_ah)
    vv=np.maximum.accumulate(vv); vv,indices=np.unique(vv,return_index=True); q=q[indices]
    grid=np.linspace(3.9,4.18,141)
    curve=savgol_filter(np.interp(grid,vv,q),11,2)
    dqdv=np.gradient(curve,grid)
    peaks,props=find_peaks(dqdv,prominence=.01,distance=10)
    valid=(peaks>=5)&(peaks<len(grid)-5)
    peaks=peaks[valid]; prominences=props['prominences'][valid]
    result=dict(ica_peak_v=np.nan,ica_peak_height=np.nan,ica_peak_area=np.nan,ica_peak_present=0.)
    if len(peaks):
        j=int(peaks[np.argmax(prominences)]); mask=np.abs(grid-grid[j])<=.020000001
        result.update(ica_peak_v=float(grid[j]),ica_peak_height=float(dqdv[j]),
                      ica_peak_area=float(np.trapz(dqdv[mask],grid[mask])),ica_peak_present=1.)
    return result


def build_features(eligible, kind='throughput'):
    config=load_config(); rows=[]; wanted=set(eligible.cycle_uid)
    extractor=throughput if kind=='throughput' else incremental_capacity
    for dataset in ['nasa','oxford','calce']:
        records, refs=getattr(loaders,dataset)(Path(config['data_root']),config)
        del refs
        for record in records:
            uid=record.metadata['cycle_uid']
            if uid not in wanted: continue
            row=extractor(record.time_s,record.voltage_v,record.current_a,
                    record.metadata['rated_capacity_ah'],config['features'],dataset)
            rows.append({'cycle_uid':uid,**row})
        del records
        print(f'{kind} charge parsing: {dataset} complete',flush=True)
    result=pd.DataFrame(rows)
    assert set(result.cycle_uid)==wanted
    out=ROOT/'data/improvements'; out.mkdir(exist_ok=True)
    result.to_parquet(out/f'{kind}_features.parquet',index=False)
    return result
